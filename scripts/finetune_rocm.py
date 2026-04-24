#!/usr/bin/env python3
"""Fine-tune Llama 3.1 8B on AMD MI300X using LoRA + ROCm.

This script is designed to run on an AMD Developer Cloud MI300X instance.
Prerequisites:
  - ROCm 7.x installed
  - PyTorch with ROCm support
  - pip install transformers peft trl datasets accelerate bitsandbytes

Usage:
  python scripts/finetune_rocm.py \
    --train-file data/training/train.jsonl \
    --val-file data/training/val.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir models/legal-extraction-lora \
    --epochs 3 \
    --batch-size 4

NOTE: For the hackathon, use 8B model for fast iteration.
      For production, fine-tune 70B with LoRA on single MI300X (fits in 192GB).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """Verify AMD GPU is available via ROCm."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info("GPU: %s (%.1f GB)", device_name, mem_gb)
            return True
        else:
            logger.warning("No GPU detected. ROCm maps AMD GPUs through torch.cuda API.")
            return False
    except ImportError:
        logger.error("PyTorch not installed")
        return False


def load_dataset(train_path: str, val_path: str):
    """Load JSONL datasets."""
    from datasets import Dataset
    
    def _load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    
    train_data = _load_jsonl(train_path)
    val_data = _load_jsonl(val_path)
    
    # Convert to HuggingFace Dataset with chat format
    def format_for_sft(record: dict) -> dict:
        output = record.get("output", "")
        if isinstance(output, dict):
            output = json.dumps(output, ensure_ascii=False)
        
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an expert in Indian property document analysis.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{record.get('instruction', '')}\n\n{record.get('input', '')}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )
        return {"text": text}
    
    train_ds = Dataset.from_list([format_for_sft(r) for r in train_data])
    val_ds = Dataset.from_list([format_for_sft(r) for r in val_data])
    
    logger.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))
    return train_ds, val_ds


def train(
    model_name: str,
    train_ds,
    val_ds,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_len: int = 4096,
):
    """Fine-tune with LoRA using TRL's SFTTrainer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    
    logger.info("Loading model: %s", model_name)
    
    # Quantization config for fitting in memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # LoRA config targeting attention layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save LoRA adapter
    adapter_dir = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter saved to %s", adapter_dir)
    
    return str(adapter_dir)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune on AMD MI300X")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output-dir", default="models/legal-extraction-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()
    
    has_gpu = check_gpu()
    if not has_gpu:
        logger.warning("No GPU found. Training will be very slow on CPU.")
    
    train_ds, val_ds = load_dataset(args.train_file, args.val_file)
    adapter_path = train(
        model_name=args.model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
    )
    print(f"Training complete. Adapter: {adapter_path}")


if __name__ == "__main__":
    main()
