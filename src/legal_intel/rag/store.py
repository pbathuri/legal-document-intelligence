from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from legal_intel.config import get_settings
from legal_intel.rag.embeddings import make_embedding_model


@lru_cache(maxsize=1)
def _shared_memory_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _vector_size_from_collection(info: Any) -> int | None:
    params = info.config.params.vectors
    if params is None:
        return None
    if isinstance(params, qm.VectorParams):
        return int(params.size)
    if isinstance(params, dict):
        # Named vectors: take first
        for v in params.values():
            if isinstance(v, qm.VectorParams):
                return int(v.size)
            if isinstance(v, dict) and "size" in v:
                return int(v["size"])
    return None


class LegalVectorStore:
    def __init__(self, embedding: Any | None = None, client: QdrantClient | None = None) -> None:
        s = get_settings()
        self._emb = embedding or make_embedding_model()
        self._collection = s.qdrant_collection
        if client is not None:
            self._client = client
        else:
            if s.qdrant_url == ":memory:":
                self._client = _shared_memory_client()
            else:
                self._client = QdrantClient(url=s.qdrant_url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        dim = self._emb.dimension
        name = self._collection
        if self._client.collection_exists(name):
            info = self._client.get_collection(name)
            existing = _vector_size_from_collection(info)
            if existing is not None and existing != dim:
                self._client.delete_collection(name)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def upsert_document_chunks(
        self, *, doc_id: str, doc_label: str, chunks: list[tuple[str, dict[str, Any]]]
    ) -> int:
        if not chunks:
            return 0
        texts = [c[0] for c in chunks]
        vectors = self._emb.encode(texts)
        points: list[qm.PointStruct] = []
        for i, ((text, extra), vec) in enumerate(zip(chunks, vectors, strict=True)):
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}"))
            payload = {
                "doc_id": doc_id,
                "doc_label": doc_label,
                "chunk_index": i,
                "text": text,
                **extra,
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query: str, limit: int, doc_id: str | None = None) -> list[dict[str, Any]]:
        s = get_settings()
        fetch_limit = limit
        if s.rerank_enabled:
            fetch_limit = min(256, max(limit, limit * s.retrieval_rerank_multiplier))

        vec = self._emb.encode([query])[0]
        q_filter = None
        if doc_id:
            q_filter = qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            )
        res = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=fetch_limit,
            query_filter=q_filter,
            with_payload=True,
        )
        out: list[dict[str, Any]] = []
        for p in res.points:
            pl = dict(p.payload or {})
            pl["score"] = getattr(p, "score", None)
            out.append(pl)

        if s.rerank_enabled and out:
            from legal_intel.rag.reranker import rerank_hits

            out = rerank_hits(query, out, top_k=limit)
        elif len(out) > limit:
            out = out[:limit]

        return out

    def scroll_chunks_for_document(
        self,
        doc_id: str,
        *,
        limit: int = 64,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Paginated payload scan for one logical document (debugging / inspection)."""
        flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        lim = max(1, min(limit, 256))
        offset = cursor if cursor else None
        records, next_off = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=lim,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        rows: list[dict[str, Any]] = []
        for r in records:
            pl = dict(r.payload or {})
            pl["point_id"] = str(r.id)
            rows.append(pl)
        next_cursor = None if next_off is None else str(next_off)
        return rows, next_cursor

    def list_chunk_texts(self, doc_id: str, *, max_chunks: int = 48) -> list[str]:
        """Collect chunk payloads for one logical document (bounded scroll — device-local index)."""
        did = (doc_id or "").strip()
        if not did:
            return []
        cap = max(1, min(max_chunks, 256))
        texts: list[str] = []
        cursor = None
        while len(texts) < cap:
            batch = min(64, cap - len(texts))
            rows, cursor = self.scroll_chunks_for_document(did, limit=batch, cursor=cursor)
            if not rows:
                break
            for r in rows:
                t = (r.get("text") or "").strip()
                if t:
                    texts.append(t)
                if len(texts) >= cap:
                    break
            if cursor is None:
                break
        return texts[:cap]

    def aggregate_indexed_documents(self, *, max_points: int = 4000) -> list[dict[str, Any]]:
        """Distinct doc_id values seen in the collection (bounded scan)."""
        cap = max(50, min(max_points, 50_000))
        counts: dict[str, dict[str, Any]] = {}
        offset = None
        scanned = 0
        while scanned < cap:
            batch = min(512, cap - scanned)
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for r in records:
                scanned += 1
                pl = r.payload or {}
                did = pl.get("doc_id")
                if not did:
                    continue
                ent = counts.setdefault(
                    str(did),
                    {"doc_id": str(did), "doc_label": pl.get("doc_label"), "chunk_count": 0},
                )
                ent["chunk_count"] = int(ent["chunk_count"]) + 1
                if pl.get("doc_label"):
                    ent["doc_label"] = pl["doc_label"]
            if offset is None:
                break
        return sorted(counts.values(), key=lambda x: str(x["doc_id"]))

    def delete_document_vectors(self, doc_id: str) -> int:
        """Remove all chunks for doc_id; returns prior point count (exact filter count)."""
        flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        try:
            cnt = self._client.count(
                collection_name=self._collection,
                count_filter=flt,
                exact=True,
            )
            n = int(getattr(cnt, "count", cnt))
        except Exception:
            n = 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=flt),
        )
        return n

    def delete_document_vectors_batch(self, doc_ids: list[str]) -> dict[str, int]:
        """Remove vectors for many logical documents; returns prior counts per id."""
        out: dict[str, int] = {}
        for raw in doc_ids:
            did = (raw or "").strip()
            if not did:
                continue
            out[did] = self.delete_document_vectors(did)
        return out

    def clear_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._ensure_collection()
