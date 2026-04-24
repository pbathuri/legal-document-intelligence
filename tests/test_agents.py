"""Test agent tool registry."""
from legal_intel.agents.tools import TOOL_REGISTRY, run_tool, get_tool_descriptions

def test_tools_registered():
    assert "check_disputes" in TOOL_REGISTRY
    assert "fetch_ec_data" in TOOL_REGISTRY
    assert "search_registrations" in TOOL_REGISTRY
    assert "redact_pii" in TOOL_REGISTRY

def test_redact_tool():
    result = run_tool("redact_pii", text="Call me at 9876543210 or Aadhaar 1234 5678 9012")
    assert "REDACTED" in result

def test_tool_descriptions():
    descs = get_tool_descriptions()
    assert len(descs) >= 4
    for d in descs:
        assert "name" in d
        assert "description" in d

def test_unknown_tool():
    result = run_tool("nonexistent_tool")
    assert "error" in result
