from legal_intel.privacy import redact_all


def test_redact_aadhaar():
    text = "Aadhaar: 1234 5678 9012"
    assert "REDACTED_AADHAAR" in redact_all(text)


def test_redact_pan():
    text = "PAN: ABCDE1234F"
    assert "REDACTED_PAN" in redact_all(text)


def test_redact_phone():
    text = "Phone: 9876543210"
    assert "REDACTED_PHONE" in redact_all(text)


def test_no_false_positive():
    text = "This is a normal sentence with no PII."
    assert redact_all(text) == text
