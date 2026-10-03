"""Unit tests for the Sovereign PII / PHI / MNPI Obfuscator."""

from __future__ import annotations

import pytest

from olaya.pii import ObfuscationMode, PIIObfuscator


@pytest.fixture
def pii_engine() -> PIIObfuscator:
    return PIIObfuscator(salt="test-salt-12345")


def test_email_and_phone_pseudonymization(pii_engine: PIIObfuscator):
    text = "Please reach out to CEO John Smith at john.smith@acme-corp.com or call 415-555-0199."
    res = pii_engine.obfuscate(text, mode=ObfuscationMode.PSEUDONYMIZE)

    assert "john.smith@acme-corp.com" not in res.sanitized_text
    assert "415-555-0199" not in res.sanitized_text
    assert "[EMAIL_1]" in res.sanitized_text
    assert "[PHONE_1]" in res.sanitized_text
    assert "[PERSON_NAME_1]" in res.sanitized_text
    assert res.risk_score > 0.5
    assert res.latency_us < 2000.0  # < 2ms


def test_referential_integrity(pii_engine: PIIObfuscator):
    text = (
        "Contact CEO John Smith at john.smith@acme.com. "
        "I repeat, send the files to CEO John Smith via john.smith@acme.com."
    )
    res = pii_engine.obfuscate(text, mode=ObfuscationMode.PSEUDONYMIZE)

    # Count occurrences: same pseudonym should be used for both mentions
    assert res.sanitized_text.count("[PERSON_NAME_1]") == 2
    assert res.sanitized_text.count("[EMAIL_1]") == 2
    assert "[PERSON_NAME_2]" not in res.sanitized_text
    assert "[EMAIL_2]" not in res.sanitized_text


def test_reversible_deanonymization(pii_engine: PIIObfuscator):
    original = (
        "CONFIDENTIAL: Project Falcon target company is PetArmor, led by Dr. Robert Miller. "
        "SSN: 123-45-6789. Email: robert.miller@petarmor.com. Address: 742 Evergreen Terrace. "
        "Card: 4532-1234-5678-9012."
    )
    res = pii_engine.obfuscate(
        original,
        mode=ObfuscationMode.PSEUDONYMIZE,
        custom_entities=["PetArmor"],
        deal_codenames=["Project Falcon"],
    )

    # Ensure none of the sensitive values exist in sanitized output
    assert "Project Falcon" not in res.sanitized_text
    assert "PetArmor" not in res.sanitized_text
    assert "123-45-6789" not in res.sanitized_text
    assert "robert.miller@petarmor.com" not in res.sanitized_text
    assert "742 Evergreen Terrace" not in res.sanitized_text

    # Reversible de-anonymization using the local vault
    restored = pii_engine.deanonymize(res.sanitized_text, res.vault)
    assert restored == original


def test_redact_mode(pii_engine: PIIObfuscator):
    text = "Server IP is 192.168.1.105 with contact user@internal.local"
    res = pii_engine.obfuscate(text, mode=ObfuscationMode.REDACT)

    assert "[REDACTED_IP_ADDRESS]" in res.sanitized_text
    assert "[REDACTED_EMAIL]" in res.sanitized_text
    assert "192.168.1.105" not in res.sanitized_text


def test_mask_mode(pii_engine: PIIObfuscator):
    text = "Card 4111-2222-3333-4444 and SSN 987-65-4321 with email test@gmail.com"
    res = pii_engine.obfuscate(text, mode=ObfuscationMode.MASK)

    assert "****-****-****-4444" in res.sanitized_text
    assert "***-**-4321" in res.sanitized_text
    assert "t***@gmail.com" in res.sanitized_text


def test_hash_mode(pii_engine: PIIObfuscator):
    text = "CEO John Smith authorized Project Titan"
    res1 = pii_engine.obfuscate(text, mode=ObfuscationMode.HASH)
    res2 = pii_engine.obfuscate(text, mode=ObfuscationMode.HASH)

    # Salted hash should be deterministic
    assert res1.sanitized_text == res2.sanitized_text
    assert "[DEAL_CODENAME_" in res1.sanitized_text


def test_empty_and_clean_text(pii_engine: PIIObfuscator):
    text = "The quick brown fox jumps over the lazy dog."
    res = pii_engine.obfuscate(text)
    assert res.sanitized_text == text
    assert len(res.entities) == 0
    assert res.risk_score == 0.0
