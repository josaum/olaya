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
        "SSN: 321-54-9876. Email: robert.miller@petarmor.com. Address: 742 Evergreen Terrace. "
        "Card: 4532-0151-1283-0366."
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
    assert "321-54-9876" not in res.sanitized_text
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
    text = "Card 4532-0151-1283-0366 and SSN 321-54-9876 with email test@gmail.com"
    res = pii_engine.obfuscate(text, mode=ObfuscationMode.MASK)

    assert "****-****-****-0366" in res.sanitized_text
    assert "***-**-9876" in res.sanitized_text
    assert "t***@gmail.com" in res.sanitized_text
    assert res.vault == {} and res.reversible is False


def test_candidate_filter_decides_after_the_regex_proposes():
    text = "card 4532-1234-5678-9012 and card 4532-0151-1283-0366"

    def luhn(entity_type: str, value: str) -> bool:
        if entity_type != "CREDIT_CARD":
            return True
        digits = [int(c) for c in value if c.isdigit()]
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d = d * 2 - 9 if d * 2 > 9 else d * 2
            total += d
        return total % 10 == 0

    res = PIIObfuscator(salt="s", candidate_filter=luhn).obfuscate(text)
    assert "4532-1234-5678-9012" in res.sanitized_text
    assert "4532-0151-1283-0366" not in res.sanitized_text
    assert [e.entity_type for e in res.entities] == ["CREDIT_CARD"]


def test_deanonymize_is_single_pass(pii_engine: PIIObfuscator):
    vault = {"[A_1]": "[A_2]", "[A_2]": "second"}
    assert pii_engine.deanonymize("[A_1] [A_2]", vault) == "[A_2] second"


def test_default_salt_is_env_or_random(monkeypatch):
    monkeypatch.setenv("OLAYA_PII_SALT", "pinned")
    assert PIIObfuscator().salt == "pinned"
    monkeypatch.delenv("OLAYA_PII_SALT")
    assert PIIObfuscator().salt != PIIObfuscator().salt


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


def test_sovereign_reason_loop(pii_engine: PIIObfuscator):
    prompt = "Review transaction for CEO Alice Adams (alice@corp.com) regarding Project Apex acquisition of BetaCo."

    def mock_llm(sanitized_input: str) -> str:
        # Assert LLM never sees real names
        assert "Alice Adams" not in sanitized_input
        assert "alice@corp.com" not in sanitized_input
        assert "Project Apex" not in sanitized_input
        assert "BetaCo" not in sanitized_input
        # Model responds referencing the surrogate tokens
        return "Approved deal for [PERSON_NAME_1] ([EMAIL_1]) under [DEAL_CODENAME_1] targeting [CUSTOM_ENTITY_1]."

    result = pii_engine.sovereign_reason(
        prompt=prompt,
        custom_entities=["BetaCo"],
        deal_codenames=["Project Apex"],
        llm_caller=mock_llm,
    )

    assert result.zero_leak_verified is True
    assert result.entities_protected >= 4
    # The output returned to the caller should have real entities restored!
    assert "Alice Adams" in result.response
    assert "alice@corp.com" in result.response
    assert "Project Apex" in result.response
    assert "BetaCo" in result.response
    assert "[PERSON_NAME_1]" not in result.response
    d = result.to_dict()
    assert d["zero_leak_verified"] is True
