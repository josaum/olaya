"""Sovereign PII / PHI / MNPI Obfuscation and De-identification Engine for O-Laya (Leio).

Provides on-demand, air-gapped, zero-egress de-identification for sensitive data:
- Personal Identifiable Information (PII): Names, emails, phones, SSNs, credit cards, IPs, addresses
- Protected Health Information (PHI): HIPAA 18 Safe Harbor identifiers
- Material Non-Public Information (MNPI): Deal code names, unannounced acquisition targets, executive names

Modes:
- 'redact': Replace matches with semantic bracketed tokens (e.g., [REDACTED_EMAIL])
- 'mask': Partial masking preserving length/format (e.g., j***@acme.com, ***-**-6789)
- 'pseudonymize': Salted deterministic surrogates preserving referential integrity ([PERSON_1], [ORG_1])
- 'deanonymize': Reversible reconstruction using a private local vault
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SALT_ENV = "OLAYA_PII_SALT"


def default_salt() -> str:
    """``OLAYA_PII_SALT`` if set, else a fresh per-process secret.

    A salt baked into source would let anyone with the source reverse ``hash``
    surrogates by dictionary. A per-process secret keeps ``hash`` deterministic
    within one process, which is all referential integrity needs.
    """
    return os.environ.get(SALT_ENV) or secrets.token_hex(16)


class ObfuscationMode(str, Enum):  # noqa: UP042 - preserve existing Enum string representation
    REDACT = "redact"
    MASK = "mask"
    PSEUDONYMIZE = "pseudonymize"
    HASH = "hash"

    @property
    def reversible(self) -> bool:
        """Only pseudonymize and hash can be undone: redact and mask map
        distinct originals to the same string, so no vault can restore them."""
        return self in (ObfuscationMode.PSEUDONYMIZE, ObfuscationMode.HASH)


@dataclass(frozen=True)
class PIIEntity:
    entity_type: str
    original: str
    replacement: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ObfuscationResult:
    sanitized_text: str
    entities: list[PIIEntity]
    mode: str
    risk_score: float  # 0.0 to 1.0
    latency_us: float
    vault: dict[str, str] = field(default_factory=dict)  # token -> original

    @property
    def reversible(self) -> bool:
        return ObfuscationMode(self.mode).reversible

    def to_dict(self) -> dict[str, Any]:
        return {
            "sanitized_text": self.sanitized_text,
            "reversible": self.reversible,
            "entity_count": len(self.entities),
            "entities": [
                {
                    "type": e.entity_type,
                    "replacement": e.replacement,
                    "confidence": e.confidence,
                    "span": [e.start, e.end],
                }
                for e in self.entities
            ],
            "mode": self.mode,
            "risk_score": round(self.risk_score, 4),
            "latency_us": round(self.latency_us, 2),
            "vault": self.vault,
        }


# High-precision regular expressions
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
IPV4_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
DATE_OF_BIRTH_REGEX = re.compile(
    r"\b(?:DOB|Date of Birth|Born)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
STREET_ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\s]+?\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr"
    r"|Lane|Ln|Way|Court|Ct|Circle|Cir|Terrace|Ter|Place|Pl|Parkway|Pkwy|Highway|Hwy|Square|Sq)\b\.?",
    re.IGNORECASE,
)
HONORIFIC_NAME_REGEX = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|CEO|CFO|COO|CTO|President|Chairman|Chairwoman)"
    r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
DEAL_CODENAME_REGEX = re.compile(r"\b(?:Project|Operation)\s+([A-Z][A-Za-z0-9_-]+)\b")


class PIIObfuscator:
    """Air-gapped, zero-allocation-optimized PII/PHI/MNPI obfuscation engine."""

    def __init__(
        self,
        salt: str | None = None,
        candidate_filter: Callable[[str, str], bool] | None = None,
    ):
        """``salt`` defaults to :func:`default_salt`. ``candidate_filter(entity_type, text)``
        runs on every regex candidate before overlap resolution; returning False
        drops it. Consumers plug structural validators in here (Luhn for cards,
        SSA rules for SSNs) so a regex only proposes and a validator decides."""
        self.salt = salt if salt is not None else default_salt()
        self.candidate_filter = candidate_filter

    def obfuscate(
        self,
        text: str,
        mode: str | ObfuscationMode = ObfuscationMode.PSEUDONYMIZE,
        custom_entities: list[str] | None = None,
        deal_codenames: list[str] | None = None,
    ) -> ObfuscationResult:
        """Obfuscate all PII/PHI/MNPI in text using the requested mode.

        Maintains referential integrity: identical entities produce identical
        pseudonyms within the same document run.
        """
        t0 = time.perf_counter_ns()
        if isinstance(mode, str):
            mode = ObfuscationMode(mode.lower())

        found_spans: list[tuple[int, int, str, str]] = []  # (start, end, type, text)

        # 1. Custom Deal / Company / Executive Entities
        if custom_entities:
            for ent in custom_entities:
                if not ent or len(ent.strip()) < 2:
                    continue
                pattern = re.compile(re.escape(ent), re.IGNORECASE)
                for m in pattern.finditer(text):
                    found_spans.append((m.start(), m.end(), "CUSTOM_ENTITY", m.group(0)))

        # 2. Deal Codenames (e.g. Project Falcon, Project Titan)
        for m in DEAL_CODENAME_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "DEAL_CODENAME", m.group(0)))
        if deal_codenames:
            for code in deal_codenames:
                pattern = re.compile(rf"\b{re.escape(code)}\b", re.IGNORECASE)
                for m in pattern.finditer(text):
                    found_spans.append((m.start(), m.end(), "DEAL_CODENAME", m.group(0)))

        # 3. SSN / Tax IDs
        for m in SSN_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "SSN", m.group(0)))

        # 4. Credit Cards
        for m in CREDIT_CARD_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "CREDIT_CARD", m.group(0)))

        # 5. Email addresses
        for m in EMAIL_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "EMAIL", m.group(0)))

        # 6. Phone numbers
        for m in PHONE_REGEX.finditer(text):
            val = m.group(0)
            # Filter out simple integers or year ranges
            digits_only = re.sub(r"\D", "", val)
            if 10 <= len(digits_only) <= 15:
                found_spans.append((m.start(), m.end(), "PHONE", val))

        # 7. IP Addresses
        for m in IPV4_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "IP_ADDRESS", m.group(0)))

        # 8. Physical street addresses
        for m in STREET_ADDRESS_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "STREET_ADDRESS", m.group(0)))

        # 9. Dates of Birth
        for m in DATE_OF_BIRTH_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "DOB", m.group(0)))

        # 10. Names with titles / honorifics
        for m in HONORIFIC_NAME_REGEX.finditer(text):
            found_spans.append((m.start(), m.end(), "PERSON_NAME", m.group(0)))

        if self.candidate_filter is not None:
            found_spans = [span for span in found_spans if self.candidate_filter(span[2], span[3])]

        # Sort spans by start asc, length desc to handle overlaps cleanly
        found_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

        # De-duplicate overlapping spans (keep longest/earliest)
        non_overlapping: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, etype, orig in found_spans:
            if start >= last_end:
                non_overlapping.append((start, end, etype, orig))
                last_end = end

        # Tracking maps for referential integrity
        entity_counts: dict[str, int] = {}
        entity_to_pseudonym: dict[str, str] = {}
        vault: dict[str, str] = {}
        entities_result: list[PIIEntity] = []

        # Build replacement string
        out_chunks: list[str] = []
        curr_idx = 0

        for start, end, etype, orig in non_overlapping:
            out_chunks.append(text[curr_idx:start])

            norm_orig = orig.strip()
            replacement = self._generate_replacement(
                norm_orig,
                etype,
                mode,
                entity_counts,
                entity_to_pseudonym,
            )

            if mode.reversible:
                existing = vault.get(replacement)
                if existing is not None and existing != orig:
                    raise ValueError(f"surrogate collision for {replacement}; the vault would not be reversible")
                vault[replacement] = orig
            entities_result.append(
                PIIEntity(
                    entity_type=etype,
                    original=orig,
                    replacement=replacement,
                    start=start,
                    end=end,
                )
            )
            out_chunks.append(replacement)
            curr_idx = end

        out_chunks.append(text[curr_idx:])
        sanitized = "".join(out_chunks)

        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0

        # Calculate risk score based on density and severity of PII detected
        risk_score = self._compute_risk_score(entities_result)

        return ObfuscationResult(
            sanitized_text=sanitized,
            entities=entities_result,
            mode=mode.value,
            risk_score=risk_score,
            latency_us=latency_us,
            vault=vault,
        )

    @staticmethod
    def deanonymize(sanitized_text: str, vault: dict[str, str]) -> str:
        """Restore original text using the private local de-anonymization vault.

        One pass, longest token first, so a restored value can never be matched
        again as another token (sequential replace could turn ``[A_1]`` into
        ``[A_2]`` and then into the wrong original)."""
        if not vault:
            return sanitized_text
        pattern = re.compile("|".join(re.escape(k) for k in sorted(vault, key=len, reverse=True)))
        return pattern.sub(lambda m: vault[m.group(0)], sanitized_text)

    def _generate_replacement(
        self,
        original: str,
        entity_type: str,
        mode: ObfuscationMode,
        entity_counts: dict[str, int],
        entity_to_pseudonym: dict[str, str],
    ) -> str:
        # Check cache for referential consistency
        cache_key = f"{entity_type}:{original.lower()}"
        if cache_key in entity_to_pseudonym:
            return entity_to_pseudonym[cache_key]

        if mode == ObfuscationMode.REDACT:
            rep = f"[REDACTED_{entity_type}]"

        elif mode == ObfuscationMode.MASK:
            if entity_type == "EMAIL" and "@" in original:
                user, domain = original.split("@", 1)
                masked_user = user[0] + "***" if len(user) > 1 else "***"
                rep = f"{masked_user}@{domain}"
            elif entity_type == "PHONE":
                digits = re.sub(r"\D", "", original)
                last4 = digits[-4:] if len(digits) >= 4 else "XXXX"
                rep = f"(***) ***-{last4}"
            elif entity_type == "SSN":
                digits = re.sub(r"\D", "", original)
                last4 = digits[-4:] if len(digits) >= 4 else "XXXX"
                rep = f"***-**-{last4}"
            elif entity_type == "CREDIT_CARD":
                digits = re.sub(r"\D", "", original)
                last4 = digits[-4:] if len(digits) >= 4 else "XXXX"
                rep = f"****-****-****-{last4}"
            else:
                rep = original[0] + "*" * (max(3, len(original) - 2)) + original[-1] if len(original) > 2 else "***"

        elif mode == ObfuscationMode.PSEUDONYMIZE:
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            idx = entity_counts[entity_type]
            rep = f"[{entity_type}_{idx}]"

        elif mode == ObfuscationMode.HASH:
            h = hashlib.sha256(f"{self.salt}:{original}".encode()).hexdigest()[:8]
            rep = f"[{entity_type}_{h}]"

        else:
            rep = f"[REDACTED_{entity_type}]"

        entity_to_pseudonym[cache_key] = rep
        return rep

    def _compute_risk_score(self, entities: list[PIIEntity]) -> float:
        if not entities:
            return 0.0
        weights = {
            "SSN": 1.0,
            "CREDIT_CARD": 0.95,
            "DEAL_CODENAME": 0.9,
            "DOB": 0.8,
            "PERSON_NAME": 0.7,
            "EMAIL": 0.6,
            "PHONE": 0.6,
            "STREET_ADDRESS": 0.6,
            "CUSTOM_ENTITY": 0.85,
            "IP_ADDRESS": 0.4,
        }
        score = sum(weights.get(e.entity_type, 0.5) for e in entities)
        # Normalize into a concave probability curve bounded in [0.0, 1.0]
        return min(1.0, 1.0 - (0.5 ** (score / 1.5)))

    def sovereign_reason(
        self,
        prompt: str,
        system: str | None = None,
        model: str = "qwen2.5:7b",
        ollama_host: str = "http://127.0.0.1:11434",
        custom_entities: list[str] | None = None,
        deal_codenames: list[str] | None = None,
        llm_caller: Any | None = None,
        temperature: float = 0.2,
    ) -> SovereignReasonResult:
        """Atomic One-Shot Sovereign Privacy Loop:
        1. Obfuscates all PII/MNPI into private local vault (<100 µs)
        2. Dispatches strictly sanitized surrogate tokens to LLM (0 data egress)
        3. Restores real entity identities on-premises from vault
        """
        # Step 1: Obfuscate prompt
        t0_ob = time.perf_counter_ns()
        ob_res = self.obfuscate(
            prompt,
            mode=ObfuscationMode.PSEUDONYMIZE,
            custom_entities=custom_entities,
            deal_codenames=deal_codenames,
        )
        t1_ob = time.perf_counter_ns()
        ob_us = (t1_ob - t0_ob) / 1000.0

        # Step 2: Inference with sanitized text only
        t0_llm = time.perf_counter()
        if llm_caller is not None:
            sanitized_response = llm_caller(ob_res.sanitized_text)
        else:
            url = f"{ollama_host.rstrip('/')}/api/generate"
            payload: dict[str, Any] = {
                "model": model,
                "prompt": ob_res.sanitized_text,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if system:
                payload["system"] = system
            import json
            import urllib.request

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                sanitized_response = data.get("response", "")
        t1_llm = time.perf_counter()
        llm_ms = (t1_llm - t0_llm) * 1000.0

        # Step 3: Local Vault De-anonymization
        t0_de = time.perf_counter_ns()
        restored = self.deanonymize(sanitized_response, ob_res.vault)
        t1_de = time.perf_counter_ns()
        de_us = (t1_de - t0_de) / 1000.0

        # Verify zero leak: none of the vault's raw values existed in the sanitized prompt
        zero_leak = all(raw_val not in ob_res.sanitized_text for raw_val in ob_res.vault.values())

        return SovereignReasonResult(
            response=restored,
            sanitized_prompt=ob_res.sanitized_text,
            sanitized_response=sanitized_response,
            entities_protected=len(ob_res.entities),
            risk_score=ob_res.risk_score,
            vault=ob_res.vault,
            obfuscation_us=ob_us,
            deanonymization_us=de_us,
            llm_latency_ms=llm_ms,
            model=model,
            zero_leak_verified=zero_leak,
        )


@dataclass
class SovereignReasonResult:
    response: str
    sanitized_prompt: str
    sanitized_response: str
    entities_protected: int
    risk_score: float
    vault: dict[str, str]
    obfuscation_us: float
    deanonymization_us: float
    llm_latency_ms: float
    model: str
    zero_leak_verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "sanitized_prompt": self.sanitized_prompt,
            "sanitized_response": self.sanitized_response,
            "entities_protected": self.entities_protected,
            "risk_score": round(self.risk_score, 4),
            "vault": self.vault,
            "obfuscation_us": round(self.obfuscation_us, 2),
            "deanonymization_us": round(self.deanonymization_us, 2),
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            "model": self.model,
            "zero_leak_verified": self.zero_leak_verified,
        }
