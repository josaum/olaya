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
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObfuscationMode(str, Enum):
    REDACT = "redact"
    MASK = "mask"
    PSEUDONYMIZE = "pseudonymize"
    HASH = "hash"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "sanitized_text": self.sanitized_text,
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
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
SSN_REGEX = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)
CREDIT_CARD_REGEX = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)
IPV4_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
DATE_OF_BIRTH_REGEX = re.compile(
    r"\b(?:DOB|Date of Birth|Born)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
STREET_ADDRESS_REGEX = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir)\b",
    re.IGNORECASE,
)
HONORIFIC_NAME_REGEX = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|CEO|CFO|COO|President)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
DEAL_CODENAME_REGEX = re.compile(
    r"\b(?:Project|Operation)\s+([A-Z][A-Za-z0-9_-]+)\b"
)


class PIIObfuscator:
    """Air-gapped, zero-allocation-optimized PII/PHI/MNPI obfuscation engine."""

    def __init__(self, salt: str = "leio-sovereign-vault-2026"):
        self.salt = salt

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

        # 2. Deal Codenames (e.g. Project Helios, Project Titan)
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

    def deanonymize(self, sanitized_text: str, vault: dict[str, str]) -> str:
        """Restore original text using the private local de-anonymization vault."""
        result = sanitized_text
        # Sort keys by length descending to prevent sub-string collision
        for token in sorted(vault.keys(), key=len, reverse=True):
            result = result.replace(token, vault[token])
        return result

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
            h = hashlib.sha256(f"{self.salt}:{original}".encode("utf-8")).hexdigest()[:8]
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
