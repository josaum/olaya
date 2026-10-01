"""Sequence formatting with ontological markers and token layout for O-Laya.

Follows the non-autoregressive sequence construction:
[CLS] <qtype> question: <instructions> [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP] <state> [SEP]
Tracks the precise positional indices of [MASK] marker tokens for the decision head.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from olaya.ontology import Concept, Ontology


@dataclass
class SequenceBundle:
    """The constructed token sequence and metadata for the decision head."""

    input_ids: list[int]
    attention_mask: list[int]
    marker_positions: list[int]  # Position of each [MASK] token
    candidate_curies: list[str]  # Corresponds 1:1 with marker_positions
    qtype: int  # See QTYPE_IDS


# Question-type ids fed to the model's question-type embedding.
QTYPE_IDS: dict[str, int] = {"concept": 0, "score": 1, "subsumes": 2, "choice": 3, "noul": 4}


def serialize_state(state: Any) -> str:
    """Deterministic serialization of input state matching Python reference."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def format_concept_option(concept: Concept, ontology: Ontology) -> str:
    """Format a concept into an axiom-annotated candidate string."""
    parts = [f"{concept.curie}: {concept.label}"]

    # Add parent context if available
    parent_labels = [ontology.concepts[p].label for p in concept.parents if p in ontology.concepts]
    if parent_labels:
        parts.append(f"(subClassOf: {', '.join(parent_labels)})")

    # Add description / criteria
    if concept.description:
        parts.append(f"- {concept.description}")

    return " ".join(parts)


def build_ontology_sequence(
    tokenizer_encode: Callable[[str], list[int]],
    special_ids: dict[str, int],  # {"cls": id, "sep": id, "mask": id, "pad": id}
    state: Any,
    instructions: str,
    active_concepts: list[Concept],
    ontology: Ontology,
    qtype_id: int = 0,
    max_len: int = 512,
    head_max_len: int = 256,
    qtype_name: str = "concept",
) -> SequenceBundle:
    """Assemble token IDs and track marker positions for candidate concepts.

    The header is capped at half of ``head_max_len`` and the first option is always kept (truncated
    if necessary), so every bundle built from a non-empty ``active_concepts`` has at least one marker.
    """
    if not active_concepts:
        raise ValueError("build_ontology_sequence: at least one candidate concept is required.")
    cls_id = special_ids.get("cls", 101)
    sep_id = special_ids.get("sep", 102)
    mask_id = special_ids.get("mask", 103)

    # 1. Header: [CLS] <qtype> question: <instructions> [SEP]
    header_text = f"{qtype_name} question: {instructions}"
    header_tokens = tokenizer_encode(header_text)[: head_max_len // 2]

    seq: list[int] = [cls_id] + header_tokens + [sep_id]
    marker_positions: list[int] = []
    candidate_curies: list[str] = []

    # 2. Options block: each option prefixed with [MASK]
    for concept in active_concepts:
        opt_text = format_concept_option(concept, ontology)
        opt_tokens = tokenizer_encode(" " + opt_text)[:48]  # cap per option

        # Check budget (the first option is always kept, truncated to whatever room is left)
        if len(seq) + 1 + len(opt_tokens) > head_max_len:
            if marker_positions:
                break
            opt_tokens = opt_tokens[: max(0, head_max_len - len(seq) - 1)]

        # Record position of the [MASK] marker
        marker_positions.append(len(seq))
        seq.append(mask_id)
        seq.extend(opt_tokens)
        candidate_curies.append(concept.curie)

    seq.append(sep_id)

    # 3. State text block: [state] [SEP]
    state_str = serialize_state(state)
    room_for_state = max(0, max_len - len(seq) - 1)
    state_tokens = tokenizer_encode(state_str)[:room_for_state]

    seq.extend(state_tokens)
    seq.append(sep_id)

    # Truncate if exceeds max_len
    final_ids = seq[:max_len]
    attention_mask = [1] * len(final_ids)

    # Keep only markers that fit in the final sequence
    valid_markers = [m for m in marker_positions if m < len(final_ids)]
    valid_curies = candidate_curies[: len(valid_markers)]

    return SequenceBundle(
        input_ids=final_ids,
        attention_mask=attention_mask,
        marker_positions=valid_markers,
        candidate_curies=valid_curies,
        qtype=qtype_id,
    )
