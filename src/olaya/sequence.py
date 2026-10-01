"""Sequence formatting with ontological markers and token layout for O-Laya.

Follows the non-autoregressive sequence construction:
[CLS] <qtype> question: <instructions> [SEP] [MASK] opt_0 [MASK] opt_1 ... [SEP] <state> [SEP]
Tracks the precise positional indices of [MASK] marker tokens for the decision head.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from olaya.ontology import Concept, Ontology


@dataclass
class SequenceBundle:
    """The constructed token sequence and metadata for the decision head."""

    input_ids: List[int]
    attention_mask: List[int]
    marker_positions: List[int]  # Position of each [MASK] token
    candidate_curies: List[str]  # Corresponds 1:1 with marker_positions
    qtype: int  # 0: concept/choice, 1: score, 2: subsumes/noul


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
    tokenizer_encode: Callable[[str], List[int]],
    special_ids: Dict[str, int],  # {"cls": id, "sep": id, "mask": id, "pad": id}
    state: Any,
    instructions: str,
    active_concepts: List[Concept],
    ontology: Ontology,
    qtype_id: int = 0,
    max_len: int = 512,
    head_max_len: int = 256,
) -> SequenceBundle:
    """Assemble token IDs and track marker positions for candidate concepts."""
    cls_id = special_ids.get("cls", 101)
    sep_id = special_ids.get("sep", 102)
    mask_id = special_ids.get("mask", 103)

    # 1. Header: [CLS] concept question: <instructions> [SEP]
    header_text = f"concept question: {instructions}"
    header_tokens = tokenizer_encode(header_text)

    seq: List[int] = [cls_id] + header_tokens + [sep_id]
    marker_positions: List[int] = []
    candidate_curies: List[str] = []

    # 2. Options block: each option prefixed with [MASK]
    for concept in active_concepts:
        opt_text = format_concept_option(concept, ontology)
        opt_tokens = tokenizer_encode(" " + opt_text)[:48]  # cap per option

        # Check budget
        if len(seq) + 1 + len(opt_tokens) > head_max_len:
            break

        # Record position of the [MASK] marker
        marker_positions.append(len(seq))
        seq.append(mask_id)
        seq.extend(opt_tokens)
        candidate_curies.append(concept.curie)

    seq.append(sep_id)

    # 3. State text block: [state] [SEP]
    state_str = serialize_state(state)
    room_for_state = max(16, max_len - len(seq) - 1)
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
