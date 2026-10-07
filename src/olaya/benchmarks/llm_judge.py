"""LLM judge stage for the OAEI Conference (ra1) matcher.

The hybrid matcher (``olaya.benchmarks.oaei.HybridMatcher``) loses most of its recall
on ra1 in three places (``benchmarks/results/oaei-error-analysis.md``): mutual-best
pairs that score 0.30-0.49 and fall under the threshold, synonyms the encoder scores
near zero (Paper / Contribution, Event / Activity, Attendee / Delegate) and entities
consumed by the lexical stage with the wrong partner; and most of its false positives
are identical-label pairs that mean different things in context.  All four are
decisions a reader with the two ontologies in front of them makes easily, so this
module adds a *judge* stage that asks a language model, with the ontology context of
both entities, whether a candidate pair is equivalent.

Pipeline (every step deterministic and cacheable):

1. **Candidates** (:func:`build_candidates`): the hybrid-accepted correspondences
   (to be verified) plus, for every entity, its top-``k`` same-kind candidates by the
   dense hybrid score (``0.5 * lexical + 0.5 * CSLS cosine``) above ``min_score`` and
   its top-``k`` by the path-text cosine.  The reference recall of this set is the
   ceiling of the whole stage and is reported before any judging.
2. **Prompt** (:func:`entity_card`, :func:`render_batch`): each entity is described by
   its label, kind, path from the root, parents / children / siblings, properties with
   domain and range (classes) or domain / range (properties), comment and ontology name;
   ``~10-20`` pairs go into one request and the verdicts come back as structured JSON.
3. **Cache** (:class:`JudgmentCache`): every verdict is stored as one JSON line keyed by
   ``sha256(model, prompt version, pair texts)``, so a run only sends unjudged pairs and
   the benchmark is reproducible offline once the judgments are committed.
4. **Decoding** (:func:`decode`): the final alignment is a function of the hybrid score
   and the verdict - judge-only, precision / recall repair of the hybrid alignment, or a
   fused score - followed by the usual mutual-best 1:1 selection and a threshold; the
   knobs are chosen by leave-one-pair-out cross-validation like the other benchmarks.

Judges: :class:`AnthropicJudge` (the real model, official SDK, key from the
environment), :class:`OpenAICompatJudge` (any OpenAI-compatible ``chat/completions`` server -
Qwen Cloud, Z.ai / Zhipu GLM, ... - standard-library HTTP, key from the environment), :class:`StubJudge` (answers from the reference alignment with optional
noise; tests and the offline sanity run) and :class:`NullJudge` (always unrelated).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from olaya.benchmarks.oaei import (
    EmbeddingMatcher,
    EmbeddingMatcherConfig,
    HybridMatcher,
    HybridMatcherConfig,
    OAEIAlignment,
    apply_threshold,
    build_entity_texts,
    camel_to_words,
    select_mutual_best,
)
from olaya.ontology import Ontology

PROMPT_VERSION = "oaei-judge-v1"

DEFAULT_MODEL = "claude-opus-5-5"
SUPPORTED_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")
#: USD per million tokens (input, output) - Anthropic first-party rates, for the cost estimate.
MODEL_PRICES_USD_PER_MTOK = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
RELATIONS = ("equivalent", "subsumes", "subsumed_by", "related", "unrelated")
_RELATION_ALIASES = {
    "equal": "equivalent",
    "same": "equivalent",
    "sameas": "equivalent",
    "broader": "subsumes",
    "narrower": "subsumed_by",
    "disjoint": "unrelated",
    "different": "unrelated",
    "none": "unrelated",
    "not_equivalent": "unrelated",
    "overlapping": "related",
    "similar": "related",
}


def _canonical_relation(rel: Any) -> Any:
    """Map common free-text relation labels a model may emit onto RELATIONS; unknown values pass through."""
    if not isinstance(rel, str):
        return rel
    key = rel.strip().lower().replace("-", "_").replace(" ", "_")
    if key in RELATIONS:
        return key
    return _RELATION_ALIASES.get(key, rel)


#: OpenAI-compatible judge: generic defaults and the Z.ai preset (``--judge zai``).
OPENAI_DEFAULT_MODEL = "qwen3.8-flash"
OPENAI_API_KEY_ENV = "LLM_JUDGE_API_KEY"
OPENAI_BASE_URL_ENV = "LLM_JUDGE_BASE_URL"
ZAI_DEFAULT_MODEL = "glm-5.3-flash"
ZAI_DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_PRICING_URL = "https://docs.z.ai/guides/overview/pricing"
ZAI_API_KEY_ENV = ("ZAI_API_KEY", "Z_AI_API_KEY")
#: USD per million tokens (input, output) per provider label.  Z.ai read from ZAI_PRICING_URL on
#: 2026-10-07 (no cache discount modelled).  Providers / models not listed here cost 0: pricing not pinned,
#: fill from the provider's pricing page (Qwen Cloud: https://maas.qwencloudapi.com).
PROVIDER_MODEL_PRICES_USD_PER_MTOK: dict[str, dict[str, tuple[float, float]]] = {
    "anthropic": MODEL_PRICES_USD_PER_MTOK,
    "zai": {
        "glm-5.3-flash": (0.15, 0.50),
        "glm-5.3-flashx": (0.37, 1.25),
        "glm-5.3": (1.4, 4.4),
        "glm-4.7-flash": (0.0, 0.0),
        "glm-4.7-flashx": (0.07, 0.40),
        "glm-4.5-flash": (0.0, 0.0),
    },
}


def price_for(provider: str, model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens, ``(0, 0)`` when not pinned."""
    return PROVIDER_MODEL_PRICES_USD_PER_MTOK.get(provider, {}).get(model, (0.0, 0.0))


_MAX_LISTED = 10


# ---------------------------------------------------------------------------
# Entity cards (the ontology context shown to the judge)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityCard:
    """Everything the judge is told about one entity.  Field order is the rendering order."""

    ontology: str
    name: str
    kind: str  # "class" | "property"
    label: str
    path: str
    parents: tuple[str, ...]
    children: tuple[str, ...]
    siblings: tuple[str, ...]
    properties: tuple[str, ...]  # classes: "has Author (Paper -> Person)"
    domain: tuple[str, ...]  # properties
    range: tuple[str, ...]  # properties
    comment: str
    property_kind: str = ""  # "object" | "datatype" for properties

    def render(self) -> str:
        lines = [f"{self.kind} `{self.label}` (id {self.name}, ontology {self.ontology})"]
        if self.path and self.path != self.label:
            lines.append(f"  path: {self.path}")
        if self.comment:
            lines.append(f"  comment: {self.comment}")
        if self.parents:
            lines.append("  parents: " + ", ".join(self.parents))
        if self.children:
            lines.append("  children: " + ", ".join(self.children))
        if self.siblings:
            lines.append("  siblings: " + ", ".join(self.siblings))
        if self.properties:
            lines.append("  properties: " + "; ".join(self.properties))
        if self.kind == "property":
            lines.append(f"  type: {self.property_kind or 'object'} property")
            lines.append("  domain: " + (", ".join(self.domain) or "(unspecified)"))
            lines.append("  range: " + (", ".join(self.range) or "(unspecified)"))
        return "\n".join(lines)


def _label(ontology: Ontology, name: str) -> str:
    c = ontology.concepts.get(name)
    if c is not None and c.metadata.get("has_label"):
        return c.label
    p = ontology.properties.get(name)
    if p is not None and p.metadata.get("has_label"):
        return p.label
    return camel_to_words(name)


def _listed(ontology: Ontology, names: Iterable[str]) -> tuple[str, ...]:
    items = [_label(ontology, n) for n in sorted(names)]
    return tuple(items[:_MAX_LISTED])


def entity_cards(ontology: Ontology) -> dict[str, EntityCard]:
    """One :class:`EntityCard` per class and property of ``ontology`` (deterministic)."""
    texts = {t.name: t for t in build_entity_texts(ontology)}
    domain_of: dict[str, set[str]] = {}
    range_of: dict[str, set[str]] = {}
    for pcurie, prop in ontology.properties.items():
        for d in prop.domains:
            domain_of.setdefault(d, set()).add(pcurie)
        for r in prop.ranges:
            range_of.setdefault(r, set()).add(pcurie)
    out: dict[str, EntityCard] = {}
    for curie, concept in sorted(ontology.concepts.items()):
        siblings: set[str] = set()
        for p in concept.parents:
            if p in ontology.concepts:
                siblings |= ontology.concepts[p].children
        siblings.discard(curie)
        props = []
        for pc in sorted(domain_of.get(curie, set()) | range_of.get(curie, set()))[:_MAX_LISTED]:
            prop = ontology.properties[pc]
            dom = ", ".join(_label(ontology, d) for d in sorted(prop.domains)) or "?"
            rng = ", ".join(_label(ontology, r) for r in sorted(prop.ranges)) or str(
                prop.metadata.get("datatype") or "?"
            )
            props.append(f"{_label(ontology, pc)} ({dom} -> {rng})")
        out[curie] = EntityCard(
            ontology=ontology.name,
            name=curie,
            kind="class",
            label=_label(ontology, curie),
            path=texts[curie].path,
            parents=_listed(ontology, concept.parents),
            children=_listed(ontology, concept.children),
            siblings=_listed(ontology, siblings),
            properties=tuple(props),
            domain=(),
            range=(),
            comment=(concept.description or "").strip(),
        )
    for curie, prop in sorted(ontology.properties.items()):
        rng = _listed(ontology, prop.ranges)
        if not rng and prop.metadata.get("datatype"):
            rng = (str(prop.metadata["datatype"]),)
        out[curie] = EntityCard(
            ontology=ontology.name,
            name=curie,
            kind="property",
            label=_label(ontology, curie),
            path=texts[curie].path,
            parents=_listed(ontology, prop.parents),
            children=_listed(ontology, prop.children),
            siblings=(),
            properties=(),
            domain=_listed(ontology, prop.domains),
            range=rng,
            comment=(prop.description or "").strip(),
            property_kind=str(prop.metadata.get("kind", "object")),
        )
    return out


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    source: str
    target: str
    kind: str
    hybrid: float  # dense hybrid score (0.5 lexical + 0.5 CSLS cosine)
    path: float  # path-text CSLS cosine
    rich: float  # raw (uncalibrated) cosine of the rich texts
    accepted: bool  # produced by the a-priori hybrid alignment (cascade, mutual-best, threshold)
    origin: tuple[str, ...]  # why it is a candidate: "accepted", "hybrid", "path", "rich"

    @property
    def pair(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass
class CandidateConfig:
    k: int = 3
    min_score: float = 0.15
    hybrid_threshold: float = 0.5  # a-priori acceptance threshold of the hybrid alignment
    csls_k: int = 10
    # Third source, added after measuring the ceiling: top-``rich_k`` per entity by the *raw* cosine
    # of the rich texts (label + comment + path + children + siblings + properties).  CSLS zeroes the
    # synonym pairs (Paper / Contribution, Event / Activity ...) because the encoder puts them in the
    # hub region; the raw cosine still ranks them near the top.  ``rich_k = 0`` switches it off.
    rich_k: int = 2
    rich_min: float = 0.0


@dataclass
class CandidateSet:
    source: str
    target: str
    candidates: list[Candidate]

    @property
    def pairs(self) -> set[tuple[str, str]]:
        return {c.pair for c in self.candidates}

    def recall(self, reference: Iterable[OAEIAlignment]) -> dict[str, float]:
        """Fraction of reference correspondences inside the candidate set, per modality."""
        by_kind = {c.pair: c.kind for c in self.candidates}
        hit = {"class": [0, 0], "property": [0, 0]}
        for r in reference:
            kind = by_kind.get((r.entity1, r.entity2))
            # A reference pair outside the set has no kind here: infer it from the entity names.
            k = kind or ("property" if r.entity1[:1].islower() else "class")
            hit[k][1] += 1
            hit[k][0] += int(kind is not None)
        n1, n2 = hit["class"], hit["property"]
        return {
            "M1": n1[0] / n1[1] if n1[1] else 1.0,
            "M2": n2[0] / n2[1] if n2[1] else 1.0,
            "M3": (n1[0] + n2[0]) / (n1[1] + n2[1]) if (n1[1] + n2[1]) else 1.0,
        }


def topk_pairs(scores: Mapping[tuple[str, str], float], k: int, min_score: float) -> set[tuple[str, str]]:
    """Top-``k`` per source entity and per target entity (union), ties broken by name."""
    if k <= 0:
        return set()
    by_source: dict[str, list[tuple[float, str]]] = {}
    by_target: dict[str, list[tuple[float, str]]] = {}
    for (s, t), v in scores.items():
        if v < min_score:
            continue
        by_source.setdefault(s, []).append((v, t))
        by_target.setdefault(t, []).append((v, s))
    out: set[tuple[str, str]] = set()
    for s, lst in by_source.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        out.update((s, t) for _, t in lst[:k])
    for t, lst in by_target.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        out.update((s, t) for _, s in lst[:k])
    return out


class CandidateGenerator:
    """Builds the judge's candidate set for an ontology pair from the hybrid matcher."""

    def __init__(self, encoder: Any, config: CandidateConfig | None = None):
        self.config = config or CandidateConfig()
        emb = EmbeddingMatcherConfig(text="both", calibration="csls", csls_k=self.config.csls_k)
        self.cascade = HybridMatcher(encoder, HybridMatcherConfig(combine="cascade", embedding=emb))
        self.dense = HybridMatcher(encoder, HybridMatcherConfig(combine="sum", embedding_weight=0.5, embedding=emb))
        self.path = EmbeddingMatcher(
            encoder, EmbeddingMatcherConfig(text="path", calibration="csls", csls_k=self.config.csls_k)
        )
        self.rich = EmbeddingMatcher(encoder, EmbeddingMatcherConfig(text="rich", calibration="none"))

    @property
    def name(self) -> str:
        c = self.config
        return f"candidates[{self.cascade.name},k={c.k},min={c.min_score:.2f},rich_k={c.rich_k}]"

    @staticmethod
    def _pairs(matcher: EmbeddingMatcher, source: Ontology, target: Ontology) -> dict[tuple[str, str], float]:
        st, tt, sim = matcher.similarity_matrix(source, target)
        return {
            (s.name, t.name): float(sim[i, j])
            for i, s in enumerate(st)
            for j, t in enumerate(tt)
            if s.kind == t.kind and sim[i, j] > 0.0
        }

    def accepted(self, source: Ontology, target: Ontology) -> list[OAEIAlignment]:
        scored = self.cascade.score_all(source, target)
        return apply_threshold(select_mutual_best(scored), self.config.hybrid_threshold)

    def build(self, source: Ontology, target: Ontology) -> CandidateSet:
        c = self.config
        accepted = {(a.entity1, a.entity2): a.measure for a in self.accepted(source, target)}
        dense = {(a.entity1, a.entity2): a.measure for a in self.dense.score_all(source, target)}
        path = self._pairs(self.path, source, target)
        rich = self._pairs(self.rich, source, target) if c.rich_k > 0 else {}
        kinds = {t.name: t.kind for t in build_entity_texts(source)}
        origins: dict[tuple[str, str], list[str]] = {}
        for p in accepted:
            origins.setdefault(p, []).append("accepted")
        for p in topk_pairs(dense, c.k, c.min_score):
            origins.setdefault(p, []).append("hybrid")
        for p in topk_pairs(path, c.k, max(c.min_score, 1e-9)):
            origins.setdefault(p, []).append("path")
        for p in topk_pairs(rich, c.rich_k, c.rich_min):
            origins.setdefault(p, []).append("rich")
        out = [
            Candidate(
                source=s,
                target=t,
                kind=kinds[s],
                hybrid=round(dense.get((s, t), 0.0), 6),
                path=round(path.get((s, t), 0.0), 6),
                rich=round(rich.get((s, t), 0.0), 6),
                accepted=(s, t) in accepted,
                origin=tuple(o),
            )
            for (s, t), o in sorted(origins.items())
        ]
        return CandidateSet(source.name, target.name, out)


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeRequest:
    source: EntityCard
    target: EntityCard

    @property
    def pair(self) -> tuple[str, str]:
        return (self.source.name, self.target.name)

    def digest(self, model: str, prompt_version: str = PROMPT_VERSION) -> str:
        h = hashlib.sha256()
        for part in (model, prompt_version, self.source.render(), self.target.render()):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()


@dataclass(frozen=True)
class Judgment:
    equivalent: bool
    confidence: float
    relation: str
    rationale: str = ""

    def score(self) -> float:
        """Judge score in ``[0, 1]``: the confidence when equivalent, else 0."""
        return float(min(max(self.confidence, 0.0), 1.0)) if self.equivalent else 0.0

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Judgment:
        rel = str(d.get("relation", "unrelated"))
        if rel not in RELATIONS:
            rel = "unrelated"
        return Judgment(
            bool(d.get("equivalent", False)), float(d.get("confidence", 0.0)), rel, str(d.get("rationale", ""))
        )


@dataclass
class Usage:
    """Token usage of one request (the cost ledger entry)."""

    model: str
    n_pairs: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    elapsed_s: float = 0.0
    request_id: str = ""
    provider: str = "anthropic"

    def cost_usd(self) -> float:
        inp, out = price_for(self.provider, self.model)
        cached = self.cache_read_input_tokens * inp * 0.1
        return (
            self.input_tokens * inp + self.cache_creation_input_tokens * inp * 1.25 + cached + self.output_tokens * out
        ) / 1e6


class Judge(Protocol):
    name: str
    model: str

    def judge(self, batch: Sequence[JudgeRequest]) -> tuple[list[Judgment], Usage]: ...


class NullJudge:
    """Always ``unrelated`` with confidence 1: the judge stage switched off."""

    def __init__(self) -> None:
        self.name = "null"
        self.model = "null"

    def judge(self, batch: Sequence[JudgeRequest]) -> tuple[list[Judgment], Usage]:
        return [Judgment(False, 1.0, "unrelated", "null judge") for _ in batch], Usage(self.model, len(batch))


class StubJudge:
    """Answers from the reference alignment, with optional symmetric noise.

    ``reference`` maps ``(source ontology, target ontology)`` to the set of reference
    pairs.  With ``noise = p`` every verdict is flipped with probability ``p`` (seeded,
    hence deterministic) and the confidence is drawn uniformly from ``[0.5, 1]`` instead
    of being 1.  Tests and the offline sanity run only; it has seen the gold answers.
    """

    def __init__(
        self,
        reference: Mapping[tuple[str, str], Iterable[tuple[str, str]]],
        *,
        noise: float = 0.0,
        seed: int = 0,
    ):
        self.reference = {k: set(v) for k, v in reference.items()}
        self.noise = noise
        self.seed = seed
        self.name = f"stub[noise={noise:.2f}]"
        self.model = "stub"

    def judge(self, batch: Sequence[JudgeRequest]) -> tuple[list[Judgment], Usage]:
        out = []
        for req in batch:
            gold = req.pair in self.reference.get((req.source.ontology, req.target.ontology), set())
            verdict = gold
            conf = 1.0
            if self.noise > 0.0:
                rng = random.Random(f"{self.seed}:{req.digest(self.model)}")
                if rng.random() < self.noise:
                    verdict = not gold
                conf = 0.5 + 0.5 * rng.random()
            rel = "equivalent" if verdict else "unrelated"
            out.append(Judgment(verdict, conf, rel, "stub: reference alignment" + (" (noisy)" if self.noise else "")))
        return out, Usage(self.model, len(batch))


SYSTEM_PROMPT = """You are an expert ontology engineer performing ontology alignment on the OAEI Conference track: seven independently written OWL ontologies of the conference-organisation domain (papers, reviews, authors, committees, events, venues, deadlines).

For each numbered candidate pair you are given two entities, one from each ontology, with the ontology context: label, kind (class or property), path from the root, parents / children / siblings, the properties a class is domain or range of (with their domain -> range), the domain and range of a property, and the rdfs:comment when the ontology has one.

Decide for each pair whether the two entities denote the same concept or the same relation in the conference domain, as a reference alignment built by domain experts would record it:
- Two classes are equivalent when their intended instances coincide (synonyms such as Paper / Contribution, Attendee / Delegate, Event / Activity count when the context agrees), not merely when they share a word or a parent.
- Identical labels are not sufficient: the same word can denote different things (e.g. a Document in one ontology may be a specific artefact in another); use the paths, siblings, properties and comments to decide.
- Two properties are equivalent when they relate the same kinds of things in the same direction; a property and its inverse are not equivalent.
- Prefer "subsumes" / "subsumed_by" when one entity is strictly more general than the other, "related" when they overlap without being equivalent, and "unrelated" otherwise.

Report your confidence that the relation you chose is the one a reference alignment would contain (0 = guess, 1 = certain). Answer only through the JSON schema, one verdict per pair in the given order, keeping the ids."""

#: Appended to the user message for JSON-mode providers that do not enforce OUTPUT_SCHEMA server-side.
OPENAI_OUTPUT_INSTRUCTION = (
    "Respond with exactly one JSON object of the form "
    '{"verdicts": [{"id": 1, "equivalent": true, "confidence": 0.9, "relation": "equivalent", "rationale": "..."}, ...]} '
    "containing one verdict for EVERY pair above, ids 1..N in order, no other keys. "
    '"relation" must be one of equivalent, subsumes, subsumed_by, related, unrelated; '
    '"equivalent" is true only when relation is equivalent.'
)

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "equivalent": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "relation": {"type": "string", "enum": list(RELATIONS)},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "equivalent", "confidence", "relation", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def render_batch(batch: Sequence[JudgeRequest]) -> str:
    """The user message for one batch of candidate pairs (deterministic)."""
    if not batch:
        return ""
    src_onto, tgt_onto = batch[0].source.ontology, batch[0].target.ontology
    parts = [
        f"Ontology A: {src_onto}.  Ontology B: {tgt_onto}.  {len(batch)} candidate pair(s).",
        "For each pair decide whether entity A and entity B are equivalent.",
        "",
    ]
    for i, req in enumerate(batch, 1):
        parts.append(f"### Pair {i}")
        parts.append("A: " + req.source.render())
        parts.append("B: " + req.target.render())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def estimate_tokens(text: str) -> int:
    """Rough token count (characters / 4) for the cost estimate without an API key."""
    return max(1, len(text) // 4)


class AnthropicJudge:
    """The real judge: one ``messages.create`` call per batch, structured JSON output.

    The API key is read by the SDK from ``ANTHROPIC_API_KEY`` (never passed here).
    Thinking is adaptive by default on these models; ``effort`` controls its depth
    (``output_config.effort``; not sent for Haiku 4.5, which does not support it).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout: float = 300.0,
        max_retries: int = 5,
        client: Any | None = None,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic[{model},effort={effort},{PROMPT_VERSION}]"

    def _request_kwargs(self, batch: Sequence[JudgeRequest]) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
        if not self.model.startswith("claude-haiku"):
            output_config["effort"] = self.effort
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": render_batch(batch)}],
            "output_config": output_config,
        }

    def count_tokens(self, batch: Sequence[JudgeRequest]) -> int:
        kw = self._request_kwargs(batch)
        resp = self.client.messages.count_tokens(model=kw["model"], system=kw["system"], messages=kw["messages"])
        return int(resp.input_tokens)

    def judge(self, batch: Sequence[JudgeRequest]) -> tuple[list[Judgment], Usage]:
        t0 = time.perf_counter()
        response = self.client.messages.create(**self._request_kwargs(batch))
        if response.stop_reason == "refusal":
            raise RuntimeError(f"judge request refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("judge response truncated (max_tokens); lower the batch size")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        by_id = {int(v["id"]): v for v in data["verdicts"]}
        verdicts = []
        for i in range(1, len(batch) + 1):
            v = by_id.get(i)
            if v is None:
                raise RuntimeError(f"judge response is missing pair {i}")
            verdicts.append(Judgment.from_dict(v))
        u = response.usage
        usage = Usage(
            model=self.model,
            n_pairs=len(batch),
            input_tokens=int(u.input_tokens),
            output_tokens=int(u.output_tokens),
            cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            elapsed_s=round(time.perf_counter() - t0, 3),
            request_id=str(getattr(response, "_request_id", "") or ""),
        )
        return verdicts, usage


# ---------------------------------------------------------------------------
# OpenAI-compatible judge (Qwen Cloud, Z.ai / GLM, ...) over the standard library
# ---------------------------------------------------------------------------

#: ``transport(url, headers, body, timeout) -> (status, body_bytes)``; raises :class:`TransportError`
#: on connection errors / timeouts.  Tests inject a fake; the default is :func:`urllib_transport`.
Transport = Callable[[str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


class TransportError(RuntimeError):
    """Connection failure or timeout (retried)."""


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https URL from config
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(str(exc)) from exc


def resolve_api_key(names: Sequence[str] | str = OPENAI_API_KEY_ENV, env: Mapping[str, str] | None = None) -> str:
    """The first non-empty environment variable among ``names`` (never hardcoded)."""
    env = os.environ if env is None else env
    names = (names,) if isinstance(names, str) else tuple(names)
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    raise RuntimeError(f"no API key: set {' or '.join(names)} in the environment")


def has_api_key_in(names: Sequence[str] | str) -> bool:
    names = (names,) if isinstance(names, str) else names
    return any(os.environ.get(n) for n in names)


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """The first JSON object or array in ``text`` (code fences and leading / trailing prose ignored)."""
    candidates = [m.strip() for m in _FENCE_RE.findall(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for i, ch in enumerate(chunk):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(chunk, i)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict | list):
                return value
    raise ValueError(f"no JSON object or array in judge response: {text[:200]!r}")


def validate_verdicts(data: Any, n: int, *, allow_partial: bool = False) -> list[Judgment] | dict[int, Judgment]:
    """``OUTPUT_SCHEMA`` semantics applied to a parsed response: one verdict per pair id ``1..n``.

    With ``allow_partial`` the validated verdicts are returned as ``{id: Judgment}`` and missing
    ids are left to the caller (used to re-judge them individually).
    """
    if isinstance(data, list):
        data = {"verdicts": data}
    if isinstance(data, dict) and not isinstance(data.get("verdicts"), list):
        # Models in JSON mode often rename the array ("results", "judgments", ...): accept a single list value.
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            data = {"verdicts": lists[0]}
    if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
        raise ValueError("judge response has no 'verdicts' array")
    by_id: dict[int, Judgment] = {}
    for idx, item in enumerate(data["verdicts"], 1):
        if isinstance(item, dict) and isinstance(item.get("verdict"), dict) and "confidence" not in item:
            item = {
                **item["verdict"],
                **{k: v for k, v in item.items() if k != "verdict"},
            }  # {"verdict": {...}} wrapper
        if not isinstance(item, dict):
            raise ValueError(f"verdict {idx} is not an object")
        if "relation" not in item:
            for alias in ("verdict", "label", "decision"):  # {"id": 1, "verdict": "equivalent"}
                if isinstance(item.get(alias), str):
                    item = {**item, "relation": item[alias]}
                    break
        if "confidence" not in item:
            item = {**item, "confidence": 0.5}  # no confidence reported: neutral
        try:
            raw_id = item.get("id", idx)
            try:
                vid = int(raw_id)
            except (TypeError, ValueError):
                vid = idx  # non-numeric id (e.g. an entity name): fall back to the position in the list
            conf = float(item["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"verdict {idx} lacks a numeric id / confidence: {item!r}") from exc
        rel = _canonical_relation(item.get("relation"))
        if rel is None and isinstance(item.get("equivalent"), bool):
            rel = "equivalent" if item["equivalent"] else "unrelated"  # relation omitted: derive it
        if "equivalent" not in item and rel in RELATIONS:
            item = {**item, "equivalent": rel == "equivalent"}
        elif isinstance(item.get("equivalent"), str):
            item = {**item, "equivalent": item["equivalent"].strip().lower() in ("true", "yes", "1")}
        if not isinstance(item.get("equivalent"), bool):
            raise ValueError(f"verdict {vid}: 'equivalent' must be a boolean")
        if rel not in RELATIONS:
            raise ValueError(f"verdict {vid}: relation {rel!r} not in {RELATIONS}")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"verdict {vid}: confidence {conf} outside [0, 1]")
        if vid in by_id:
            if allow_partial:
                continue  # JSON-mode models occasionally repeat an id; keep the first verdict
            raise ValueError(f"verdict id {vid} repeated")
        by_id[vid] = Judgment(item["equivalent"], conf, rel, str(item.get("rationale", "")))
    if allow_partial:
        return by_id
    missing = [i for i in range(1, n + 1) if i not in by_id]
    if missing:
        raise ValueError(f"judge response is missing pair(s) {missing}")
    return [by_id[i] for i in range(1, n + 1)]


class OpenAICompatJudge:
    """Judge over any OpenAI-compatible ``POST {base_url}/chat/completions`` server.

    ``provider`` is a short label (``qwen``, ``zai``) used for the cache filename, the results and
    the price table.  ``base_url`` falls back to ``LLM_JUDGE_BASE_URL``; the key is read from the
    environment variable(s) in ``api_key_env`` (default ``LLM_JUDGE_API_KEY``) unless ``api_key``
    is given.  Requests use ``temperature 0`` and, with ``json_mode``, ``response_format
    json_object`` (dropped automatically after a 400 that mentions ``response_format``);
    ``thinking`` adds the Z.ai-style ``{"type": "enabled" | "disabled"}`` field (``None`` = not
    sent; dropped automatically after a 400 that mentions it).  The answer is parsed with
    :func:`extract_json` (fences / prose tolerated; any ``reasoning_content`` ignored) and checked
    with :func:`validate_verdicts`.  429 / 5xx / transport errors are retried with exponential
    backoff (``max_retries``).  Instances are thread-safe.
    """

    def __init__(
        self,
        model: str = OPENAI_DEFAULT_MODEL,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: Sequence[str] | str = OPENAI_API_KEY_ENV,
        json_mode: bool = True,
        thinking: str | None = None,
        thinking_style: str = "auto",
        max_tokens: int = 16000,
        timeout: float = 180.0,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if thinking not in (None, "off", "on"):
            raise ValueError("thinking must be None, 'off' or 'on'")
        if thinking_style not in ("auto", "zai", "qwen"):
            raise ValueError("thinking_style must be 'auto', 'zai' or 'qwen'")
        if thinking_style == "auto":
            thinking_style = "qwen" if "qwen" in f"{provider} {model}".lower() else "zai"
        self.thinking_style = thinking_style
        base_url = base_url or os.environ.get(OPENAI_BASE_URL_ENV) or ""
        if not base_url:
            raise RuntimeError(f"no base URL: pass base_url or set {OPENAI_BASE_URL_ENV}")
        self.model = model
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else resolve_api_key(api_key_env)
        self.json_mode = json_mode
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.transport: Transport = transport if transport is not None else urllib_transport
        self.sleep = sleep
        self._lock = threading.Lock()
        self._partial_rejudged = 0
        self._unparseable = 0
        self.name = (
            f"{provider}[{model},json={'on' if json_mode else 'off'},thinking={thinking or 'unset'},{PROMPT_VERSION}]"
        )

    @classmethod
    def zai(cls, model: str = ZAI_DEFAULT_MODEL, **kwargs: Any) -> OpenAICompatJudge:
        """The Z.ai / Zhipu GLM preset: base URL from ``ZAI_BASE_URL``, key from ``ZAI_API_KEY`` / ``Z_AI_API_KEY``."""
        kwargs.setdefault("provider", "zai")
        kwargs.setdefault("base_url", os.environ.get("ZAI_BASE_URL") or ZAI_DEFAULT_BASE_URL)
        kwargs.setdefault("api_key_env", (*ZAI_API_KEY_ENV, OPENAI_API_KEY_ENV))
        kwargs.setdefault("thinking", "off")
        return cls(model, **kwargs)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request_body(self, batch: Sequence[JudgeRequest], nudge: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": render_batch(batch)
                    + "\n"
                    + OPENAI_OUTPUT_INSTRUCTION
                    + (
                        "\nYour previous answer did not follow this format. Answer again with ONLY the JSON object.\n"
                        if nudge
                        else "\n"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            if self.thinking_style == "qwen":
                # DashScope / Qwen compatible mode: a boolean switch; Qwen thinks by default.
                body["enable_thinking"] = self.thinking == "on"
            else:
                body["thinking"] = {"type": "enabled" if self.thinking == "on" else "disabled"}
        return body

    def _call(self, url: str, body: bytes | None) -> dict[str, Any]:
        """One request with retries on 429 / 5xx / transport errors; returns the decoded JSON body."""
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                status, raw = self.transport(url, self._headers(), body, self.timeout)
            except TransportError as exc:
                last = exc
            else:
                text = raw.decode("utf-8", errors="replace")
                if status == 200:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise HTTPStatusError(status, f"non-JSON body: {text}") from exc
                last = HTTPStatusError(status, text)
                if not (status == 429 or status >= 500):
                    raise last
            if attempt < self.max_retries:
                self.sleep(self.backoff_s * (2**attempt) + random.random() * 0.1)
        assert last is not None
        raise RuntimeError(f"{self.provider} request failed after {self.max_retries + 1} attempts: {last}") from last

    def _chat(self, batch: Sequence[JudgeRequest], nudge: bool = False) -> dict[str, Any]:
        """``_call`` with the one-time fallbacks: drop ``response_format`` / ``thinking`` on a 400 naming them."""
        url = f"{self.base_url}/chat/completions"
        for _ in range(3):
            body = json.dumps(self._request_body(batch, nudge=nudge)).encode("utf-8")
            try:
                return self._call(url, body)
            except HTTPStatusError as exc:
                if exc.status != 400:
                    raise
                with self._lock:
                    if self.json_mode and "response_format" in exc.body:
                        self.json_mode = False
                        continue
                    if self.thinking is not None and "thinking" in exc.body:
                        self.thinking = None
                        continue
                raise
        raise AssertionError("unreachable")

    def list_models(self) -> list[str]:
        """Model ids from ``GET {base_url}/models`` (OpenAI listing shape; some providers do not serve it)."""
        data = self._call(f"{self.base_url}/models", None)
        items = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError(f"unexpected models listing: {str(data)[:300]}")
        return sorted(str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in items)

    def judge(self, batch: Sequence[JudgeRequest], *, _retry: bool = False) -> tuple[list[Judgment], Usage]:
        t0 = time.perf_counter()
        data = self._chat(batch, nudge=_retry)
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"{self.provider} response has no choices: {str(data)[:300]}") from exc
        if choice.get("finish_reason") == "length":
            if len(batch) == 1:
                # A single-pair answer that never terminates is a runaway generation: abstain.
                with self._lock:
                    self._unparseable += 1
                verdict = Judgment(False, 0.0, "unrelated", "unparseable judge response: truncated at max_tokens")
                u = data.get("usage") or {}
                usage = Usage(
                    model=self.model,
                    n_pairs=1,
                    input_tokens=int(u.get("prompt_tokens", 0) or 0),
                    output_tokens=int(u.get("completion_tokens", 0) or 0),
                    provider=self.provider,
                    request_id=str(data.get("id", "")),
                    latency_s=time.perf_counter() - t0,
                )
                return [verdict], usage
            raise RuntimeError("judge response truncated (max_tokens); lower the batch size")
        content = (choice.get("message") or {}).get("content") or ""
        if not isinstance(content, str):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        try:
            by_id = validate_verdicts(extract_json(content), len(batch), allow_partial=True)
            assert isinstance(by_id, dict)
            missing = [i for i in range(1, len(batch) + 1) if i not in by_id]
            if missing and len(batch) == 1:
                raise ValueError("judge response is missing pair(s) [1]")
            for i in missing:
                # Models in JSON mode sometimes stop early; re-judge the leftovers one pair at a time.
                single, _ = self.judge([batch[i - 1]])
                by_id[i] = single[0]
                self._partial_rejudged += 1
            verdicts = [by_id[i] for i in range(1, len(batch) + 1)]
        except ValueError as exc:
            dump = os.environ.get("OLAYA_JUDGE_DEBUG_DIR")
            if dump:
                Path(dump).mkdir(parents=True, exist_ok=True)
                (Path(dump) / f"bad-response-{int(time.time() * 1000)}.txt").write_text(content, encoding="utf-8")
            if not _retry:
                # One re-ask with an explicit nudge; if still unparseable, abstain rather than kill the run.
                try:
                    return self.judge(batch, _retry=True)
                except ValueError:
                    pass
            if len(batch) == 1:
                with self._lock:
                    self._unparseable += 1
                verdicts = [Judgment(False, 0.0, "unrelated", f"unparseable judge response: {content[:120]!r}")]
            else:
                raise ValueError(f"{exc} (raw content starts: {content[:300]!r})") from exc
        u = data.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        usage = Usage(
            model=self.model,
            n_pairs=len(batch),
            input_tokens=int(u.get("prompt_tokens", 0) or 0),
            output_tokens=int(u.get("completion_tokens", 0) or 0),
            cache_read_input_tokens=int(details.get("cached_tokens", 0) or 0),
            elapsed_s=round(time.perf_counter() - t0, 3),
            request_id=str(data.get("id", "") or ""),
            provider=self.provider,
        )
        return verdicts, usage


def make_judge(kind: str, **kwargs: Any) -> Judge:
    if kind == "null":
        return NullJudge()
    if kind == "stub":
        return StubJudge(kwargs.pop("reference"), **kwargs)
    if kind == "anthropic":
        return AnthropicJudge(**kwargs)
    if kind == "openai":
        return OpenAICompatJudge(**kwargs)
    if kind == "zai":
        return OpenAICompatJudge.zai(**kwargs)
    raise ValueError(f"unknown judge {kind!r}")


# ---------------------------------------------------------------------------
# Judgment cache (JSON lines, committed)
# ---------------------------------------------------------------------------


class JudgmentCache:
    """Append-only JSONL store of verdicts keyed by :meth:`JudgeRequest.digest`."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._records[rec["key"]] = rec

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> Judgment | None:
        rec = self._records.get(key)
        return Judgment.from_dict(rec["verdict"]) if rec else None

    def put(self, key: str, req: JudgeRequest, model: str, verdict: Judgment, *, flush: bool = True) -> None:
        rec = {
            "key": key,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "source_ontology": req.source.ontology,
            "target_ontology": req.target.ontology,
            "source": req.source.name,
            "target": req.target.name,
            "kind": req.source.kind,
            "verdict": asdict(verdict),
        }
        with self._lock:
            if self._records.get(key) == rec:
                return  # idempotent: a verdict already stored is not appended twice
            self._records[key] = rec
            if flush:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")

    def rewrite(self) -> None:
        """Rewrite the file sorted by key (canonical form for committing)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for key in sorted(self._records):
                fh.write(json.dumps(self._records[key], sort_keys=True) + "\n")


def batched(items: Sequence[JudgeRequest], size: int) -> list[list[JudgeRequest]]:
    size = max(1, size)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def judge_candidates(
    cands: CandidateSet,
    source_cards: Mapping[str, EntityCard],
    target_cards: Mapping[str, EntityCard],
    judge: Judge,
    cache: JudgmentCache | None = None,
    *,
    batch_size: int = 15,
    ledger: list[Usage] | None = None,
    concurrency: int = 1,
) -> dict[tuple[str, str], Judgment]:
    """Verdict for every candidate; only pairs missing from ``cache`` are sent to ``judge``.

    With ``concurrency > 1`` the batches are sent from a thread pool (the judge must be
    thread-safe; :class:`OpenAICompatJudge` is); results, ledger and cache writes keep the batch order.
    """
    requests = [JudgeRequest(source_cards[c.source], target_cards[c.target]) for c in cands.candidates]
    out: dict[tuple[str, str], Judgment] = {}
    pending: list[tuple[str, JudgeRequest]] = []
    for req in requests:
        key = req.digest(judge.model)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            out[req.pair] = hit
        else:
            pending.append((key, req))
    # Deterministic order: by kind, then names (the candidates are already sorted by name).
    pending.sort(key=lambda kr: (kr[1].source.kind, kr[1].pair))
    batches = batched([r for _, r in pending], batch_size)
    if concurrency > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(judge.judge, batches))
    else:
        results = [judge.judge(b) for b in batches]
    for batch, (verdicts, usage) in zip(batches, results, strict=True):
        if ledger is not None:
            ledger.append(usage)
        for req, verdict in zip(batch, verdicts, strict=True):
            out[req.pair] = verdict
            if cache is not None:
                cache.put(req.digest(judge.model), req, judge.model, verdict)
    return out


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

DECODE_MODES = ("judge", "repair", "fused")


def decode(
    cands: CandidateSet,
    verdicts: Mapping[tuple[str, str], Judgment],
    *,
    mode: str,
    threshold: float,
    weight: float = 0.5,
) -> list[OAEIAlignment]:
    """Final alignment from the candidates, their hybrid scores and the verdicts.

    ``judge``: score = judge score (confidence when equivalent, else 0); mutual-best, threshold.
    ``repair``: the hybrid-accepted pairs minus those the judge rejects (not equivalent with
    confidence >= ``threshold``) keep their hybrid score raised to 1; every other candidate
    the judge accepts (equivalent, confidence >= ``threshold``) scores its confidence;
    mutual-best on the result (the threshold has already been applied).
    ``fused``: score = (1 - w) * hybrid + w * judge score; mutual-best, threshold.
    """
    scored: list[OAEIAlignment] = []
    if mode == "judge":
        for c in cands.candidates:
            s = verdicts[c.pair].score()
            if s > 0.0:
                scored.append(OAEIAlignment(c.source, c.target, "=", round(s, 6)))
        return apply_threshold(select_mutual_best(scored), threshold)
    if mode == "repair":
        for c in cands.candidates:
            v = verdicts[c.pair]
            if c.accepted:
                rejected = (not v.equivalent) and v.confidence >= threshold
                if not rejected:
                    scored.append(OAEIAlignment(c.source, c.target, "=", round(max(c.hybrid, v.score(), 0.5), 6)))
            elif v.equivalent and v.confidence >= threshold:
                scored.append(OAEIAlignment(c.source, c.target, "=", round(v.score(), 6)))
        return select_mutual_best(scored)
    if mode == "fused":
        for c in cands.candidates:
            s = (1.0 - weight) * c.hybrid + weight * verdicts[c.pair].score()
            if s > 0.0:
                scored.append(OAEIAlignment(c.source, c.target, "=", round(s, 6)))
        return apply_threshold(select_mutual_best(scored), threshold)
    raise ValueError(f"unknown decode mode {mode!r}")


def config_label(mode: str, threshold: float, weight: float = 0.5) -> str:
    if mode == "fused":
        return f"mode=fused,w={weight:.2f},th={threshold:.2f}"
    return f"mode={mode},th={threshold:.2f}"


def judge_quality(
    cands: CandidateSet,
    verdicts: Mapping[tuple[str, str], Judgment],
    reference: Iterable[OAEIAlignment],
    *,
    threshold: float = 0.5,
) -> dict[str, int]:
    """The judge's own TP / FP / FN on the candidate set (accept = equivalent and confidence >= threshold)."""
    gold = {(r.entity1, r.entity2) for r in reference}
    tp = fp = fn = 0
    for c in cands.candidates:
        v = verdicts[c.pair]
        accept = v.equivalent and v.confidence >= threshold
        if accept and c.pair in gold:
            tp += 1
        elif accept:
            fp += 1
        elif c.pair in gold:
            fn += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def cost_estimate(
    batches: Sequence[Sequence[JudgeRequest]],
    *,
    output_tokens_per_pair: int = 60,
    counted_input: Sequence[int] | None = None,
    prices: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-model cost estimate of judging ``batches`` (input from chars / 4 or ``counted_input``).

    ``prices`` defaults to the Anthropic table; pass ``PROVIDER_MODEL_PRICES_USD_PER_MTOK[provider]``.
    """
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    inputs = (
        list(counted_input)
        if counted_input is not None
        else [sys_tokens + estimate_tokens(render_batch(b)) for b in batches]
    )
    n_pairs = sum(len(b) for b in batches)
    total_in = sum(inputs)
    total_out = n_pairs * output_tokens_per_pair
    out = {}
    for model, (pin, pout) in (MODEL_PRICES_USD_PER_MTOK if prices is None else prices).items():
        out[model] = {
            "requests": len(batches),
            "pairs": n_pairs,
            "input_tokens": total_in,
            "output_tokens_assumed": total_out,
            "usd": round((total_in * pin + total_out * pout) / 1e6, 2),
            "method": "count_tokens" if counted_input is not None else "chars/4 estimate",
        }
    return out


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class PairJudgeResult:
    """Everything the benchmark keeps per ontology pair."""

    pair: str
    candidates: CandidateSet
    verdicts: dict[tuple[str, str], Judgment]
    candidate_recall: dict[str, float]
    judge_counts: dict[str, int] = field(default_factory=dict)


__all__ = [
    "DECODE_MODES",
    "DEFAULT_MODEL",
    "MODEL_PRICES_USD_PER_MTOK",
    "OPENAI_API_KEY_ENV",
    "OPENAI_BASE_URL_ENV",
    "OPENAI_DEFAULT_MODEL",
    "OUTPUT_SCHEMA",
    "PROVIDER_MODEL_PRICES_USD_PER_MTOK",
    "PROMPT_VERSION",
    "RELATIONS",
    "SUPPORTED_MODELS",
    "SYSTEM_PROMPT",
    "ZAI_API_KEY_ENV",
    "ZAI_DEFAULT_BASE_URL",
    "ZAI_DEFAULT_MODEL",
    "ZAI_PRICING_URL",
    "AnthropicJudge",
    "Candidate",
    "CandidateConfig",
    "CandidateGenerator",
    "CandidateSet",
    "EntityCard",
    "HTTPStatusError",
    "Judge",
    "JudgeRequest",
    "Judgment",
    "JudgmentCache",
    "NullJudge",
    "OpenAICompatJudge",
    "PairJudgeResult",
    "StubJudge",
    "Transport",
    "TransportError",
    "Usage",
    "batched",
    "config_label",
    "cost_estimate",
    "decode",
    "entity_cards",
    "estimate_tokens",
    "extract_json",
    "has_api_key",
    "has_api_key_in",
    "judge_candidates",
    "judge_quality",
    "make_judge",
    "price_for",
    "render_batch",
    "resolve_api_key",
    "topk_pairs",
    "urllib_transport",
    "validate_verdicts",
]
