"""LLM-distilled labelled corpus for the decision head, with leakage controls.

The corpus is built in two independent LLM calls per generating class:

1. **Generation**: a model is asked to write realistic observations (log lines, product
   descriptions, JSON-LD snippets, ...) that an operator or a system would see for *one*
   ontology class, without naming the class. Every observation that still contains the
   class label, its CURIE or its local name as a substring (case-insensitive) is dropped
   (:func:`label_leaks`), so the texts cannot be solved by label matching.
2. **Labelling**: a second call sees only the observation and a shuffled candidate list
   (the gold class, its siblings, the lexical router's top-8 and four random classes;
   never told which is gold) and picks one class with a confidence. An observation is
   kept only when the labeller agrees with the generating class; both verdicts are kept
   in the row for audit.

Every row records its provenance (models, prompt version, timestamps, candidate list,
labeller confidence). Rows are grouped by *generation batch* (one generation call), which
is the unit of the group-disjoint split (:func:`split_corpus`); the split is therefore
by generation batch, **not** by class: concept classification needs every class in
training, so a class usually appears in several partitions, and
:func:`split_report` states how many test classes were never seen in training.

Labels are LLM-distilled and filtered, not human. Agreement between a generator and a
labeller shows that the text is *recognisable* as the class it was written for; it does not
establish that the class is the one a human annotator would choose.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olaya.candidate_router import CandidateRouter
from olaya.evaluation import SourceObservation, split_observations, stable_id
from olaya.ontology import Concept, Ontology

PROMPT_VERSION = "corpus-v1"
SOURCE_KIND = "llm_distilled"

# Registers the generator is asked to cover; each generation batch gets one register so a
# batch (= one split group) is internally homogeneous and the split holds out whole registers.
REGISTERS: dict[str, tuple[str, ...]] = {
    "incident": (
        "terse monitoring alert or log line (timestamps, host names, error codes, metrics)",
        "prose paragraph written by an on-call engineer in a ticket or post-mortem",
        "structured JSON-ish event payload with keys such as service, error, metric, message",
    ),
    "gs1": (
        "terse packing slip, scan record or GTIN/EPCIS line (codes, quantities, dates)",
        "prose product description as it would appear in a catalogue or a retailer listing",
        "structured JSON-ish master-data record with attribute keys and values",
    ),
    "schemaorg": (
        "short snippet of visible web page text (headline, teaser, listing, caption)",
        "JSON-LD fragment with properties but WITHOUT any @type value",
        "one or two prose sentences describing the thing as a user or a crawler would",
    ),
}
DEFAULT_REGISTERS = REGISTERS["schemaorg"]

DOMAIN_SCENE: dict[str, str] = {
    "incident": "an operational incident seen by an SRE or on-call engineer (alerts, logs, tickets)",
    "gs1": "a trade item or logistics record in a GS1-style retail / supply-chain system",
    "schemaorg": "a web entity described on a page or in structured data",
}


# --- LLM client --------------------------------------------------------------------------


class LLMError(RuntimeError):
    """A request failed after retries (rate limit, server error, transport)."""


class RateLimited(LLMError):
    """The endpoint kept answering 429 after every retry."""


ChatFn = Callable[[list[dict[str, str]], str], str]


class OpenAICompatClient:
    """Minimal thread-safe ``chat/completions`` client with exponential backoff on 429 / 5xx.

    The key is read from the environment only (never logged). ``extra`` goes into every
    request body (for example ``{"enable_thinking": False}`` for Qwen models).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 6,
        backoff_s: float = 1.5,
        extra: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        from olaya.benchmarks.llm_judge import urllib_transport

        if not base_url or not api_key:
            raise RuntimeError("OpenAICompatClient needs a base URL and an API key from the environment")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.extra = dict(extra or {})
        self.sleep = sleep
        self.transport = urllib_transport
        self.rate_limit_hits = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # never expose the key
        return f"OpenAICompatClient(base_url={self.base_url!r})"

    def chat(
        self, messages: list[dict[str, str]], model: str, *, max_tokens: int = 2000, temperature: float = 0.0
    ) -> str:
        from olaya.benchmarks.llm_judge import HTTPStatusError, TransportError

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            **self.extra,
        }
        raw = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        last: Exception | None = None
        rate_limited = False
        for attempt in range(self.max_retries + 1):
            try:
                status, data = self.transport(url, headers, raw, self.timeout)
            except TransportError as exc:
                last = exc
            else:
                text = data.decode("utf-8", errors="replace")
                if status == 200:
                    payload = json.loads(text)
                    with self._lock:
                        self.calls += 1
                        usage = payload.get("usage") or {}
                        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                        self.completion_tokens += int(usage.get("completion_tokens") or 0)
                    return str(payload["choices"][0]["message"]["content"])
                last = HTTPStatusError(status, text)
                if status == 429:
                    rate_limited = True
                    with self._lock:
                        self.rate_limit_hits += 1
                elif status == 400 and "response_format" in text:
                    body.pop("response_format", None)
                    raw = json.dumps(body).encode("utf-8")
                elif status < 500:
                    raise LLMError(str(last)) from last
            if attempt < self.max_retries:
                self.sleep(self.backoff_s * (2**attempt) + random.random() * 0.5)
        if rate_limited:
            raise RateLimited(f"rate limited after {self.max_retries + 1} attempts") from last
        raise LLMError(f"request failed after {self.max_retries + 1} attempts: {last}") from last


# --- class cards and sampling ----------------------------------------------------------------


def concept_depth(ontology: Ontology, curie: str) -> int:
    """Longest parent path depth; cycles count as depth 0 beyond the cycle."""
    memo: dict[str, int] = {}

    def depth(c: str, stack: frozenset[str]) -> int:
        if c in memo:
            return memo[c]
        parents = [p for p in ontology.concepts[c].parents if p in ontology.concepts and p not in stack]
        value = 1 + max((depth(p, stack | {c}) for p in parents), default=-1)
        memo[c] = value
        return value

    return depth(curie, frozenset())


def eligible_classes(ontology: Ontology) -> list[str]:
    """Classes with a usable label, sorted by CURIE."""
    return sorted(
        c
        for c, concept in ontology.concepts.items()
        if concept.label and len(concept.label.strip()) > 1 and not concept.label.strip().isdigit()
    )


def sample_classes(ontology: Ontology, max_classes: int | None, seed: int = 42) -> list[str]:
    """All eligible classes, or ``max_classes`` of them stratified across hierarchy depth."""
    pool = eligible_classes(ontology)
    if max_classes is None or len(pool) <= max_classes:
        return pool
    by_depth: dict[int, list[str]] = {}
    for c in pool:
        by_depth.setdefault(concept_depth(ontology, c), []).append(c)
    rng = random.Random(seed)
    for bucket in by_depth.values():
        rng.shuffle(bucket)
    chosen: list[str] = []
    # Round-robin over depths so every depth is represented, deepest buckets get no priority.
    while len(chosen) < max_classes:
        progressed = False
        for d in sorted(by_depth):
            if by_depth[d] and len(chosen) < max_classes:
                chosen.append(by_depth[d].pop())
                progressed = True
        if not progressed:
            break
    return sorted(chosen)


def class_card(ontology: Ontology, curie: str) -> str:
    """Human-readable description of a class and its neighbourhood for prompts."""
    c = ontology.concepts[curie]
    parents = [ontology.concepts[p].label for p in sorted(c.parents) if p in ontology.concepts]
    children = [ontology.concepts[k].label for k in sorted(c.children) if k in ontology.concepts]
    siblings = sorted(
        {ontology.concepts[s].label for p in c.parents if p in ontology.concepts for s in ontology.concepts[p].children}
        - {c.label}
    )
    lines = [f"Class: {c.label} ({curie})"]
    if c.description:
        lines.append(f"Definition: {c.description}")
    if c.synonyms:
        lines.append(f"Synonyms: {', '.join(c.synonyms)}")
    if parents:
        lines.append(f"Parent classes: {', '.join(parents)}")
    if children:
        lines.append(f"Subclasses: {', '.join(children[:12])}")
    if siblings:
        lines.append(f"Sibling classes (do NOT describe these): {', '.join(siblings[:12])}")
    return "\n".join(lines)


def _local_name(curie: str) -> str:
    return curie.split(":", 1)[1] if ":" in curie else curie


def label_leaks(text: str, concept: Concept) -> bool:
    """True when the text contains the class label, CURIE or local name as a substring."""
    hay = text.casefold()
    needles = {concept.label, concept.curie, _local_name(concept.curie)}
    needles |= {s for s in concept.synonyms if len(s) > 3}
    # Also the space-separated form of CamelCase labels / local names ("DatabaseOutage" -> "database outage").
    for n in list(needles):
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", n)
        needles.add(spaced)
        needles.add(n.replace("_", " "))
    return any(n and n.casefold() in hay for n in needles)


# --- prompts and parsing ---------------------------------------------------------------------


def generation_messages(
    ontology: Ontology, curie: str, domain: str, register: str, n: int, batch_seed: int
) -> list[dict[str, str]]:
    scene = DOMAIN_SCENE.get(domain, "a record a system or an operator would see")
    c = ontology.concepts[curie]
    system = (
        "You write realistic synthetic observations for a classification corpus. You answer with a JSON object only."
    )
    user = (
        f"Domain: {scene}.\n\n{class_card(ontology, curie)}\n\n"
        f"Write {n} distinct, realistic observations that are instances of this class and that a "
        f"system or operator would actually see. Register for all of them: {register}.\n"
        "Rules:\n"
        f"- NEVER write the class name '{c.label}', its identifier '{curie}' or '{_local_name(curie)}' "
        "or any close paraphrase of the name; describe concrete, specific content instead "
        "(names, numbers, places, codes, symptoms, properties).\n"
        "- Each observation must be clearly this class and not one of its sibling classes.\n"
        "- Vary wording, length (20-80 words), entities and details between observations; "
        f"variation seed {batch_seed}.\n"
        '- Answer as {"observations": ["...", "..."]} and nothing else.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_generations(text: str) -> list[str]:
    from olaya.benchmarks.llm_judge import extract_json

    data = extract_json(text)
    items = data.get("observations", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("generation answer has no observation list")
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("text") or item.get("observation") or json.dumps(item, ensure_ascii=False)
        s = " ".join(str(item).split())
        if len(s) >= 15:
            out.append(s)
    return out


def candidate_list(
    ontology: Ontology, router: CandidateRouter, gold: str, text: str, rng: random.Random, pool: Sequence[str]
) -> list[str]:
    """Gold + siblings + lexical top-8 + 4 random classes, de-duplicated and shuffled."""
    c = ontology.concepts[gold]
    siblings = sorted({s for p in c.parents if p in ontology.concepts for s in ontology.concepts[p].children} - {gold})
    routed = [concept.curie for concept, _ in router.route_candidates(text, top_k=8)]
    others = [x for x in pool if x != gold]
    randoms = rng.sample(others, min(4, len(others)))
    seen: list[str] = []
    for x in [gold, *siblings[:8], *routed, *randoms]:
        if x in ontology.concepts and x not in seen:
            seen.append(x)
    rng.shuffle(seen)
    return seen


def labelling_messages(ontology: Ontology, text: str, candidates: Sequence[str], domain: str) -> list[dict[str, str]]:
    scene = DOMAIN_SCENE.get(domain, "a record")
    lines = []
    for i, curie in enumerate(candidates):
        c = ontology.concepts[curie]
        parents = [ontology.concepts[p].label for p in sorted(c.parents) if p in ontology.concepts]
        desc = f" - {c.description}" if c.description else ""
        par = f" (subclass of {', '.join(parents)})" if parents else ""
        lines.append(f"{i}. {curie}: {c.label}{par}{desc}")
    system = "You are a careful ontology annotator. You answer with a JSON object only."
    user = (
        f'Context: {scene}.\n\nObservation:\n"""\n{text}\n"""\n\n'
        "Candidate classes (exactly one is the best, most specific fit; pick 'none' only if no "
        "candidate fits at all):\n" + "\n".join(lines) + "\n\n"
        'Answer as {"index": <candidate number or -1>, "curie": "<curie or none>", '
        '"confidence": <0.0-1.0>, "reason": "<one short sentence>"}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_label(text: str, candidates: Sequence[str]) -> tuple[str | None, float, str]:
    from olaya.benchmarks.llm_judge import extract_json

    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("label answer is not an object")
    curie = data.get("curie")
    idx = data.get("index")
    chosen: str | None = None
    if isinstance(curie, str) and curie in candidates:
        chosen = curie
    elif isinstance(idx, int) and 0 <= idx < len(candidates):
        chosen = candidates[idx]
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    return chosen, conf, str(data.get("reason", ""))[:300]


# --- rows ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRow:
    """One kept observation with full provenance."""

    row_id: str
    domain: str
    target_curie: str
    text: str
    register: str
    batch_id: str
    batch_seed: int
    generator_model: str
    labeller_model: str
    prompt_version: str
    generated_at: str
    labelled_at: str
    candidates: list[str]
    labeller_curie: str
    labeller_confidence: float
    labeller_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CorpusRow:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class BuildStats:
    classes: int = 0
    generation_calls: int = 0
    generated: int = 0
    leaked: int = 0
    duplicates: int = 0
    labelling_calls: int = 0
    disagreed: int = 0
    label_failures: int = 0
    kept: int = 0
    rate_limited: bool = False
    stopped_early: str = ""
    generation_errors: int = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_corpus(
    ontology: Ontology,
    domain: str,
    chat: ChatFn,
    *,
    generator_model: str,
    labeller_model: str,
    classes: Sequence[str] | None = None,
    batches_per_class: int = 3,
    per_batch: int = 3,
    seed: int = 42,
    concurrency: int = 4,
    registers: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[CorpusRow], BuildStats]:
    """Generate, filter and independently label observations for every class.

    ``chat(messages, model) -> str`` is the only I/O. Generation batches run concurrently; a
    :class:`RateLimited` error stops the build early (``stats.stopped_early``) and returns
    whatever was kept so far.
    """
    classes = list(classes) if classes is not None else eligible_classes(ontology)
    registers = tuple(registers or REGISTERS.get(domain, DEFAULT_REGISTERS))
    pool = eligible_classes(ontology)
    router = CandidateRouter(ontology)
    stats = BuildStats(classes=len(classes))
    stop = threading.Event()
    lock = threading.Lock()
    seen_text: set[str] = set()
    rows: list[CorpusRow] = []

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    def one_batch(curie: str, b: int) -> None:
        if stop.is_set():
            return
        register = registers[b % len(registers)]
        batch_seed = seed * 1000 + b
        batch_id = stable_id("batch", domain, curie, str(batch_seed), register)
        gen_at = _now()
        try:
            texts = parse_generations(
                chat(generation_messages(ontology, curie, domain, register, per_batch, batch_seed), generator_model)
            )
        except RateLimited as exc:
            with lock:
                stats.rate_limited = True
                stats.stopped_early = f"generation rate-limited: {exc}"
            stop.set()
            return
        except (LLMError, ValueError, KeyError) as exc:
            with lock:
                stats.generation_errors += 1
            say(f"generation failed for {curie}: {exc}")
            return
        with lock:
            stats.generation_calls += 1
            stats.generated += len(texts)
        concept = ontology.concepts[curie]
        rng = random.Random(stable_id("cands", domain, curie, str(batch_seed)))
        for text in texts:
            if stop.is_set():
                return
            if label_leaks(text, concept):
                with lock:
                    stats.leaked += 1
                continue
            key = " ".join(text.casefold().split())
            with lock:
                if key in seen_text:
                    stats.duplicates += 1
                    continue
                seen_text.add(key)
            cands = candidate_list(ontology, router, curie, text, rng, pool)
            try:
                answer = chat(labelling_messages(ontology, text, cands, domain), labeller_model)
                chosen, conf, reason = parse_label(answer, cands)
            except RateLimited as exc:
                with lock:
                    stats.rate_limited = True
                    stats.stopped_early = f"labelling rate-limited: {exc}"
                stop.set()
                return
            except (LLMError, ValueError, KeyError) as exc:
                with lock:
                    stats.label_failures += 1
                say(f"labelling failed: {exc}")
                continue
            with lock:
                stats.labelling_calls += 1
                if chosen != curie:
                    stats.disagreed += 1
                    continue
                stats.kept += 1
                rows.append(
                    CorpusRow(
                        row_id=stable_id("row", domain, curie, key),
                        domain=domain,
                        target_curie=curie,
                        text=text,
                        register=register,
                        batch_id=batch_id,
                        batch_seed=batch_seed,
                        generator_model=generator_model,
                        labeller_model=labeller_model,
                        prompt_version=PROMPT_VERSION,
                        generated_at=gen_at,
                        labelled_at=_now(),
                        candidates=list(cands),
                        labeller_curie=chosen,
                        labeller_confidence=conf,
                        labeller_reason=reason,
                    )
                )

    jobs = [(c, b) for c in classes for b in range(batches_per_class)]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool_exec:
        for i, _ in enumerate(pool_exec.map(lambda job: one_batch(*job), jobs), 1):
            if i % 20 == 0:
                say(f"{domain}: {i}/{len(jobs)} batches, kept {stats.kept}")
    rows.sort(key=lambda r: r.row_id)
    return rows, stats


# --- persistence, hashing, splitting ---------------------------------------------------------


def write_jsonl(rows: Iterable[CorpusRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[CorpusRow]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(CorpusRow.from_dict(json.loads(line)))
    return rows


def corpus_sha256(rows: Sequence[CorpusRow]) -> str:
    """Order-independent hash over the evidence fields (not the timestamps)."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.row_id):
        digest.update(stable_id(row.row_id, row.domain, row.target_curie, row.text, row.batch_id).encode())
    return digest.hexdigest()


def to_observations(rows: Iterable[CorpusRow]) -> list[SourceObservation]:
    """Source observations grouped by generation batch (the split unit)."""
    return [SourceObservation(row.row_id, row.batch_id, row.target_curie, row.text, SOURCE_KIND) for row in rows]


def split_corpus(rows: Sequence[CorpusRow], seed: int = 42) -> dict[str, list[CorpusRow]]:
    """Group-disjoint 60/15/10/15 split by generation batch via :func:`split_observations`."""
    by_id = {row.row_id: row for row in rows}
    splits = split_observations(to_observations(rows), seed=seed)
    return {name: [by_id[o.source_id] for o in part] for name, part in splits.items()}


def split_corpus_by_class(rows: Sequence[CorpusRow], seed: int = 42) -> dict[str, list[CorpusRow]]:
    """Class-disjoint 60/15/10/15 split: every row of a class lands in the same part.

    The group is the target class, so no test class was seen in training. This measures transfer of
    a trained encoder to classes it never saw, the question the batch-disjoint split cannot answer.
    Requires at least four classes.
    """
    by_id = {row.row_id: row for row in rows}
    observations = [
        SourceObservation(row.row_id, f"class:{row.target_curie}", row.target_curie, row.text, SOURCE_KIND)
        for row in rows
    ]
    splits = split_observations(observations, seed=seed)
    return {name: [by_id[o.source_id] for o in part] for name, part in splits.items()}


def split_report(splits: Mapping[str, Sequence[CorpusRow]], unit: str = "batch") -> dict[str, Any]:
    train_classes = {r.target_curie for r in splits.get("train", ())}
    units = {
        "batch": "generation batch (one LLM generation call); classes are NOT held out",
        "class": "target class; every test class is unseen in training",
    }
    report: dict[str, Any] = {"unit": units.get(unit, unit)}
    for name, part in splits.items():
        classes = {r.target_curie for r in part}
        report[name] = {
            "rows": len(part),
            "batches": len({r.batch_id for r in part}),
            "classes": len(classes),
            "classes_unseen_in_train": sorted(classes - train_classes) if name != "train" else [],
        }
    return report


def corpus_manifest(
    ontology: Ontology,
    ontology_path: Path | None,
    rows: Sequence[CorpusRow],
    splits: Mapping[str, Sequence[CorpusRow]],
    stats: BuildStats | None,
) -> dict[str, Any]:
    from olaya.evaluation import dataset_manifest

    obs_splits = {name: to_observations(part) for name, part in splits.items()}
    manifest = dataset_manifest(ontology, obs_splits)
    manifest.update(
        {
            "corpus_sha256": corpus_sha256(rows),
            "rows": len(rows),
            "classes": len({r.target_curie for r in rows}),
            "ontology_file_sha256": hashlib.sha256(ontology_path.read_bytes()).hexdigest() if ontology_path else None,
            "prompt_version": PROMPT_VERSION,
            "generator_models": sorted({r.generator_model for r in rows}),
            "labeller_models": sorted({r.labeller_model for r in rows}),
            "build_stats": asdict(stats) if stats else None,
            "split_report": split_report(splits),
            "label_provenance": "LLM-distilled: generated for a class, label-substring filtered, kept only when an independent labeller call agreed. Not human-labelled.",
        }
    )
    for part in manifest["splits"].values():
        part.pop("source_ids", None)  # row ids are in the JSONL; keep the manifest small
    return manifest
