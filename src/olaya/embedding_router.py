"""Embedding candidate router, hybrid lexical+embedding router and zero-shot scorer.

The lexical :class:`~olaya.candidate_router.CandidateRouter` is the ceiling of the deployed
``concept`` path: ``OLaya.system_one`` scores only the most specific routed concepts, so a class the
router never surfaces cannot be answered. This module adds three pieces that plug into the engine
without changing its projection (ontology kernel, lineage, ``axioms_verified``):

* :class:`EmbeddingRouter` encodes every class once with *path-context* text (label, description,
  path from the root, synonyms; see :func:`class_context_text`) through any
  :class:`~olaya.embeddings.Encoder`, encodes the serialised observation state at query time and
  ranks classes by cosine similarity, optionally hubness-corrected with CSLS
  (Conneau et al. 2018; the class-side background is the mean similarity of a class to its ``k``
  nearest *classes*, so it needs no query set).
* :class:`HybridRouter` fuses the lexical and the embedding rankings, either by interleaving the
  two top lists (``mode="union"``) or by reciprocal-rank fusion (``mode="rrf"``).
* :class:`EmbeddingScorer` turns a router's similarity into candidate logits for the engine
  (``logit = similarity / temperature``); ``OLaya(scorer=...)`` feeds those logits to the existing
  hierarchical projection instead of the decision head's logits for ``concept`` questions.

:func:`finetune_bi_encoder` fine-tunes the encoder contrastively (in-batch negatives plus sibling
hard negatives) on a labelled split under a wall-clock budget, and :func:`load_finetuned_encoder`
reloads the saved weights on top of the base checkpoint.

Nothing here downloads anything: the encoder is injected, and tests use
:class:`~olaya.embeddings.HashEncoder`.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from olaya.candidate_router import CandidateRouter, tokenize_normalize
from olaya.embeddings import Encoder
from olaya.ontology import Concept, Ontology

DEFAULT_SCORER_TEMPERATURE = 0.05
_MAX_SYNONYMS = 8


# --- class texts --------------------------------------------------------------------------------


def camel_to_words(name: str) -> str:
    """``"DatabaseOutage"`` -> ``"Database Outage"``; local name of a CURIE when given one."""
    local = name.split(":", 1)[1] if ":" in name else name
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", local.replace("_", " "))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return " ".join(s.split())


def _display(ontology: Ontology, curie: str) -> str:
    c = ontology.concepts.get(curie)
    if c is not None and c.label and c.label.strip():
        return c.label.strip()
    return camel_to_words(curie)


def root_path(ontology: Ontology, curie: str) -> list[str]:
    """Labels from a root down to ``curie`` along the lexicographically first parent chain."""
    chain = [curie]
    seen = {curie}
    cur = curie
    while True:
        concept = ontology.concepts.get(cur)
        parents = sorted(p for p in (concept.parents if concept else ()) if p not in seen)
        if not parents:
            break
        cur = parents[0]
        seen.add(cur)
        chain.append(cur)
    return [_display(ontology, c) for c in reversed(chain)]


def class_context_text(ontology: Ontology, curie: str) -> str:
    """Path-context description of a class, mirroring the OAEI ``EntityText.rich`` recipe.

    ``<label>. <description>. Path: Root > ... > label. Also called <synonyms>.`` Siblings and
    children are deliberately left out: they would pull a class towards its neighbours, which is
    exactly what a router must separate.
    """
    concept = ontology.concepts[curie]
    parts = [_display(ontology, curie) + "."]
    if concept.description:
        parts.append(concept.description.strip().rstrip(".") + ".")
    path = root_path(ontology, curie)
    if len(path) > 1:
        parts.append("Path: " + " > ".join(path) + ".")
    synonyms = [s.strip() for s in concept.synonyms if s and s.strip()]
    if synonyms:
        parts.append("Also called " + ", ".join(sorted(set(synonyms))[:_MAX_SYNONYMS]) + ".")
    return " ".join(parts)


def build_class_texts(ontology: Ontology) -> dict[str, str]:
    """``{curie: class_context_text}`` for every concept, in sorted CURIE order."""
    return {curie: class_context_text(ontology, curie) for curie in sorted(ontology.concepts)}


# --- routers ------------------------------------------------------------------------------------


def _ranked(curies: Sequence[str], scores: np.ndarray, allowed: set[str] | None) -> list[tuple[str, float]]:
    """Descending by score, ties broken by CURIE; restricted to ``allowed`` when given."""
    pairs = [(c, float(s)) for c, s in zip(curies, scores, strict=True) if allowed is None or c in allowed]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs


class EmbeddingRouter:
    """Rank ontology classes by encoder similarity between the state text and class texts.

    ``csls_k`` enables the hubness correction: ``score = 2 cos - r_query - r_class`` where
    ``r_query`` is the mean of the query's ``k`` best class similarities and ``r_class`` the mean of
    the class's ``k`` best similarities to *other classes* (precomputed, gold-free). ``r_query`` is
    constant for one query, so only ``r_class`` changes the ranking; it demotes classes that are
    close to everything (generic parents, boilerplate descriptions).
    """

    def __init__(
        self,
        ontology: Ontology,
        encoder: Encoder,
        *,
        max_candidates: int = 12,
        csls_k: int | None = None,
        class_texts: Mapping[str, str] | None = None,
    ):
        if not ontology.concepts:
            raise ValueError("EmbeddingRouter needs a non-empty ontology")
        self.ontology = ontology
        self.encoder = encoder
        self.max_candidates = max_candidates
        self.csls_k = csls_k
        texts = dict(class_texts) if class_texts is not None else build_class_texts(ontology)
        self.curies: list[str] = sorted(ontology.concepts)
        self.class_texts: dict[str, str] = {c: texts.get(c) or class_context_text(ontology, c) for c in self.curies}
        self.index: dict[str, int] = {c: i for i, c in enumerate(self.curies)}
        # Prompted encoders embed class texts on the document side and states on the query side.
        encode_docs = getattr(encoder, "encode_documents", None) or encoder.encode
        self.matrix: np.ndarray = np.asarray(encode_docs([self.class_texts[c] for c in self.curies]), dtype=np.float32)
        self.class_background: np.ndarray | None = None
        if csls_k is not None:
            if csls_k < 1:
                raise ValueError("csls_k must be >= 1")
            sim = self.matrix @ self.matrix.T
            np.fill_diagonal(sim, -np.inf)
            k = min(csls_k, max(1, len(self.curies) - 1))
            top = np.sort(sim, axis=1)[:, -k:]
            top = np.where(np.isfinite(top), top, 0.0)
            self.class_background = top.mean(axis=1).astype(np.float32)
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        tag = getattr(self.encoder, "name", type(self.encoder).__name__)
        return f"embedding:{tag}" + (f"+csls{self.csls_k}" if self.csls_k else "")

    def encode_query(self, state_text: str) -> np.ndarray:
        vec = self._query_cache.get(state_text)
        if vec is None:
            vec = np.asarray(self.encoder.encode([state_text]), dtype=np.float32)[0]
            if len(self._query_cache) > 4096:
                self._query_cache.clear()
            self._query_cache[state_text] = vec
        return vec

    def similarities(self, state_text: str) -> np.ndarray:
        """Similarity of the state to every class (``self.curies`` order), CSLS-adjusted when enabled."""
        cos = self.matrix @ self.encode_query(state_text)
        if self.class_background is None:
            return cos
        k = min(self.csls_k or 1, len(cos))
        r_query = float(np.sort(cos)[-k:].mean())
        return 2.0 * cos - r_query - self.class_background

    def _allowed(self, root_curie: str | None) -> set[str] | None:
        if root_curie is None:
            return None
        if root_curie not in self.ontology.concepts:
            raise ValueError(f"unknown root concept {root_curie!r}")
        return self.ontology.get_descendants(root_curie)

    def rank(self, state_text: str, root_curie: str | None = None) -> list[tuple[str, float]]:
        """Every (allowed) class with its score, best first; ties broken by CURIE."""
        return _ranked(self.curies, self.similarities(state_text), self._allowed(root_curie))

    def route_candidates(
        self, state_text: str, root_curie: str | None = None, top_k: int | None = None
    ) -> list[tuple[Concept, float]]:
        k = top_k or self.max_candidates
        return [(self.ontology.concepts[c], s) for c, s in self.rank(state_text, root_curie)[:k]]

    def extract_decision_subgraph(
        self, state_text: str, root_curie: str | None = None, max_candidates: int | None = None
    ) -> Ontology:
        limit = max_candidates or self.max_candidates
        curies = [c.curie for c, _ in self.route_candidates(state_text, root_curie=root_curie, top_k=limit)]
        return self.ontology.extract_subgraph(curies, include_parents=True)

    def score_candidates(self, state_text: str, candidate_curies: Sequence[str]) -> np.ndarray:
        """Similarity of the state to each candidate (unknown CURIEs score the minimum of the known ones)."""
        sims = self.similarities(state_text)
        floor = float(sims.min()) if len(sims) else 0.0
        return np.array([float(sims[self.index[c]]) if c in self.index else floor for c in candidate_curies])


def lexical_rank(router: CandidateRouter, state_text: str, root_curie: str | None = None) -> list[tuple[str, float]]:
    """The lexical router's full ranking, empty when no query token hits the index (no fallback)."""
    evidence = any(router.inverted_index.get(t) for t in tokenize_normalize(state_text))
    if not evidence:
        return []
    ranked = router.route_candidates(state_text, root_curie=root_curie, top_k=len(router.ontology.concepts))
    return [(c.curie, float(s)) for c, s in ranked]


class HybridRouter:
    """Fuse the lexical and the embedding rankings.

    ``mode="union"`` interleaves the two top lists (lexical first, then embedding, ...) and
    de-duplicates, so the top-``k`` holds roughly ``k/2`` of each; the score is ``1 / (1 + position)``.
    ``mode="rrf"`` is reciprocal-rank fusion: ``w / (rrf_k + rank_lex) + (1 - w) / (rrf_k + rank_emb)``,
    where a class the lexical router has no evidence for only receives the embedding term.
    Scoring (:meth:`score_candidates`) always uses the embedding similarity; the fusion is for routing.
    """

    def __init__(
        self,
        lexical: CandidateRouter,
        embedding: EmbeddingRouter,
        *,
        mode: str = "rrf",
        max_candidates: int = 12,
        rrf_k: int = 60,
        lexical_weight: float = 0.5,
    ):
        if mode not in {"union", "rrf"}:
            raise ValueError(f"unknown hybrid mode {mode!r}")
        if lexical.ontology is not embedding.ontology and set(lexical.ontology.concepts) != set(
            embedding.ontology.concepts
        ):
            raise ValueError("lexical and embedding routers must index the same ontology")
        self.lexical = lexical
        self.embedding = embedding
        self.ontology = embedding.ontology
        self.mode = mode
        self.max_candidates = max_candidates
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight

    @property
    def name(self) -> str:
        return f"hybrid-{self.mode}({self.embedding.name})"

    def rank(self, state_text: str, root_curie: str | None = None) -> list[tuple[str, float]]:
        lex = lexical_rank(self.lexical, state_text, root_curie)
        emb = self.embedding.rank(state_text, root_curie)
        if self.mode == "union":
            seen: dict[str, None] = {}
            for i in range(max(len(lex), len(emb))):
                for lst in (lex, emb):
                    if i < len(lst):
                        seen.setdefault(lst[i][0], None)
            return [(c, 1.0 / (1 + pos)) for pos, c in enumerate(seen)]
        w = self.lexical_weight
        fused: dict[str, float] = {}
        for r, (c, _) in enumerate(lex):
            fused[c] = fused.get(c, 0.0) + w / (self.rrf_k + r + 1)
        for r, (c, _) in enumerate(emb):
            fused[c] = fused.get(c, 0.0) + (1.0 - w) / (self.rrf_k + r + 1)
        return sorted(fused.items(), key=lambda x: (-x[1], x[0]))

    def route_candidates(
        self, state_text: str, root_curie: str | None = None, top_k: int | None = None
    ) -> list[tuple[Concept, float]]:
        k = top_k or self.max_candidates
        return [(self.ontology.concepts[c], s) for c, s in self.rank(state_text, root_curie)[:k]]

    def extract_decision_subgraph(
        self, state_text: str, root_curie: str | None = None, max_candidates: int | None = None
    ) -> Ontology:
        limit = max_candidates or self.max_candidates
        curies = [c.curie for c, _ in self.route_candidates(state_text, root_curie=root_curie, top_k=limit)]
        return self.ontology.extract_subgraph(curies, include_parents=True)

    def score_candidates(self, state_text: str, candidate_curies: Sequence[str]) -> np.ndarray:
        return self.embedding.score_candidates(state_text, candidate_curies)


# --- scorer -------------------------------------------------------------------------------------


class EmbeddingScorer:
    """Zero-shot candidate scorer for ``OLaya``: ``logits = router.score_candidates(...) / temperature``.

    The engine applies the temperature with its usual rule (pre-multiplying by the cardinality
    temperature so the projection's division yields ``softmax(similarity / temperature)``). Fit
    ``temperature`` on a calibration split (``scripts/evaluate_routers.py``); the default is the
    conventional contrastive value for cosine similarities.
    """

    def __init__(self, router: Any, temperature: float = DEFAULT_SCORER_TEMPERATURE, name: str | None = None):
        if not hasattr(router, "score_candidates"):
            raise TypeError("scorer router must provide score_candidates(state_text, candidate_curies)")
        if not (temperature > 0.0 and np.isfinite(temperature)):
            raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
        self.router = router
        self.temperature = float(temperature)
        self.name = name or f"scorer({getattr(router, 'name', type(router).__name__)})"

    def score(self, state_text: str, candidate_curies: Sequence[str]) -> np.ndarray:
        scores = np.asarray(self.router.score_candidates(state_text, candidate_curies), dtype=np.float64)
        if scores.shape != (len(candidate_curies),):
            raise ValueError(f"scorer returned shape {scores.shape} for {len(candidate_curies)} candidates")
        return scores


# --- fine-tuning --------------------------------------------------------------------------------


@dataclass
class FineTuneResult:
    checkpoint: Path
    sha256: str
    base_model: str
    epochs_run: int
    runtime_s: float
    history: list[dict[str, Any]] = field(default_factory=list)
    best_validation: float = -1.0
    best_epoch: int = 0


def sibling_curies(ontology: Ontology, curie: str) -> list[str]:
    concept = ontology.concepts[curie]
    out: set[str] = set()
    for p in concept.parents:
        if p in ontology.concepts:
            out |= ontology.concepts[p].children
    out.discard(curie)
    return sorted(out)


def mined_hard_negatives(
    query_emb: np.ndarray, class_emb: np.ndarray, class_curies: Sequence[str], golds: Sequence[str], per_query: int
) -> list[list[str]]:
    """For each query, the ``per_query`` highest-scoring classes that are not its gold class.

    Pure ranking over the given (normalised) embeddings, so the result is deterministic and the
    function can be tested without a model.
    """
    if per_query <= 0:
        return [[] for _ in golds]
    sims = query_emb @ class_emb.T
    out: list[list[str]] = []
    for i, gold in enumerate(golds):
        order = np.argsort(-sims[i], kind="stable")
        picked = [class_curies[j] for j in order if class_curies[j] != gold][:per_query]
        out.append(picked)
    return out


def finetune_bi_encoder(
    encoder: Any,
    ontology: Ontology | None,
    train_pairs: Sequence[tuple[str, str]],
    *,
    checkpoint: Path,
    validate: Callable[[Any], float] | None = None,
    budget_s: float = 900.0,
    max_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-5,
    temperature: float = 0.05,
    seed: int = 42,
    class_texts: Mapping[str, str] | None = None,
    hard_negatives: Callable[[str], Sequence[str]] | None = None,
    mine_hard_negatives: int = 0,
    progress: Callable[[str], None] | None = None,
) -> FineTuneResult:
    """Contrastive fine-tuning of a :class:`~olaya.embeddings.TransformerEncoder` in place.

    ``train_pairs`` are ``(state_text, gold_curie)``. Each step encodes a batch of queries, the
    class text of every gold class in the batch and one sibling class per query as a hard negative;
    the loss is InfoNCE over the batch's gold texts plus the hard negatives (in-batch negatives are
    the other queries' gold classes). ``validate(encoder) -> float`` is called after every epoch and
    the best epoch's weights are saved to ``checkpoint`` (the final weights of the encoder are the
    best ones). Stops at ``budget_s`` wall-clock seconds.

    ``hard_negatives(gold_curie)`` names the structural hard negatives to sample one from (default:
    the gold class's siblings in ``ontology``; required when ``ontology`` is None, e.g. for a joint
    run over several ontologies whose ``class_texts`` are passed as one mapping). With
    ``mine_hard_negatives = m > 0``, the encoder's own top-``m`` non-gold classes for every training
    query are recomputed at the start of each epoch (model in eval mode) and added as negatives, so
    the loss concentrates on the confusions the current model actually makes.
    """
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = encoder.model
    prompted = callable(getattr(encoder, "embed_batch", None))  # SentenceTransformerEncoder
    tokenizer = None if prompted else encoder.tokenizer
    prefix = getattr(encoder, "prefix", "")
    max_length = getattr(encoder, "max_length", 128)
    if class_texts is None:
        if ontology is None:
            raise ValueError("class_texts is required when ontology is None")
        class_texts = build_class_texts(ontology)
    texts = dict(class_texts)
    if hard_negatives is None:
        if ontology is None:
            raise ValueError("hard_negatives is required when ontology is None")
        onto = ontology

        def hard_negatives(gold: str) -> Sequence[str]:
            return sibling_curies(onto, gold)

    pairs = [(q, c) for q, c in train_pairs if c in texts]
    if not pairs:
        raise ValueError("no training pairs with a known class")
    class_order = sorted(texts)
    device = next(model.parameters()).device
    if device.type == "mps":
        # Unified memory: recompute activations in the backward pass. Without it a step of ~100
        # sequences through a 24-layer encoder needs ~35 GB on MPS and swaps instead of training.
        hf = getattr(model[0], "auto_model", None) if prompted else model
        if hasattr(hf, "gradient_checkpointing_enable"):
            hf.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def embed(batch_texts: list[str], documents: bool = False) -> torch.Tensor:
        if prompted:
            return encoder.embed_batch(batch_texts, documents=documents)
        enc = tokenizer(
            [prefix + t for t in batch_texts], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    def save() -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "base_model": encoder.name,
            "max_length": max_length,
            "prefix": prefix,
            "format": "olaya-bi-encoder-v1",
        }
        if prompted:
            payload.update(
                {
                    "format": "olaya-st-encoder-v1",
                    "query_prefix": encoder.query_prefix,
                    "document_prefix": encoder.document_prefix,
                    "text_only": getattr(encoder, "text_only", True),
                    "truncate_dim": getattr(encoder, "truncate_dim", None),
                }
            )
        torch.save(payload, checkpoint)

    if hasattr(encoder, "cache"):
        encoder.cache = None  # weights change: the on-disk cache keyed by the base model is no longer valid
    t0 = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_val, best_epoch = -1.0, 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_run = 0
    stop = False
    for epoch in range(1, max_epochs + 1):
        mined: list[list[str]] = [[] for _ in pairs]
        if mine_hard_negatives > 0:
            model.eval()
            with torch.no_grad():
                c_emb = torch.cat(
                    [embed([texts[c] for c in class_order[i : i + 64]], True) for i in range(0, len(class_order), 64)]
                )
                q_emb = torch.cat(
                    [embed([pairs[i][0] for i in range(j, min(j + 64, len(pairs)))]) for j in range(0, len(pairs), 64)]
                )
            mined = mined_hard_negatives(
                q_emb.cpu().numpy(), c_emb.cpu().numpy(), class_order, [c for _, c in pairs], mine_hard_negatives
            )
        model.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        tot_loss = 0.0
        steps = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            queries = [pairs[i][0] for i in idx]
            golds = [pairs[i][1] for i in idx]
            # Positives: one column per distinct gold class in the batch (same class -> same column).
            columns: list[str] = list(dict.fromkeys(golds))
            hard: list[str] = []
            for g in golds:
                sibs = [s for s in hard_negatives(g) if s in texts and s not in columns and s not in hard]
                if sibs:
                    hard.append(rng.choice(sibs))
            for i in idx:
                for m in mined[i]:
                    if m in texts and m not in columns and m not in hard:
                        hard.append(m)
            columns += hard
            targets = torch.tensor([columns.index(g) for g in golds], dtype=torch.long, device=device)
            q = embed(queries)
            d = embed([texts[c] for c in columns], True)
            logits = (q @ d.T) / temperature
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += float(loss.detach())
            steps += 1
            if time.perf_counter() - t0 > budget_s:
                stop = True
                break
        model.eval()
        epochs_run = epoch
        if hasattr(encoder, "_query_cache"):
            encoder._query_cache.clear()
        val = float(validate(encoder)) if validate is not None else -float(tot_loss / max(1, steps))
        elapsed = time.perf_counter() - t0
        history.append(
            {"epoch": epoch, "train_loss": tot_loss / max(1, steps), "validation": val, "elapsed_s": elapsed}
        )
        say(f"epoch {epoch}: loss {tot_loss / max(1, steps):.3f} validation {val:.3f} ({elapsed:.0f}s)")
        if val > best_val:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop or elapsed > budget_s:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    save()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return FineTuneResult(
        checkpoint=checkpoint,
        sha256=digest,
        base_model=encoder.name,
        epochs_run=epochs_run,
        runtime_s=time.perf_counter() - t0,
        history=history,
        best_validation=best_val,
        best_epoch=best_epoch,
    )


def load_finetuned_encoder(checkpoint: Path, **encoder_kwargs: Any) -> Any:
    """Rebuild a :class:`~olaya.embeddings.TransformerEncoder` from a :func:`finetune_bi_encoder` file.

    The on-disk embedding cache is disabled (it is keyed by the base checkpoint) and ``name`` is
    ``<base>+ft:<sha256 prefix>`` so result files record which weights produced the numbers.
    """
    import torch

    from olaya.embeddings import TransformerEncoder

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fmt = data.get("format") if isinstance(data, dict) else None
    if fmt == "olaya-st-encoder-v1":
        from olaya.embeddings import SentenceTransformerEncoder

        kwargs = {
            "max_length": data.get("max_length", 128),
            "text_only": data.get("text_only", True),
            "truncate_dim": data.get("truncate_dim"),
            **{k: v for k, v in encoder_kwargs.items() if k not in ("use_cache", "prefix")},
        }
        enc: Any = SentenceTransformerEncoder(data["base_model"], **kwargs)
        # The prompts were saved as literal prefixes, not prompt names.
        enc.query_prefix = data.get("query_prefix", "")
        enc.document_prefix = data.get("document_prefix", "")
        enc.prefix = enc.query_prefix
    elif fmt == "olaya-bi-encoder-v1":
        kwargs = {"max_length": data.get("max_length", 128), "prefix": data.get("prefix", ""), **encoder_kwargs}
        kwargs["use_cache"] = False
        enc = TransformerEncoder(data["base_model"], **kwargs)
    else:
        raise ValueError(f"{checkpoint} is not an olaya bi-encoder checkpoint")
    enc.model.load_state_dict(data["model_state_dict"], strict=True)
    enc.model.eval()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    enc.name = f"{data['base_model']}+ft:{digest[:12]}"
    return enc


__all__ = [
    "DEFAULT_SCORER_TEMPERATURE",
    "EmbeddingRouter",
    "EmbeddingScorer",
    "FineTuneResult",
    "HybridRouter",
    "build_class_texts",
    "camel_to_words",
    "class_context_text",
    "finetune_bi_encoder",
    "lexical_rank",
    "load_finetuned_encoder",
    "root_path",
    "sibling_curies",
]
