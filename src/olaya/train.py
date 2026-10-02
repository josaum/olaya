"""Training pipeline for the O-Laya Ontology Decision Head.

Trains the Transformer backbone and marker-pooling decision head across all
5 typed operational decision tasks:
1. `concept`: Multi-class triage grounded in ontology leaf candidates
2. `subsumes`: Calibrated yes/no membership in the principal down-set of any concept
3. `choice`: Operational remediation selection
4. `score`: Continuous/ordinal incident disruption severity
5. `noul`: Operational escalation probability (P(true))
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel

from olaya.category_theory import CategoricalLoss, extract_morphism_matrices
from olaya.data_gen import (
    ALL_REMEDIATION_ACTIONS,
    DEFAULT_SCORE_CRITERIA,
    IncidentInstance,
    generate_synthetic_incidents,
)
from olaya.engine import _flat_subgraph
from olaya.model import OntologyDecisionModel
from olaya.ontology import Concept, Ontology, create_incident_ontology
from olaya.sequence import (
    QTYPE_IDS,
    SequenceBundle,
    build_ontology_sequence,
)


@dataclass
class TrainingSample:
    bundle: SequenceBundle
    target_idx: int
    act_target: int
    qtype_name: str
    ontology: Ontology | None = None


def build_training_samples(
    instances: list[IncidentInstance],
    ontology: Ontology,
    tokenizer_encode: Any,
    special_ids: dict[str, int],
    seed: int = 42,
) -> list[TrainingSample]:
    """Expand incident instances into typed training samples."""
    rng = random.Random(seed)
    samples: list[TrainingSample] = []

    # Identify all candidate leaves under ROOT for concept triage
    leaf_curies = sorted(
        [c for c, concept in ontology.concepts.items() if not concept.children and c != "INCIDENT:ROOT"]
    )
    leaf_subgraph = ontology.extract_subgraph(leaf_curies)
    leaf_concepts = [ontology.concepts[c] for c in leaf_curies]

    for inst in instances:
        state = inst.state
        target_curie = inst.target_curie
        act_target = 1 if inst.escalate else 0

        # 1. Concept triage sample
        if target_curie in leaf_curies:
            bundle_concept = build_ontology_sequence(
                tokenizer_encode=tokenizer_encode,
                special_ids=special_ids,
                state=state,
                instructions="Classify the operational incident into the appropriate ontology category.",
                active_concepts=leaf_concepts,
                ontology=leaf_subgraph,
                qtype_id=QTYPE_IDS["concept"],
                qtype_name="concept",
            )
            if target_curie in bundle_concept.candidate_curies:
                target_idx = bundle_concept.candidate_curies.index(target_curie)
                samples.append(
                    TrainingSample(
                        bundle=bundle_concept,
                        target_idx=target_idx,
                        act_target=act_target,
                        qtype_name="concept",
                        ontology=ontology,
                    )
                )

        # 2. Subsumes sample (Positive or Negative)
        # 50% positive (true ancestor/self), 50% negative (disjoint/unrelated)
        ancestors = ontology.get_ancestors(target_curie)
        is_positive = rng.random() < 0.5
        if is_positive:
            query_curie = rng.choice(sorted(ancestors))
            expected_label = 1  # true
        else:
            non_ancestors = sorted(set(ontology.concepts.keys()) - ancestors - {"INCIDENT:ROOT"})
            query_curie = rng.choice(non_ancestors) if non_ancestors else "INCIDENT:SECURITY"
            expected_label = 0  # false

        concept_obj = ontology.concepts[query_curie]
        descendants = sorted(ontology.get_descendants(query_curie) - {query_curie})
        desc_labels = [ontology.concepts[d].label for d in descendants if d in ontology.concepts]
        desc_str = f" or descendants ({', '.join(desc_labels[:6])})" if desc_labels else ""

        false_desc = f"State is not subsumed by {concept_obj.label} ({query_curie})"
        true_desc = f"State is subsumed by {concept_obj.label} ({query_curie}){desc_str}"
        subsumes_concepts = [
            Concept(curie="false", label="false", description=false_desc),
            Concept(curie="true", label="true", description=true_desc),
        ]
        subsumes_subgraph = _flat_subgraph("subsumes_subgraph", subsumes_concepts)
        instructions = (
            f"Does state satisfy {concept_obj.label} ({query_curie}) or any of its descendants?"
            if desc_labels
            else f"Does state satisfy {concept_obj.label} ({query_curie})?"
        )
        bundle_subsumes = build_ontology_sequence(
            tokenizer_encode=tokenizer_encode,
            special_ids=special_ids,
            state=state,
            instructions=instructions,
            active_concepts=subsumes_concepts,
            ontology=subsumes_subgraph,
            qtype_id=QTYPE_IDS["subsumes"],
            qtype_name="subsumes",
        )
        samples.append(
            TrainingSample(
                bundle=bundle_subsumes,
                target_idx=expected_label,
                act_target=act_target,
                qtype_name="subsumes",
            )
        )

        # 3. Choice sample (Remediation Action)
        correct_action = inst.target_action
        distractors = rng.sample([a for a in ALL_REMEDIATION_ACTIONS if a != correct_action], 3)
        choice_options = distractors + [correct_action]
        rng.shuffle(choice_options)
        choice_idx = choice_options.index(correct_action)

        choice_concepts = [Concept(curie=str(i), label=opt, description="") for i, opt in enumerate(choice_options)]
        choice_subgraph = _flat_subgraph("choice_subgraph", choice_concepts)
        bundle_choice = build_ontology_sequence(
            tokenizer_encode=tokenizer_encode,
            special_ids=special_ids,
            state=state,
            instructions="Select the primary automated remediation action for this operational incident.",
            active_concepts=choice_concepts,
            ontology=choice_subgraph,
            qtype_id=QTYPE_IDS["choice"],
            qtype_name="choice",
        )
        samples.append(
            TrainingSample(
                bundle=bundle_choice,
                target_idx=choice_idx,
                act_target=act_target,
                qtype_name="choice",
            )
        )

        # 4. Score sample (Severity / disruption level 0..3)
        score_concepts = [
            Concept(curie=str(i), label=str(i), description=lvl) for i, lvl in enumerate(DEFAULT_SCORE_CRITERIA)
        ]
        score_subgraph = _flat_subgraph("score_subgraph", score_concepts)
        bundle_score = build_ontology_sequence(
            tokenizer_encode=tokenizer_encode,
            special_ids=special_ids,
            state=state,
            instructions="Rate the operational disruption level on the ordered severity scale.",
            active_concepts=score_concepts,
            ontology=score_subgraph,
            qtype_id=QTYPE_IDS["score"],
            qtype_name="score",
        )
        samples.append(
            TrainingSample(
                bundle=bundle_score,
                target_idx=inst.severity,
                act_target=act_target,
                qtype_name="score",
            )
        )

        # 5. Noul sample (Escalation P(true))
        noul_concepts = [
            Concept(curie="false", label="false", description="Statement does not hold"),
            Concept(curie="true", label="true", description="Statement holds"),
        ]
        noul_subgraph = _flat_subgraph("noul_subgraph", noul_concepts)
        bundle_noul = build_ontology_sequence(
            tokenizer_encode=tokenizer_encode,
            special_ids=special_ids,
            state=state,
            instructions="Should on-call engineering be paged immediately?",
            active_concepts=noul_concepts,
            ontology=noul_subgraph,
            qtype_id=QTYPE_IDS["noul"],
            qtype_name="noul",
        )
        noul_label = 1 if inst.escalate else 0
        samples.append(
            TrainingSample(
                bundle=bundle_noul,
                target_idx=noul_label,
                act_target=act_target,
                qtype_name="noul",
            )
        )

    rng.shuffle(samples)
    return samples


class DecisionDataset(Dataset):
    def __init__(self, samples: list[TrainingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self.samples[idx]


def collate_decision_batch(batch: list[TrainingSample], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad bundles into batched model inputs, including category-theoretic morphism matrices."""
    n = len(batch)
    max_seq_len = max(len(s.bundle.input_ids) for s in batch)
    max_markers = max(len(s.bundle.marker_positions) for s in batch)

    input_ids = torch.full((n, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_seq_len), dtype=torch.long)
    marker_pos = torch.zeros((n, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((n, max_markers), dtype=torch.bool)
    qtypes = torch.zeros((n,), dtype=torch.long)
    targets = torch.zeros((n,), dtype=torch.long)
    act_targets = torch.zeros((n,), dtype=torch.long)

    # Category theory morphism matrices
    subsumption_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)
    parent_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)
    child_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)
    disjoint_adj = torch.zeros((n, max_markers, max_markers), dtype=torch.float32)

    for i, s in enumerate(batch):
        b = s.bundle
        seq_len = len(b.input_ids)
        input_ids[i, :seq_len] = torch.tensor(b.input_ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        num_m = len(b.marker_positions)
        if num_m > 0:
            marker_pos[i, :num_m] = torch.tensor(b.marker_positions, dtype=torch.long)
            marker_mask[i, :num_m] = True

            # Extract category morphism matrices if ontology is available
            if s.ontology is not None and len(b.candidate_curies) > 0:
                sub_m, p_m, c_m, disj_m = extract_morphism_matrices(
                    b.candidate_curies, s.ontology, max_k=num_m
                )
                subsumption_adj[i, :num_m, :num_m] = torch.from_numpy(sub_m)
                parent_adj[i, :num_m, :num_m] = torch.from_numpy(p_m)
                child_adj[i, :num_m, :num_m] = torch.from_numpy(c_m)
                disjoint_adj[i, :num_m, :num_m] = torch.from_numpy(disj_m)

        qtypes[i] = b.qtype
        targets[i] = s.target_idx
        act_targets[i] = s.act_target

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtypes,
        "targets": targets,
        "act_targets": act_targets,
        "subsumption_adj": subsumption_adj,
        "parent_adj": parent_adj,
        "child_adj": child_adj,
        "disjoint_adj": disjoint_adj,
    }


def create_trainable_model(
    vocab_size: int = 50368,
    hidden_size: int = 256,
    backbone: str | None = None,
) -> OntologyDecisionModel:
    """Create decision model with either pretrained backbone (e.g. ModernBERT) or lightweight custom BERT."""
    if backbone and backbone != "custom_bert":
        return OntologyDecisionModel(backbone_name_or_config=backbone)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
    )
    custom_encoder = BertModel(cfg)
    return OntologyDecisionModel(hidden_size=hidden_size, custom_encoder=custom_encoder)


def train_olaya(
    ontology: Ontology | None = None,
    tokenizer: Any | None = None,
    num_samples: int = 1200,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: str | None = None,
    output_path: str | Path = "models/olaya_decision_model.pt",
    seed: int = 42,
) -> tuple[OntologyDecisionModel, dict[str, float]]:
    """Train the O-Laya decision model and save checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    onto = ontology or create_incident_ontology()

    # Determine device
    if device is None:
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # Tokenizer setup
    if tokenizer is None:
        from tests.conftest import load_local_tokenizer

        tok = load_local_tokenizer()
    else:
        tok = tokenizer

    special_ids = {
        "cls": getattr(tok, "cls_token_id", 50281) or 50281,
        "sep": getattr(tok, "sep_token_id", 50282) or 50282,
        "mask": getattr(tok, "mask_token_id", 50284) or 50284,
        "pad": getattr(tok, "pad_token_id", 50283) or 0,
    }

    # Generate synthetic incidents and build samples
    print(f"Generating {num_samples} synthetic operational incidents...")
    instances = generate_synthetic_incidents(num_samples=num_samples, seed=seed)
    split_idx = int(0.85 * len(instances))
    train_insts, val_insts = instances[:split_idx], instances[split_idx:]

    print("Building typed decision sequence bundles...")
    train_samples = build_training_samples(
        train_insts, onto, lambda t: tok.encode(t, add_special_tokens=False), special_ids, seed=seed
    )
    val_samples = build_training_samples(
        val_insts, onto, lambda t: tok.encode(t, add_special_tokens=False), special_ids, seed=seed + 1
    )

    print(f"Dataset assembled: {len(train_samples)} training samples, {len(val_samples)} validation samples")

    train_loader = DataLoader(
        DecisionDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=special_ids["pad"]),
    )
    val_loader = DataLoader(
        DecisionDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_decision_batch(b, pad_id=special_ids["pad"]),
    )

    vocab_size = max(len(tok), getattr(tok, "vocab_size", 30522)) + 256
    model = create_trainable_model(vocab_size=vocab_size, hidden_size=256)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    criterion_dec = nn.CrossEntropyLoss()
    criterion_act = nn.NLLLoss()
    criterion_cat = CategoricalLoss(weight_mono=0.5, weight_disj=0.3, weight_cone=0.1)

    best_val_acc = 0.0
    metrics: dict[str, float] = {}

    print(f"Starting training on {dev} for {epochs} epochs with Category-Theoretic Loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cat_loss = 0.0
        correct_dec = 0
        total_items = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            marker_pos = batch["marker_pos"].to(dev)
            marker_mask = batch["marker_mask"].to(dev)
            qtypes = batch["qtype"].to(dev)
            targets = batch["targets"].to(dev)
            act_targets = batch["act_targets"].to(dev)
            subsumption_adj = batch["subsumption_adj"].to(dev)
            parent_adj = batch["parent_adj"].to(dev)
            child_adj = batch["child_adj"].to(dev)
            disjoint_adj = batch["disjoint_adj"].to(dev)

            optimizer.zero_grad()
            logits, act_probs, marker_states = model(
                input_ids,
                attention_mask,
                marker_pos,
                marker_mask,
                qtypes,
                parent_adj=parent_adj,
                child_adj=child_adj,
                disjoint_adj=disjoint_adj,
                return_marker_states=True,
            )

            loss_dec = criterion_dec(logits, targets)
            loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
            loss_cat, _ = criterion_cat(
                logits=logits,
                marker_states=marker_states,
                subsumption_mask=subsumption_adj,
                disjoint_mask=disjoint_adj,
                marker_mask=marker_mask,
            )
            loss = loss_dec + 0.2 * loss_act + loss_cat

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item()) * len(targets)
            total_cat_loss += float(loss_cat.item()) * len(targets)
            preds = torch.argmax(logits, dim=-1)
            correct_dec += int((preds == targets).sum().item())
            total_items += len(targets)

        train_loss = total_loss / total_items
        train_cat_loss = total_cat_loss / total_items
        train_acc = correct_dec / total_items

        # Validation
        model.eval()
        val_loss = 0.0
        val_cat_loss = 0.0
        val_mono_viols: list[float] = []
        val_disj_colls: list[float] = []
        val_correct = 0
        val_items = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                marker_pos = batch["marker_pos"].to(dev)
                marker_mask = batch["marker_mask"].to(dev)
                qtypes = batch["qtype"].to(dev)
                targets = batch["targets"].to(dev)
                act_targets = batch["act_targets"].to(dev)
                subsumption_adj = batch["subsumption_adj"].to(dev)
                parent_adj = batch["parent_adj"].to(dev)
                child_adj = batch["child_adj"].to(dev)
                disjoint_adj = batch["disjoint_adj"].to(dev)

                logits, act_probs, marker_states = model(
                    input_ids,
                    attention_mask,
                    marker_pos,
                    marker_mask,
                    qtypes,
                    parent_adj=parent_adj,
                    child_adj=child_adj,
                    disjoint_adj=disjoint_adj,
                    return_marker_states=True,
                )
                loss_dec = criterion_dec(logits, targets)
                loss_act = criterion_act(torch.log(act_probs + 1e-8), act_targets)
                loss_cat, cat_metrics = criterion_cat(
                    logits=logits,
                    marker_states=marker_states,
                    subsumption_mask=subsumption_adj,
                    disjoint_mask=disjoint_adj,
                    marker_mask=marker_mask,
                )
                loss = loss_dec + 0.2 * loss_act + loss_cat

                val_loss += float(loss.item()) * len(targets)
                val_cat_loss += float(loss_cat.item()) * len(targets)
                val_mono_viols.append(cat_metrics["mono_violation_rate"])
                val_disj_colls.append(cat_metrics["disj_collision_rate"])
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == targets).sum().item())
                val_items += len(targets)

        val_loss /= val_items
        val_cat_loss /= val_items
        val_acc = val_correct / val_items
        avg_mono_viol = float(np.mean(val_mono_viols)) if val_mono_viols else 0.0
        avg_disj_coll = float(np.mean(val_disj_colls)) if val_disj_colls else 0.0

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} (Cat: {train_cat_loss:.4f}) Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | Mono Viol: {avg_mono_viol:.1%} Disj Coll: {avg_disj_coll:.1%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "intermediate_size": 512,
                    "max_position_embeddings": 512,
                },
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_cat_loss": val_cat_loss,
                "val_mono_violation_rate": avg_mono_viol,
                "val_disj_collision_rate": avg_disj_coll,
            }
            torch.save(checkpoint, str(out_file))
            print(f"  -> Saved best checkpoint (acc: {val_acc:.2%}) to {out_file}")

    metrics = {
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss,
        "final_cat_loss": val_cat_loss,
        "final_mono_violation_rate": avg_mono_viol,
        "final_disj_collision_rate": avg_disj_coll,
    }
    return model, metrics
