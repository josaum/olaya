r"""Category-Theoretic Neural Architecture & Axiomatic Loss Functions for Leio.

Formal Category Theory Mapping:
1. Category C (Ontology Poset):
   - Objects: Ontological Concepts c in Ob(C).
   - Morphisms: Hom(A, B) has at most one arrow A -> B iff A subClassOf B (A <= B).
   - Composition: Subsumption transitivity (A <= B and B <= C => A <= C).
   - Initial Object (Bottom _|_): Mutually disjoint meet A \sqcap B = _|_.
   - Terminal Object (Top T): Root concept / owl:Thing (A <= T for all A).

2. Functorality to Probability Poset (Prob, >=):
   - A decision state x induces a probability assignment P(- | x): Ob(C) -> [0, 1].
   - Functor Axiom: Monotone preservation of morphisms:
     A <= B ==> P(A | x) <= P(B | x).
   - In category terms, this is an order-preserving functor F_x: C -> (Prob, >=).

3. Categorical Disjointness & Orthogonality:
   - For disjoint pairs A \sqcap B = _|_, coproduct exclusivity dictates:
     P(A \sqcap B) = 0, and in soft categorical relaxation P(A) * P(B) -> 0.

4. Order-Cone Latent Embeddings:
   - Concepts map into an order cone in R^H:
     A <= B ==> h_A <= h_B  (element-wise in positive cone),
     penalized differentiably by ||ReLU(h_A - h_B)||^2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from olaya.ontology import Ontology


def extract_morphism_matrices(
    candidate_curies: list[str],
    ontology: Ontology | None,
    max_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract subsumption, parent, child, and disjointness adjacency matrices for candidate CURIEs.

    Returns:
        (subsumption_matrix, parent_matrix, child_matrix, disjointness_matrix)
        Each is an [K, K] float32 numpy array, where K = max_k or len(candidate_curies).
    """
    k = max_k if max_k is not None else len(candidate_curies)
    n = min(len(candidate_curies), k)

    subsumption_mat = np.zeros((k, k), dtype=np.float32)
    parent_mat = np.zeros((k, k), dtype=np.float32)
    child_mat = np.zeros((k, k), dtype=np.float32)
    disjoint_mat = np.zeros((k, k), dtype=np.float32)

    if ontology is None or n == 0:
        return subsumption_mat, parent_mat, child_mat, disjoint_mat

    # Pre-extract ancestors & direct parents for each valid candidate
    valid_curies = [c for c in candidate_curies[:n] if c in ontology.concepts]
    curie_idx = {c: i for i, c in enumerate(candidate_curies[:n])}

    for c in valid_curies:
        i = curie_idx[c]
        concept = ontology.concepts[c]
        ancestors = ontology.get_ancestors(c)

        # 1. Subsumptions: c_i <= c_j (i != j)
        for anc in ancestors:
            if anc != c and anc in curie_idx:
                j = curie_idx[anc]
                subsumption_mat[i, j] = 1.0

        # 2. Direct parent morphisms: c_j in parents(c_i)
        for parent in concept.parents:
            if parent in curie_idx:
                j = curie_idx[parent]
                parent_mat[i, j] = 1.0
                child_mat[j, i] = 1.0

        # 3. Disjointness: c_i \sqcap c_j = _|_
        for other in valid_curies:
            j = curie_idx[other]
            if i != j and ontology.are_disjoint(c, other):
                disjoint_mat[i, j] = 1.0

    return subsumption_mat, parent_mat, child_mat, disjoint_mat


class FunctorMonotonicityLoss(nn.Module):
    """Enforces functor monotonicity F: C -> (Prob, >=).

    For every subsumption morphism A <= B, penalizes p(A) > p(B) differentiably:
    L_mono = (1 / (|E_<=| + eps)) * sum_{A <= B} ReLU(p(A) - p(B))^2
    """

    def __init__(self, eps: float = 1e-7, tol: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.tol = tol

    def forward(
        self,
        logits: torch.Tensor,  # [B, K]
        subsumption_mask: torch.Tensor,  # [B, K, K] (1.0 where child=i, parent=j)
        marker_mask: torch.Tensor | None = None,  # [B, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (loss, violation_rate)"""
        # Exclude masked options from softmax
        if marker_mask is not None:
            logits_clean = torch.where(
                marker_mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
            )
        else:
            logits_clean = logits

        probs = F.softmax(logits_clean, dim=-1)  # [B, K]

        # Pairwise probability difference: diff[b, i, j] = p_i - p_j
        # For subsumption_mask[b, i, j] == 1, i is child and j is parent.
        # Monotonicity requires p_j >= p_i, so violation occurs when p_i > p_j (i.e. diff > 0).
        p_child = probs.unsqueeze(-1)  # [B, K, 1]
        p_parent = probs.unsqueeze(-2)  # [B, 1, K]
        diff = p_child - p_parent  # [B, K, K]

        # ReLU penalty on violations
        violations = F.relu(diff)
        masked_violations = violations * subsumption_mask

        # Sum of squared violations
        loss = (masked_violations**2).sum() / (subsumption_mask.sum() + self.eps)

        # Compute empirical violation rate
        with torch.no_grad():
            violating_pairs = ((diff > self.tol) * (subsumption_mask > 0.5)).sum()
            total_pairs = (subsumption_mask > 0.5).sum()
            violation_rate = violating_pairs / (total_pairs + self.eps)

        return loss, violation_rate


class CategoricalDisjointnessLoss(nn.Module):
    r"""Enforces categorical exclusivity for disjoint objects A \sqcap B = _|_.

    Penalizes co-activation of mutually exclusive concepts:
    L_disj = (1 / (|E_disj| + eps)) * sum_{A \sqcap B = _|_} p(A) * p(B)
    """

    def __init__(self, eps: float = 1e-7, collision_thresh: float = 0.3):
        super().__init__()
        self.eps = eps
        self.collision_thresh = collision_thresh

    def forward(
        self,
        logits: torch.Tensor,  # [B, K]
        disjoint_mask: torch.Tensor,  # [B, K, K] (symmetric, 1.0 where disjoint)
        marker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (loss, collision_rate)"""
        if marker_mask is not None:
            logits_clean = torch.where(
                marker_mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
            )
        else:
            logits_clean = logits

        probs = F.softmax(logits_clean, dim=-1)  # [B, K]

        # Product of probabilities: prod[b, i, j] = p_i * p_j
        p_i = probs.unsqueeze(-1)  # [B, K, 1]
        p_j = probs.unsqueeze(-2)  # [B, 1, K]
        pairwise_prod = p_i * p_j  # [B, K, K]

        masked_prod = pairwise_prod * disjoint_mask
        # Note: disjoint_mask is symmetric with zeros on diagonal, divide by 2 for unique pairs
        num_pairs = (disjoint_mask.sum() / 2.0) + self.eps
        loss = (masked_prod.sum() / 2.0) / num_pairs

        with torch.no_grad():
            min_pair_prob = torch.minimum(p_i, p_j)
            collisions = ((min_pair_prob > self.collision_thresh) * (disjoint_mask > 0.5)).sum() / 2.0
            collision_rate = collisions / num_pairs

        return loss, collision_rate


class OrderConeEmbeddingLoss(nn.Module):
    """Enforces geometric poset order-cone embeddings in latent marker space.

    For A <= B, concept A must lie within the order cone of concept B:
    L_cone = (1 / (|E_<=| * H + eps)) * sum_{A <= B} ||ReLU(h_A - h_B)||^2
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        marker_states: torch.Tensor,  # [B, K, H]
        subsumption_mask: torch.Tensor,  # [B, K, K] (1.0 where child=i, parent=j)
    ) -> torch.Tensor:
        # h_child: [B, K, 1, H], h_parent: [B, 1, K, H]
        h_child = marker_states.unsqueeze(2)  # [B, K, 1, H]
        h_parent = marker_states.unsqueeze(1)  # [B, 1, K, H]

        diff = h_child - h_parent  # [B, K, K, H]
        cone_violations = F.relu(diff) ** 2  # [B, K, K, H]

        # Mask across dimensions
        mask_expanded = subsumption_mask.unsqueeze(-1)  # [B, K, K, 1]
        masked_violations = (cone_violations * mask_expanded).sum()

        num_constraints = (subsumption_mask.sum() * marker_states.shape[-1]) + self.eps
        return masked_violations / num_constraints


class CategoricalLoss(nn.Module):
    """Composite category-theoretic loss combining monotonicity, disjointness, and order cones."""

    def __init__(
        self,
        weight_mono: float = 0.5,
        weight_disj: float = 0.3,
        weight_cone: float = 0.1,
    ):
        super().__init__()
        self.weight_mono = weight_mono
        self.weight_disj = weight_disj
        self.weight_cone = weight_cone

        self.mono_loss_fn = FunctorMonotonicityLoss()
        self.disj_loss_fn = CategoricalDisjointnessLoss()
        self.cone_loss_fn = OrderConeEmbeddingLoss()

    def forward(
        self,
        logits: torch.Tensor,  # [B, K]
        marker_states: torch.Tensor,  # [B, K, H]
        subsumption_mask: torch.Tensor,  # [B, K, K]
        disjoint_mask: torch.Tensor,  # [B, K, K]
        marker_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_mono, mono_viol = self.mono_loss_fn(logits, subsumption_mask, marker_mask)
        loss_disj, disj_coll = self.disj_loss_fn(logits, disjoint_mask, marker_mask)
        loss_cone = self.cone_loss_fn(marker_states, subsumption_mask)

        total_loss = self.weight_mono * loss_mono + self.weight_disj * loss_disj + self.weight_cone * loss_cone

        metrics = {
            "loss_cat_total": float(total_loss.item()),
            "loss_mono": float(loss_mono.item()),
            "loss_disj": float(loss_disj.item()),
            "loss_cone": float(loss_cone.item()),
            "mono_violation_rate": float(mono_viol.item()),
            "disj_collision_rate": float(disj_coll.item()),
        }

        return total_loss, metrics


class CategoricalMorphismLayer(nn.Module):
    """Relational morphism message-passing layer encoding ontology categories and posets.

    Propagates representations along morphisms (subsumption edges, parent/child morphisms,
    and disjointness relations) directly in hidden state space before logit projection.

    Zero-initialization ensures this layer acts as a pure identity at initialization
    and when no morphism matrices are provided, maintaining 100% backward compatibility
    with vanilla inference and ONNX runtimes.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Morphism projections for relation types
        self.w_parent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_child = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_disjoint = nn.Linear(hidden_size, hidden_size, bias=False)

        # Morphism fusion
        self.fusion = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection initialized to zeros (identity start)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        marker_states: torch.Tensor,  # [B, K, H]
        parent_adj: torch.Tensor | None = None,  # [B, K, K]
        child_adj: torch.Tensor | None = None,  # [B, K, K]
        disjoint_adj: torch.Tensor | None = None,  # [B, K, K]
    ) -> torch.Tensor:
        """Propagate morphism messages across candidates.

        If any adjacency is None or all zeros, returns marker_states unchanged (residual + norm).
        """
        if parent_adj is None and child_adj is None and disjoint_adj is None:
            return marker_states

        B, K, H = marker_states.shape

        # Normalize adjacencies by out-degree
        def norm_adj(adj: torch.Tensor | None) -> torch.Tensor:
            if adj is None:
                return torch.zeros((B, K, K), device=marker_states.device, dtype=marker_states.dtype)
            degrees = adj.sum(dim=-1, keepdim=True)
            return adj / (degrees + 1e-7)

        p_norm = norm_adj(parent_adj)  # [B, K, K]
        c_norm = norm_adj(child_adj)  # [B, K, K]
        d_norm = norm_adj(disjoint_adj)  # [B, K, K]

        # Check if graph is completely empty
        if (p_norm.sum() + c_norm.sum() + d_norm.sum()) == 0.0:
            return marker_states

        # Message passing: [B, K, K] @ [B, K, H] -> [B, K, H]
        m_parent = torch.bmm(p_norm, self.w_parent(marker_states))
        m_child = torch.bmm(c_norm, self.w_child(marker_states))
        m_disjoint = torch.bmm(d_norm, self.w_disjoint(marker_states))

        # Concatenate self state with relational morphism messages
        combined = torch.cat([marker_states, m_parent, m_child, m_disjoint], dim=-1)  # [B, K, 4H]
        m_fused = self.act(self.fusion(combined))
        m_fused = self.dropout(m_fused)

        # Residual connection with zero-init projection
        out = self.norm(marker_states + self.out_proj(m_fused))
        return out
