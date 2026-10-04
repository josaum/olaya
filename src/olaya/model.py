"""ModernBERT-based Decision Model with marker pooling and calibrated ontology head.

Matches Laya / Jev non-autoregressive architecture:
- Single forward pass
- Extracts hidden representations at [MASK] marker token indices
- Outputs raw logits over candidate option slots and confidence estimates
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from olaya.category_theory import CategoricalMorphismLayer


class OntologyDecisionHead(nn.Module):
    """Pools hidden states at marker positions and computes option logits."""

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_question_types: int = 5):
        super().__init__()
        # Question-type conditioning (concept / score / subsumes / choice / noul), added to the pooled
        # marker and [CLS] states. Zero-initialised so it starts as an identity.
        self.qtype_embedding = nn.Embedding(num_question_types, hidden_size)
        nn.init.zeros_(self.qtype_embedding.weight)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Categorical morphism layer (relational message passing over morphisms)
        self.morphism_layer = CategoricalMorphismLayer(hidden_size, dropout=dropout)

        self.classifier = nn.Linear(hidden_size, 1)

        # Act / escalate auxiliary head (schema compatible with Laya/Jev). It is trained against the
        # ``act_targets`` built by ``olaya.train`` / ``scripts/train_real_ontologies.py`` and exposed by
        # the engine as ``SystemOneResult.act_probabilities``; the ONNX export emits it as ``act_probs``.
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(
        self,
        sequence_hidden_states: torch.Tensor,  # [B, L, H]
        marker_pos: torch.Tensor,  # [B, K]
        marker_mask: torch.Tensor,  # [B, K] (bool)
        cls_hidden_state: torch.Tensor,  # [B, H]
        qtype: torch.Tensor,  # [B]
        parent_adj: torch.Tensor | None = None,  # [B, K, K]
        child_adj: torch.Tensor | None = None,  # [B, K, K]
        disjoint_adj: torch.Tensor | None = None,  # [B, K, K]
        return_marker_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K = marker_pos.shape
        H = sequence_hidden_states.shape[-1]

        # Gather hidden states at marker positions: [B, K, H]
        # Expand marker_pos to [B, K, H]
        idx = marker_pos.unsqueeze(-1).expand(B, K, H)
        marker_states = torch.gather(sequence_hidden_states, 1, idx)
        qtype_emb = self.qtype_embedding(qtype)  # [B, H]
        marker_states = marker_states + qtype_emb.unsqueeze(1)

        # Transform marker states
        h = self.norm(self.act(self.dense(marker_states)))
        h = self.dropout(h)

        # Apply category-theoretic relational morphism layer
        h_morph = self.morphism_layer(
            h,
            parent_adj=parent_adj,
            child_adj=child_adj,
            disjoint_adj=disjoint_adj,
        )

        # Logits for each candidate slot: [B, K]
        logits = self.classifier(h_morph).squeeze(-1)

        # Mask out padded/unused option slots with -1e4
        logits = torch.where(marker_mask, logits, torch.tensor(-1e4, device=logits.device, dtype=logits.dtype))

        # Act / escalate probability: [B, 2]
        act_logits = self.act_head(cls_hidden_state + qtype_emb)
        act_probs = torch.softmax(act_logits, dim=-1)

        if return_marker_states:
            return logits, act_probs, h_morph
        return logits, act_probs


class OntologyDecisionModel(nn.Module):
    """Complete Decision Model combining a bidirectional transformer with an ontology decision head."""

    def __init__(
        self,
        backbone_name_or_config: str | AutoConfig = "answerdotai/ModernBERT-base",
        hidden_size: int = 768,
        custom_encoder: nn.Module | None = None,
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
            self.hidden_size = hidden_size
        else:
            try:
                self.encoder = AutoModel.from_pretrained(backbone_name_or_config)
                self.hidden_size = self.encoder.config.hidden_size
            except Exception:
                # Fallback to local config if offline
                cfg = (
                    AutoConfig.from_pretrained(backbone_name_or_config)
                    if isinstance(backbone_name_or_config, str)
                    else backbone_name_or_config
                )
                self.encoder = AutoModel.from_config(cfg)
                self.hidden_size = cfg.hidden_size

        self.head = OntologyDecisionHead(self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        marker_pos: torch.Tensor,  # [B, K]
        marker_mask: torch.Tensor,  # [B, K]
        qtype: torch.Tensor,  # [B]
        parent_adj: torch.Tensor | None = None,  # [B, K, K]
        child_adj: torch.Tensor | None = None,  # [B, K, K]
        disjoint_adj: torch.Tensor | None = None,  # [B, K, K]
        return_marker_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode full bidirectional sequence (cross-attention across tokens)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_hidden = outputs.last_hidden_state  # [B, L, H]
        cls_hidden = sequence_hidden[:, 0, :]  # [B, H]

        return self.head(
            sequence_hidden,
            marker_pos,
            marker_mask,
            cls_hidden,
            qtype,
            parent_adj=parent_adj,
            child_adj=child_adj,
            disjoint_adj=disjoint_adj,
            return_marker_states=return_marker_states,
        )
