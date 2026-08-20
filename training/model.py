"""AntID model: EfficientNet-B4 backbone with an embedding + linear head.

The 1792-dim vector *before* the final linear layer is the embedding used for
prototype/cosine-similarity inference. Training optimizes the linear head with
cross-entropy so the embedding space separates species; at serve time the head
is discarded and only the backbone (→ embedding) is exported to ONNX.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover - surfaced at runtime on GPU box
    raise SystemExit(
        "timm is required for training. Install with: pip install -r requirements.txt"
    ) from e


class AntIDModel(nn.Module):
    """EfficientNet-B4 → GlobalAvgPool → Dropout → Linear(embedding_dim, num_classes).

    forward() returns logits (for training); embed() returns the L2-normalizable
    pre-head embedding (for prototype computation and ONNX export).
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.3,
        embedding_dim: int = 1792,
    ) -> None:
        super().__init__()
        # num_classes=0, global_pool="avg" → backbone outputs a pooled feature
        # vector of width `num_features` (1792 for B4). No classifier inside timm.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        if feat_dim != embedding_dim:
            # B4 is 1792; guard against a backbone swap silently breaking shapes.
            raise ValueError(
                f"{backbone} has {feat_dim}-dim features but config expects "
                f"{embedding_dim}. Update config.model.embedding_dim."
            )
        self.embedding_dim = feat_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled 1792-dim embedding (pre-dropout, pre-head)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        return self.classifier(self.dropout(emb))


class BackboneForExport(nn.Module):
    """Thin wrapper exported to ONNX: image tensor → embedding.

    Exporting this (not the full model) keeps the serving interface to a single
    artifact whose output is exactly the vector the API runs cosine similarity on.
    """

    def __init__(self, model: AntIDModel) -> None:
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
