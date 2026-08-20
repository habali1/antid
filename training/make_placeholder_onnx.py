#!/usr/bin/env python3
"""make_placeholder_onnx.py — emit a RANDOM-weight backbone.onnx for plumbing.

NOT training. Builds an untrained EfficientNet-B4 AntIDModel and exports just the
backbone (image -> 1792-dim embedding) so the API has a runnable ONNX graph whose
input/output shapes match the real artifact. Accuracy is meaningless by design.

Run (from anywhere; Python adds this file's dir to sys.path):
    .venv/Scripts/python.exe training/make_placeholder_onnx.py
"""
from __future__ import annotations

from pathlib import Path

from export import export_backbone   # training/export.py
from model import AntIDModel         # training/model.py

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts" / "backbone.onnx"


def main() -> None:
    model = AntIDModel(
        num_classes=3,
        backbone="tf_efficientnet_b4",
        pretrained=False,        # random weights — no ImageNet download, no training
        dropout=0.3,
        embedding_dim=1792,
    )
    model.eval()
    path = export_backbone(model, OUT, image_size=380, device="cpu")
    size = path.stat().st_size
    print(f"[placeholder] wrote {path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
