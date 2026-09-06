#!/usr/bin/env python3
"""export.py — export the EfficientNet-B4 backbone to ONNX.

Primary use is the importable `export_backbone()` called at the end of
train.py. A standalone CLI is also provided to (re)export from a saved
checkpoint:

    python export.py --checkpoint artifacts/model.pth --taxonomy artifacts/taxonomy.json

ONNX contract (consumed by api/inference.py):
  input  "input"  float32  (batch, 3, H, W)   — batch is dynamic, H=W=image_size
  output "embedding" float32 (batch, 1792)
  opset 17
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from model import AntIDModel, BackboneForExport

HERE = Path(__file__).resolve().parent


def export_backbone(model: AntIDModel, out_path: Path, image_size: int,
                    device: torch.device | str = "cpu") -> Path:
    """Export `model`'s backbone to ONNX at out_path. Returns out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # BackboneForExport shares model.backbone; .to('cpu') moves it in place, so
    # capture the original device to restore afterward — otherwise the caller's
    # model is left on CPU and any later GPU use (e.g. evaluation) crashes with a
    # device mismatch.
    try:
        orig_device = next(model.backbone.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")

    wrapper = BackboneForExport(model).to("cpu").eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    # Export to a temp path and os.replace into place at the end, so a
    # crash or failed onnx-checker validation never leaves a truncated or
    # partially-written file at out_path.
    tmp_path = out_path.parent / f"{out_path.stem}.tmp{os.getpid()}{out_path.suffix}"
    try:
        try:
            # torch>=2.x routes through the dynamo exporter by default; we use
            # the legacy (dynamic_axes) path explicitly for a stable opset-17 graph.
            torch.onnx.export(wrapper, dummy, str(tmp_path), dynamo=False, **export_kwargs)
        except TypeError:
            # Older torch without a `dynamo` kwarg.
            torch.onnx.export(wrapper, dummy, str(tmp_path), **export_kwargs)

        # Sanity-check the exported graph loads and runs.
        try:
            import onnx
            onnx.checker.check_model(onnx.load(str(tmp_path)))
        except ImportError:
            pass

        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        model.backbone.to(orig_device)  # restore the caller's model to its device
    return out_path


def _load_model_from_checkpoint(ckpt: Path, taxonomy: Path) -> tuple[AntIDModel, int]:
    tax = json.loads(Path(taxonomy).read_text())
    num_classes = len(tax)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)  # our own trusted checkpoints
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    model_cfg = cfg.get("model", {})
    model = AntIDModel(
        num_classes=num_classes,
        backbone=model_cfg.get("backbone", "tf_efficientnet_b4"),
        pretrained=False,
        dropout=model_cfg.get("dropout", cfg.get("dropout", 0.3)),
        embedding_dim=model_cfg.get("embedding_dim", 1792),
    )
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()
    return model, cfg.get("image_size", 380)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=HERE / "artifacts" / "model.pth")
    ap.add_argument("--taxonomy", type=Path, default=HERE / "artifacts" / "taxonomy.json")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "backbone.onnx")
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    model, ckpt_size = _load_model_from_checkpoint(args.checkpoint, args.taxonomy)
    size = args.image_size or ckpt_size
    out = export_backbone(model, args.out, size)
    print(f"[export] wrote {out} (input {size}x{size}, opset 17, dynamic batch)")


if __name__ == "__main__":
    main()
