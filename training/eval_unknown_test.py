#!/usr/bin/env python3
"""eval_unknown_test.py — the ONE independent evaluation of the frozen
abstention rule (see data/calibration_v1/calibration_v1.json:
frozen_candidate_abstention_threshold): abstain when max_sim < 0.60.

This does not tune, sweep, or search anything. The threshold is loaded from
calibration_v1.json's frozen_candidate_abstention_threshold.machine_readable_rule
(value + operator), not re-derived from unknown_test_v1's own distributions,
and validated against the one comparison this script implements before use.
If the result here is bad, that is a finding to report; it is not a cue to
adjust the value and re-run.

Note on provenance: the original 0.60 selection and the original
unknown_test_v1 evaluation (the run whose numbers are recorded in
unknown_test_v1_eval.json and calibration_v1.json's phase_c_validation) both
predate machine_readable_rule and used 0.60 as a hardcoded constant in this
file. This script was updated afterward to load and validate the value from
calibration_v1.json for future reproductions; the rule itself did not change
(still 0.60, still max_sim < value) and unknown_test_v1_eval.json was not
regenerated -- only its hashes block was amended to record calibration_v1.json's
sha256 and this provenance note.

Raw cosine only, no geo re-ranking, same as eval_calibration.py. Same
fail-closed verification as eval_benchmark.py / eval_calibration.py: refuses
to run unless every unknown_test_v1.csv row resolves to exactly one
hash-matched local image.

Reports, at the frozen threshold only (no curve, no sweep):
  - known-photo rejection rate (= FRR over ALL known_holdout images, correct
    and incorrect predictions alike)
  - correct-vs-incorrect rejection breakdown (does the rule behave like a
    confidence gate here too)
  - accuracy among accepted known-species predictions
  - false acceptance rate for each OOD category (out_of_scope_ant,
    non_ant_insect, unrelated) separately

Usage:
    python eval_unknown_test.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import AntDataset, Sample
from model import AntIDModel

HERE = Path(__file__).resolve().parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SUPPORTED_OPERATOR = "max_sim < value"  # the only comparison this script implements


def load_frozen_threshold(calibration_json_path: Path) -> float:
    """Load and validate the frozen candidate threshold from calibration_v1.json.

    Fails loudly rather than silently reinterpreting an operator this script
    doesn't implement -- if a future calibration_v1.json records a different
    comparison (e.g. a combined max_sim+gap rule), this script must not just
    apply `value` as if it were still a plain max_sim cutoff.
    """
    d = json.loads(calibration_json_path.read_text())
    rule = d["frozen_candidate_abstention_threshold"]["machine_readable_rule"]
    if rule["operator"] != SUPPORTED_OPERATOR:
        raise SystemExit(
            f"calibration_v1.json records operator {rule['operator']!r}, but this "
            f"script only implements {SUPPORTED_OPERATOR!r}. Update eval_unknown_test.py "
            f"before evaluating a rule of a different shape."
        )
    value = float(rule["value"])
    if not (0.0 < value < 1.0):
        raise SystemExit(f"frozen threshold value {value} is outside the valid "
                         f"cosine-similarity range (0, 1)")
    return value


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_unknown_test(csv_path: Path, image_dir: Path, slug_to_idx: dict[str, int]):
    """Strict, fail-closed resolution -- identical policy to eval_calibration.py."""
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    samples: list[Sample] = []
    meta: list[dict] = []
    problems: list[str] = []

    for r in rows:
        slug, photo_id = r["slug"], r["photo_id"]
        matches = sorted((image_dir / slug).glob(f"{photo_id}.*")) \
            if (image_dir / slug).exists() else []
        if len(matches) == 0:
            problems.append(f"{slug}/{photo_id}: no local image file")
            continue
        if len(matches) > 1:
            problems.append(f"{slug}/{photo_id}: {len(matches)} files match (ambiguous)")
            continue
        path = matches[0]
        if path.suffix.lower() not in IMG_EXTS:
            problems.append(f"{slug}/{photo_id}: unrecognized extension {path.suffix}")
            continue
        digest = sha256_file(path)
        if digest != r["sha256"]:
            problems.append(f"{slug}/{photo_id}: sha256 mismatch")
            continue

        label = slug_to_idx.get(slug, -1) if slug_to_idx else -1
        samples.append(Sample(str(path), max(label, 0), slug))
        meta.append({
            "photo_id": photo_id, "slug": slug, "species": r["species"],
            "category": r["category"], "taxon_id": r.get("taxon_id"),
            "known_label": label,
        })

    if problems or len(samples) != len(rows):
        print(f"[unknown-test-eval] FAILED verification: {len(problems)}/{len(rows)} "
             f"row(s) did not resolve to exactly one hash-verified image.")
        for p in problems[:50]:
            print(f"  ! {p}")
        raise SystemExit(
            f"[unknown-test-eval] Refusing to evaluate a partial or unverified test set "
            f"({len(samples)}/{len(rows)} rows OK). Restore with:\n"
            f"    python ../data_pipeline/scrape_unknown_test.py --restore --out {image_dir}"
        )
    print(f"[unknown-test-eval] verified {len(samples)}/{len(rows)} rows: exactly one "
         f"hash-matched image each")
    return samples, meta


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unknown-test-csv", type=Path,
                    default=HERE.parent / "data" / "unknown_test_v1" / "unknown_test_v1.csv")
    ap.add_argument("--unknown-test-dir", type=Path,
                    default=HERE.parent / "data" / "unknown_test_v1")
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--calibration-json", type=Path,
                    default=HERE.parent / "data" / "calibration_v1" / "calibration_v1.json",
                    help="Source of the frozen threshold (machine_readable_rule).")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent / "data" / "unknown_test_v1" / "unknown_test_v1_eval.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    taxonomy_raw = json.loads((args.artifacts / "taxonomy.json").read_text())
    taxonomy = {int(k): v for k, v in taxonomy_raw.items()}
    slug_to_idx = {v["slug"]: k for k, v in taxonomy.items()}
    protos_np = np.load(args.artifacts / "prototypes.npy")
    n_classes = protos_np.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or (args.artifacts / "model.pth")
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = AntIDModel(num_classes=n_classes, backbone=cfg["model"]["backbone"],
                       pretrained=False, dropout=cfg["dropout"],
                       embedding_dim=cfg["model"]["embedding_dim"]).to(device)
    model.load_state_dict(sd)
    model.eval()

    samples, meta = load_unknown_test(args.unknown_test_csv, args.unknown_test_dir, slug_to_idx)
    ds = AntDataset(samples, cfg, train=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg.get("num_workers", 0))

    protos = nn.functional.normalize(
        torch.as_tensor(protos_np, dtype=torch.float32, device=device), dim=1)

    records = []
    cursor = 0
    with torch.no_grad():
        for imgs, _labels in loader:
            imgs = imgs.to(device)
            emb = nn.functional.normalize(model.embed(imgs), dim=1)
            sims = (emb @ protos.T).cpu().numpy()
            for j in range(sims.shape[0]):
                s = sims[j]
                order = np.argsort(-s)
                top1_idx, top2_idx = int(order[0]), int(order[1])
                m = meta[cursor + j]
                records.append({
                    **m,
                    "max_sim": float(s[top1_idx]),
                    "top1_slug": taxonomy[top1_idx]["slug"],
                    "top1_is_true_known_class": bool(top1_idx == m["known_label"])
                                                if m["known_label"] >= 0 else None,
                    "top2_sim": float(s[top2_idx]),
                    "gap": float(s[top1_idx] - s[top2_idx]),
                })
            cursor += sims.shape[0]
    assert cursor == len(samples)

    # ---- apply the FROZEN rule, exactly once, no sweep ----
    T = load_frozen_threshold(args.calibration_json)
    known = [r for r in records if r["category"] == "known_holdout"]
    correct = [r for r in known if r["top1_is_true_known_class"]]
    incorrect = [r for r in known if not r["top1_is_true_known_class"]]
    accepted = [r for r in known if r["max_sim"] >= T]

    def rate(numer_list, cond):
        return sum(1 for r in numer_list if cond(r)) / len(numer_list) if numer_list else None

    known_rejection_rate = rate(known, lambda r: r["max_sim"] < T)
    correct_rejected_rate = rate(correct, lambda r: r["max_sim"] < T)
    incorrect_rejected_rate = rate(incorrect, lambda r: r["max_sim"] < T)
    accuracy_before = rate(known, lambda r: r["top1_is_true_known_class"])
    accuracy_among_accepted = rate(accepted, lambda r: r["top1_is_true_known_class"])

    far_by_category = {}
    for cat in ("out_of_scope_ant", "non_ant_insect", "unrelated"):
        pool = [r for r in records if r["category"] == cat]
        far_by_category[cat] = {
            "n": len(pool),
            "false_acceptance_rate": rate(pool, lambda r: r["max_sim"] >= T),
        }

    hashes = {
        "unknown_test_v1_csv": sha256_file(args.unknown_test_csv),
        "checkpoint": sha256_file(ckpt_path),
        "prototypes_npy": sha256_file(args.artifacts / "prototypes.npy"),
        "taxonomy_json": sha256_file(args.artifacts / "taxonomy.json"),
        "config_yaml": sha256_file(args.config),
        "calibration_v1_json": sha256_file(args.calibration_json),
    }

    results = {
        "frozen_threshold": T,
        "threshold_source": "data/calibration_v1/calibration_v1.json:"
                            "frozen_candidate_abstention_threshold.machine_readable_rule "
                            "(loaded and validated, not re-derived here)",
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "n_records": len(records),
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "hashes": hashes,
        "known_photo_rejection": {
            "description": "Fraction of ALL known_holdout images rejected (max_sim < "
                           "threshold), correct and incorrect predictions alike.",
            "n": len(known), "rate": known_rejection_rate,
        },
        "selective_classification": {
            "accuracy_before_abstention": accuracy_before,
            "correct_predictions_rejected": {"n": len(correct), "rate": correct_rejected_rate},
            "incorrect_predictions_rejected": {"n": len(incorrect), "rate": incorrect_rejected_rate},
            "accuracy_among_accepted": {"n": len(accepted), "rate": accuracy_among_accepted},
        },
        "false_acceptance_by_category": far_by_category,
        "records": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n[unknown-test-eval] frozen threshold = {T}")
    print(f"[unknown-test-eval] known_holdout n={len(known)}  "
         f"rejection_rate={known_rejection_rate*100:.1f}%  "
         f"accuracy_before={accuracy_before*100:.1f}%")
    print(f"[unknown-test-eval] correct rejected:   {correct_rejected_rate*100:.1f}% "
         f"(n={len(correct)})")
    print(f"[unknown-test-eval] incorrect rejected: {incorrect_rejected_rate*100:.1f}% "
         f"(n={len(incorrect)})")
    print(f"[unknown-test-eval] accuracy among accepted: {accuracy_among_accepted*100:.1f}% "
         f"(n={len(accepted)})")
    for cat, v in far_by_category.items():
        print(f"[unknown-test-eval] FAR {cat}: {v['false_acceptance_rate']*100:.1f}% (n={v['n']})")
    print(f"[unknown-test-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
