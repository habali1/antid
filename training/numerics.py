"""numerics.py — pinned full-FP32 numerical policy and deterministic RNG
setup shared by train.py, evaluate.py, and eval_benchmark.py.

This exists so all three scripts apply and report the *same* explicit
numerical policy rather than each silently inheriting whatever PyTorch's
installed-version ambient defaults happen to be. Recon on this environment
found a mixed-precision default: cudnn.allow_tf32=True (backbone convolutions
may use reduced-precision TF32) while cuda.matmul.allow_tf32=False /
float32_matmul_precision="highest" (the final matmul/cosine path is already
full FP32). apply_numerical_policy() below pins ALL of these to full FP32.

IMPORTANT: this does NOT make PyTorch-CUDA numerically bit-identical to the
ONNX-CPU serving path. It only removes one ambient reduced-precision
variable (TF32) and the run-to-run cudnn algorithm-selection variable
(benchmark mode). ONNX-CPU parity for a new model is a separate, later
required step -- do not cite this module as proof of serving parity.
"""
from __future__ import annotations

import random

import numpy as np
import torch

# The single source of truth for the pinned policy. Every value here is a
# deliberate choice for the first controlled B4-65 development run, not a
# default being read back -- see apply_numerical_policy().
NUMERICAL_POLICY: dict = {
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "amp": False,
}


def apply_numerical_policy() -> dict:
    """Apply the pinned full-FP32 numerical policy process-wide.

    Idempotent: always sets the same fixed values (never reads ambient
    state), so calling it more than once in a process (or once each in
    train.py and evaluate.py against the same checkpoint) is safe and has no
    cumulative effect. Returns NUMERICAL_POLICY for logging/provenance.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(NUMERICAL_POLICY)


def current_numerical_policy() -> dict:
    """Read back the ambient state of the same knobs NUMERICAL_POLICY pins.

    Used to confirm apply_numerical_policy() actually took effect (e.g. in
    run_manifest.json) rather than trusting that the call happened.
    """
    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "amp": False,
    }


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python random, NumPy, torch CPU, and torch CUDA (all devices)
    from one integer seed. Returns a freshly seeded torch.Generator for the
    caller to pass explicitly as DataLoader(..., generator=...) -- rather
    than relying on the implicit global default generator, whose state
    silently depends on whatever else has already consumed random draws.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the default generator on every CUDA device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # explicit, not relied on implicitly
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """Top-level, Windows-picklable per-DataLoader-worker seeding.

    Windows DataLoader workers are spawned (not forked), so worker_init_fn
    must be an importable top-level function -- a closure or lambda would
    fail to pickle and crash at DataLoader startup on this platform.

    torch already offsets each worker's default RNG uniquely before calling
    this hook (torch.initial_seed() returns a per-worker value derived from
    the main process's seeded state), so deriving Python's `random` and
    NumPy's RNG from it here makes every source of randomness inside a
    worker process (e.g. any future Pillow-side randomness) deterministic
    too, not just torch's own calls.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict:
    """Snapshot every RNG this module seeds, for a resumable checkpoint."""
    state = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Inverse of capture_rng_state(). Restores exactly what was captured."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
