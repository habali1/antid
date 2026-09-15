#!/usr/bin/env python3
"""pair_adjudication_v2_review.py — local side-by-side Tkinter reviewer
for the Phase 5F4 v2 adjudication queue.

ALL PIL/Tkinter imports, and all image opening, are deferred to inside
launch_review_ui() -- importing this module, or calling any function
other than launch_review_ui()/main(), never touches PIL, Tkinter, or an
image file. This lets preparation-phase tests exercise every piece of
this module's OWN logic (progress/session-boundary computation, the
two-step select+confirm state machine, field construction) with
synthetic PIL images and a stub Tk-like object, without ever launching a
real window or opening a real candidate image.

Uses ONLY the shared v2 authority/recording functions from
adjudicate_pairs_v2.py (load_verified_v2_state, first_eligible_row,
record_adjudication) -- this module never re-implements validation,
locking, or ledger I/O.

UI contract:
  - Shows both images side by side, plus domain, identities, pHash/dHash
    distances, workstream, overall progress, and the current session
    block.
  - Four labeled buttons/keys (one per frozen adjudication label) select
    a pending judgment; a separate Confirm button/key is required to
    actually record it -- a single accidental keypress can never
    permanently append a judgment.
  - Never auto-advances across a session/domain boundary: after
    confirming the LAST row of a session block, the UI stops and reports
    the boundary rather than silently loading the next domain's first
    row.
  - Never opens the next pair's images before the current record has
    been fsynced and post-verified by adjudicate_pairs_v2.record_adjudication
    (which itself only returns after that verification).

Run with --launch to open the real window (NOT invoked during
preparation/testing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import pair_adjudication_v2_contract as v2c  # noqa: E402
import adjudicate_pairs_v2 as adj2  # noqa: E402
import manual_review_contract as mrc  # noqa: E402

LABEL_KEYS = {"1": "same_source_image", "2": "different_photo_same_observation",
             "3": "different_image", "4": "uncertain"}
assert set(LABEL_KEYS.values()) == v2c.ADJUDICATION_LABELS


class ReviewUIError(RuntimeError):
    """A state or resolution problem. Always fails closed."""


def _fail(msg: str) -> None:
    raise ReviewUIError(msg)


def load_v1_contract_content(repo: Path) -> dict:
    """The v1 (Phase 5F1) frozen contract supplies the ONLY
    domain_part_image_layouts this module trusts for image-path
    resolution -- v2 defines no layout rules of its own. Verifies the
    loaded contract's content_sha256 against the v2 contract's own
    PHASE_5F1_CONTRACT_CONTENT_SHA256 binding before trusting it."""
    import json
    path = Path(repo) / mrc.CONTRACT_REL_PATH
    if not path.exists():
        _fail(f"Phase 5F1 contract does not exist: {path}")
    contract = json.loads(path.read_bytes().decode("utf-8"))
    problems = mrc.validate_contract_structure(contract)
    if problems:
        _fail(f"Phase 5F1 contract failed structural validation: {problems}")
    if contract["content_sha256"] != v2c.PHASE_5F1_CONTRACT_CONTENT_SHA256:
        _fail("Phase 5F1 contract content_sha256 does not match the frozen v2 binding")
    return contract["content"]


def image_paths_for_row(repo: Path, v1_contract_content: dict, row: dict) -> tuple[Path, Path]:
    """Never opens either file -- Path construction/.exists() only."""
    a = v2c.resolve_image_path_for_identity(repo, v1_contract_content, row["identity_a"])
    b = v2c.resolve_image_path_for_identity(repo, v1_contract_content, row["identity_b"])
    return a, b


def session_progress(queue_rows: list[dict], records: list[dict], current_row: dict) -> dict:
    """Pure, metadata-only: describes the current row's position within
    its own session block, and overall queue progress -- used both by
    the real UI and by preparation-phase tests (with synthetic rows)."""
    session_rows = sorted((r for r in queue_rows if r["suggested_session"] == current_row["suggested_session"]),
                          key=lambda r: r["queue_index"])
    position_in_session = next(i for i, r in enumerate(session_rows) if r["pair_id"] == current_row["pair_id"])
    adjudicated = {r.get("pair_id") for r in records}
    return {
        "suggested_session": current_row["suggested_session"], "domain": current_row["domain"],
        "workstream": current_row["workstream"], "session_block": current_row["session_block"],
        "position_in_session": position_in_session + 1, "session_row_count": len(session_rows),
        "is_last_row_in_session": position_in_session == len(session_rows) - 1,
        "overall_adjudicated_count": len(adjudicated), "overall_total_count": len(queue_rows),
    }


class ReviewSession:
    """The UI-independent state machine: select -> confirm -> record.
    Exercised directly by preparation-phase tests without any Tkinter or
    PIL dependency. The real Tkinter UI (built only inside
    launch_review_ui()) is a thin wrapper around this class."""

    def __init__(self, repo: Path, *, reviewer_id: str, session_id: str):
        self.repo = Path(repo)
        self.reviewer_id = reviewer_id
        self.session_id = session_id
        self.pending_label: str | None = None
        self.state = adj2.load_verified_v2_state(self.repo)
        self.v1_contract_content = load_v1_contract_content(self.repo)
        # Refuse to launch/record for a session different from the
        # current frozen block: session_id is never an arbitrary label,
        # it must be exactly the eligible row's own suggested_session
        # (record_adjudication() would reject a mismatch too, but this
        # UI-layer check fails BEFORE any image is opened or any pending
        # selection can be made against the wrong block).
        current = self.global_current_row()
        if current is not None and session_id != current["suggested_session"]:
            _fail(f"session_id {session_id!r} does not match the current frozen queue block "
                  f"{current['suggested_session']!r} -- refusing to launch/record for a different session; "
                  f"pass --session-id {current['suggested_session']!r}")

    def current_records(self) -> list[dict]:
        return adj2.load_verified_ledger_v2(self.repo, self.state)

    def global_current_row(self) -> dict | None:
        """Return the protocol-wide next eligible row, without opening an
        image. This is used only to establish/describe a session boundary."""
        return adj2.first_eligible_row(self.state["queue_rows"], self.current_records())

    def current_row(self) -> dict | None:
        """Return the next row only while it belongs to this frozen session.

        A blocking decision can stop a workstream in the middle of a block.
        In that case the protocol-wide next row may immediately belong to a
        different session. Returning None here prevents load_current() from
        opening that next session's images before an explicit relaunch.
        """
        row = self.global_current_row()
        if row is not None and row["suggested_session"] != self.session_id:
            return None
        return row

    def next_session_id(self) -> str | None:
        row = self.global_current_row()
        return row["suggested_session"] if row is not None else None

    def select(self, label: str) -> None:
        """Step 1 of 2: marks a pending judgment. Records nothing."""
        if label not in v2c.ADJUDICATION_LABELS:
            _fail(f"label must be one of {sorted(v2c.ADJUDICATION_LABELS)}, got {label!r}")
        self.pending_label = label

    def clear_selection(self) -> None:
        self.pending_label = None

    def confirm(self, *, reviewed_at_utc: str, notes: str = "") -> dict:
        """Step 2 of 2: the ONLY call in this class that actually appends
        a ledger record -- requires a prior select(). Delegates entirely
        to adjudicate_pairs_v2.record_adjudication, which itself acquires
        the exclusive lock, reverifies, fsyncs, and post-verifies before
        returning; this method never opens the next pair's images before
        that call returns."""
        if self.pending_label is None:
            _fail("confirm() called with no pending selection -- select a label first "
                  "(two-step select+confirm: a single keypress can never record a judgment)")
        row = self.current_row()
        if row is None:
            _fail("no pair is currently eligible for adjudication")
        record = adj2.record_adjudication(
            self.repo, pair_id=row["pair_id"], label=self.pending_label, reviewer_id=self.reviewer_id,
            session_id=self.session_id, reviewed_at_utc=reviewed_at_utc, notes=notes)
        self.pending_label = None
        return record

    def progress(self) -> dict | None:
        row = self.current_row()
        if row is None:
            return None
        return session_progress(self.state["queue_rows"], self.current_records(), row)


# ------------------------------------------------------------- real UI --
def launch_review_ui(repo: Path, *, reviewer_id: str, session_id: str) -> None:
    """Opens the real Tkinter window. PIL/Tkinter are imported HERE, not
    at module scope, and no image is opened until this function runs.
    Never auto-advances past a session boundary: after the last row of a
    block, or after a blocking decision stops a workstream mid-block, the
    window reports the boundary and waits for the reviewer to explicitly
    start the next frozen session rather than loading it itself."""
    import datetime
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk

    session = ReviewSession(repo, reviewer_id=reviewer_id, session_id=session_id)

    root = tk.Tk()
    root.title("Phase 5F4 Pair Adjudication v2")

    status_var = tk.StringVar()
    progress_var = tk.StringVar()
    canvas_a = tk.Label(root)
    canvas_b = tk.Label(root)
    canvas_a.grid(row=0, column=0, padx=8, pady=8)
    canvas_b.grid(row=0, column=1, padx=8, pady=8)
    tk.Label(root, textvariable=status_var, justify="left").grid(row=1, column=0, columnspan=2, sticky="w")
    tk.Label(root, textvariable=progress_var, justify="left").grid(row=2, column=0, columnspan=2, sticky="w")

    photo_refs: list = []  # keep references alive against GC

    def load_current() -> None:
        photo_refs.clear()
        row = session.current_row()
        if row is None:
            status_var.set("Done -- no pair is currently eligible for adjudication.")
            progress_var.set("")
            canvas_a.configure(image="")
            canvas_b.configure(image="")
            return
        path_a, path_b = image_paths_for_row(session.repo, session.v1_contract_content, row)
        img_a = Image.open(path_a)
        img_b = Image.open(path_b)
        img_a.thumbnail((480, 480))
        img_b.thumbnail((480, 480))
        photo_a = ImageTk.PhotoImage(img_a)
        photo_b = ImageTk.PhotoImage(img_b)
        photo_refs.extend([photo_a, photo_b])
        canvas_a.configure(image=photo_a)
        canvas_b.configure(image=photo_b)
        status_var.set(f"domain={row['domain']} workstream={row['workstream']} "
                       f"phash={row['phash_distance']} dhash={row['dhash_distance']}\n"
                       f"a={row['identity_a']}\nb={row['identity_b']}\n"
                       f"pending selection: {session.pending_label or '(none)'}")
        progress = session.progress()
        progress_var.set(f"session={progress['suggested_session']} "
                         f"row {progress['position_in_session']}/{progress['session_row_count']} "
                         f"overall {progress['overall_adjudicated_count']}/{progress['overall_total_count']}")

    def on_select(label: str) -> None:
        session.select(label)
        load_current()

    def on_confirm() -> None:
        if session.pending_label is None:
            messagebox.showwarning("No selection", "Select a label before confirming.")
            return
        row_before = session.current_row()
        record = session.confirm(reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"))
        next_session_id = session.next_session_id()
        if next_session_id != session.session_id:
            next_text = (f" Next frozen session: {next_session_id}." if next_session_id is not None
                         else " No eligible pair remains.")
            status_var.set(f"Recorded {record['pair_id']}. Session {row_before['suggested_session']} "
                           f"complete or blocking-stopped -- stopping at the session boundary before "
                           f"opening another image.{next_text} Close and relaunch explicitly to continue.")
            progress_var.set("")
            canvas_a.configure(image="")
            canvas_b.configure(image="")
            return
        load_current()

    button_frame = tk.Frame(root)
    button_frame.grid(row=3, column=0, columnspan=2, pady=8)
    for key, label in LABEL_KEYS.items():
        tk.Button(button_frame, text=f"[{key}] {label}", command=lambda l=label: on_select(l)).pack(side="left", padx=4)
    tk.Button(root, text="Confirm", command=on_confirm, bg="#c8e6c9").grid(row=4, column=0, columnspan=2, pady=8)

    for key, label in LABEL_KEYS.items():
        root.bind(key, lambda event, l=label: on_select(l))
    root.bind("<Return>", lambda event: on_confirm())

    load_current()
    root.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--launch", action="store_true", help="opens the real Tkinter window")
    ap.add_argument("--reviewer-id")
    ap.add_argument("--session-id")
    args = ap.parse_args()

    if not args.launch:
        print("pair_adjudication_v2_review.py: pass --launch to open the real reviewer window "
              "(not run automatically).")
        return 0
    if not args.reviewer_id or not args.session_id:
        print("[pair_adjudication_v2_review] FAILURE: --launch requires --reviewer-id and --session-id")
        return 1
    try:
        launch_review_ui(args.repo, reviewer_id=args.reviewer_id, session_id=args.session_id)
        return 0
    except (ReviewUIError, adj2.AdjudicationV2Error) as e:
        print(f"[pair_adjudication_v2_review] FAILURE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
