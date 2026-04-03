"""
Atomic correction writer for _proofread.csv.

Applies identity corrections to the proofread copy.
Supports: split+swap, merge fragments, delete track, erase flicker,
reassign chain (N-way relabeling), and fragment-level edit ops from the
track editor.
Never touches the original CSV.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from hydra_suite.refinekit.core.track_editor_model import EditOp, OpKind

logger = logging.getLogger(__name__)

_NEW_ID_OFFSET = 100_000


# ---------------------------------------------------------------------------
# Atomic operations (pure functions on DataFrames)
# ---------------------------------------------------------------------------


def merge_fragments(
    df: pd.DataFrame,
    track_ids: List[int],
) -> pd.DataFrame:
    """Merge multiple trajectory IDs into the lowest ID.

    All rows whose TrajectoryID is in *track_ids* are relabeled to
    ``min(track_ids)``.
    """
    if len(track_ids) < 2:
        return df
    df = df.copy()
    target = min(track_ids)
    for tid in track_ids:
        if tid != target:
            df.loc[df["TrajectoryID"] == tid, "TrajectoryID"] = target
    return df


class CorrectionWriter:
    """
    Manages the _proofread.csv lifecycle: open -> apply corrections -> close.

    Creates a proofread copy once from the original CSV. Subsequent opens
    load the existing proofread copy without overwriting.
    """

    def __init__(self, source_csv: Path | str):
        self.source_csv = Path(source_csv)
        stem = self.source_csv.stem
        self.proofread_path = self.source_csv.with_name(f"{stem}_proofread.csv")
        self._df: pd.DataFrame | None = None

    def open(self) -> None:
        """Create proofread copy if needed, then load it into memory."""
        if not self.proofread_path.exists():
            shutil.copy2(self.source_csv, self.proofread_path)
            logger.info("Created proofread copy: %s", self.proofread_path)
        self._df = pd.read_csv(self.proofread_path)

    def apply_merge(self, track_ids: List[int]) -> None:
        """Merge fragment track IDs into one and persist."""
        if self._df is None:
            raise RuntimeError("Call open() before apply_merge()")
        self._df = merge_fragments(self._df, track_ids)
        self._write_atomic()

    def apply_swap_merge(
        self,
        source_id: int,
        target_id: int,
        swap_frame: int,
    ) -> None:
        """Fix an identity swap by relabeling *target*'s post-swap rows.

        After this operation:

        * *source_id* has a continuous trajectory (original pre-swap data
          plus *target*'s post-swap data).
        * *target_id* ends at ``swap_frame - 1`` (becomes a dead fragment).
        * Any (wrong) detections attributed to *source* at or after
          *swap_frame* are removed.
        """
        if self._df is None:
            raise RuntimeError("Call open() before apply_swap_merge()")
        df = self._df.copy()

        # Remove source's rows at/after swap_frame (they were wrong)
        mask_remove = (df["TrajectoryID"] == source_id) & (df["FrameID"] >= swap_frame)
        df = df[~mask_remove]

        # Relabel target's post-swap rows to source_id
        mask_relabel = (df["TrajectoryID"] == target_id) & (df["FrameID"] >= swap_frame)
        df.loc[mask_relabel, "TrajectoryID"] = source_id

        self._df = df
        self._write_atomic()

    def _write_atomic(self) -> None:
        """Write to .tmp then atomically replace the proofread file."""
        tmp = self.proofread_path.with_suffix(".tmp")
        self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.proofread_path)

    def apply_edit_ops(self, ops: List[EditOp]) -> None:
        """Apply a batch of fragment-level edit operations and persist.

        Operations are applied in order: DELETEs first, then REASSIGNs.
        """
        if self._df is None:
            raise RuntimeError("Call open() before apply_edit_ops()")
        df = self._df.copy()

        # Deletes first (so reassign doesn't move rows we want to remove)
        for op in ops:
            if op.kind == OpKind.DELETE:
                mask = (
                    (df["TrajectoryID"] == op.track_id)
                    & (df["FrameID"] >= op.frame_start)
                    & (df["FrameID"] <= op.frame_end)
                )
                df = df[~mask]

        # Reassigns — use temp offset to avoid collision
        offset = _NEW_ID_OFFSET
        for op in ops:
            if op.kind == OpKind.REASSIGN and op.new_track_id is not None:
                mask = (
                    (df["TrajectoryID"] == op.track_id)
                    & (df["FrameID"] >= op.frame_start)
                    & (df["FrameID"] <= op.frame_end)
                )
                df.loc[mask, "TrajectoryID"] = op.new_track_id + offset

        # Resolve temp IDs
        df.loc[
            df["TrajectoryID"] >= offset,
            "TrajectoryID",
        ] = (
            df.loc[df["TrajectoryID"] >= offset, "TrajectoryID"] - offset
        )

        self._df = df.reset_index(drop=True)
        self._write_atomic()

    def close(self) -> None:
        """Release the in-memory DataFrame."""
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Return the current in-memory DataFrame."""
        if self._df is None:
            raise RuntimeError("Call open() first")
        return self._df
