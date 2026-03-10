"""
Atomic correction writer for _proofread.csv.

Applies split + identity swap corrections to the proofread copy.
Never touches the original CSV.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_NEW_ID_OFFSET = 100_000


def apply_split_and_swap(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    split_frame: int,
    swap_post: bool,
) -> pd.DataFrame:
    """
    Split track_a and track_b at split_frame, optionally swapping post-split IDs.

    New segment IDs:
      track_a pre-split  -> track_a  (unchanged)
      track_a post-split -> track_b + _NEW_ID_OFFSET  (if swapped)
                            or track_a + _NEW_ID_OFFSET
      track_b pre-split  -> track_b  (unchanged)
      track_b post-split -> track_a + _NEW_ID_OFFSET  (if swapped)
                            or track_b + _NEW_ID_OFFSET
    """
    df = df.copy()
    mask_a_post = (df["TrajectoryID"] == track_a) & (df["FrameID"] >= split_frame)
    mask_b_post = (df["TrajectoryID"] == track_b) & (df["FrameID"] >= split_frame)

    if swap_post:
        df.loc[mask_a_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
    else:
        df.loc[mask_a_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET

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

    def apply_correction(
        self,
        track_a: int,
        track_b: int,
        split_frame: int,
        swap_post: bool,
    ) -> None:
        """Apply a split+swap correction and write atomically."""
        if self._df is None:
            raise RuntimeError("Call open() before apply_correction()")
        self._df = apply_split_and_swap(
            self._df,
            track_a,
            track_b,
            split_frame,
            swap_post,
        )
        self._write_atomic()

    def _write_atomic(self) -> None:
        """Write to .tmp then atomically replace the proofread file."""
        tmp = self.proofread_path.with_suffix(".tmp")
        self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.proofread_path)

    def close(self) -> None:
        """Release the in-memory DataFrame."""
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Return the current in-memory DataFrame."""
        if self._df is None:
            raise RuntimeError("Call open() first")
        return self._df
