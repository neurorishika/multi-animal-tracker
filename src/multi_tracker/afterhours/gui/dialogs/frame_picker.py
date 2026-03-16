"""Frame picker dialog for MAT-afterhours.

Lets the user choose a split frame within a swap suspicion event's frame
range. All frames in the range are pre-decoded in a background thread so
the slider seeks instantly. Trajectory overlays for the two tracks under
review are baked in at load time so the user can clearly see both animals.
"""

from __future__ import annotations

_CROP_MARGIN = 80
_CONTEXT_FRAMES = 15  # extra frames loaded before / after the event window
