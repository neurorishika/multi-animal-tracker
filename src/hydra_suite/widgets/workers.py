"""BaseWorker — standard QThread base class for all background tasks."""

from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Base class for all background task workers.

    Subclasses implement ``execute()`` only.  ``run()`` is owned by this
    class and guarantees:
    - ``finished`` is always emitted (success or failure).
    - Unhandled exceptions in ``execute()`` emit ``error`` instead of
      crashing the thread silently.

    Standard signals
    ----------------
    progress(int)  — 0–100 completion percentage
    status(str)    — human-readable status update
    error(str)     — error message; emitted only on exception
    finished()     — inherited from QThread; always emitted when the worker stops
    """

    progress: Signal = Signal(int)
    status: Signal = Signal(str)
    error: Signal = Signal(str)

    def run(self) -> None:
        try:
            self.execute()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def execute(self) -> None:
        """Override in subclasses with the actual work."""
        raise NotImplementedError
