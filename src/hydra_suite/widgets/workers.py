"""BaseWorker — standard QThread base class for all background tasks."""

from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Base class for all background task workers.

    Subclasses implement ``execute()`` only.  ``run()`` is owned by this
    class and guarantees:
    - Unhandled exceptions in ``execute()`` emit ``error`` instead of
      crashing the thread silently.
    - Qt automatically emits the inherited ``QThread.finished`` signal
      when ``run()`` returns, whether execution succeeded or failed.
      Do not redefine ``finished`` in subclasses — it would shadow
      Qt's mechanism and cause missed or double emissions.

    Standard signals
    ----------------
    progress(int)  — 0–100 completion percentage
    status(str)    — human-readable status update
    error(str)     — error message; emitted only on exception
    finished()     — inherited from QThread; emitted automatically by Qt
                     when run() returns (success or failure)
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
        raise NotImplementedError(f"{type(self).__name__} must implement execute()")
