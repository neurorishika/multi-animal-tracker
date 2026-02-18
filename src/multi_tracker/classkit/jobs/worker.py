import sys
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.str)
    result
        object data returned from processing, anything
    progress
        int indication of progress
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handle worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    """

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the progress callback to kwargs if the function accepts it contextually
        # BUT for generic usage, we usually assume the function takes 'progress_callback' or similar
        # For now, let's inject it into kwargs if needed or let the user pass it via kwargs mapping

        # We'll just pass signals to be safe if the function wants to emit progress
        # self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
