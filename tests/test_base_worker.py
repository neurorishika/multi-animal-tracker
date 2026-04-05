# tests/test_base_worker.py
import sys

import pytest
from PySide6.QtCore import QCoreApplication


@pytest.fixture(scope="session")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv[:1])
    return app


def test_base_worker_execute_called(qapp):
    """execute() is called when worker is started."""
    from hydra_suite.widgets.workers import BaseWorker

    class _EchoWorker(BaseWorker):
        def execute(self):
            self.status.emit("hello")
            self.progress.emit(100)

    received = []
    worker = _EchoWorker()
    worker.status.connect(received.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert received == ["hello"]


def test_base_worker_finished_always_fires(qapp):
    """finished signal fires even when execute raises."""
    from hydra_suite.widgets.workers import BaseWorker

    class _CrashWorker(BaseWorker):
        def execute(self):
            raise RuntimeError("boom")

    finished_calls = []
    errors = []
    worker = _CrashWorker()
    worker.finished.connect(lambda: finished_calls.append(1))
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert len(finished_calls) == 1
    assert "boom" in errors[0]


def test_base_worker_error_emitted_on_exception(qapp):
    """error signal carries the exception message."""
    from hydra_suite.widgets.workers import BaseWorker

    class _BadWorker(BaseWorker):
        def execute(self):
            raise ValueError("bad value")

    errors = []
    worker = _BadWorker()
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert len(errors) == 1
    assert "bad value" in errors[0]


def test_base_worker_no_error_on_success(qapp):
    """error signal is not emitted when execute succeeds, but finished is."""
    from hydra_suite.widgets.workers import BaseWorker

    class _OkWorker(BaseWorker):
        def execute(self):
            pass

    errors = []
    finished_calls = []
    worker = _OkWorker()
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished_calls.append(1))
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert errors == []
    assert len(finished_calls) == 1


def test_base_worker_subclass_extra_signals(qapp):
    """Subclasses can add extra signals beyond the base set."""
    from PySide6.QtCore import Signal

    from hydra_suite.widgets.workers import BaseWorker

    class _ResultWorker(BaseWorker):
        result = Signal(int)

        def execute(self):
            self.result.emit(42)

    results = []
    worker = _ResultWorker()
    worker.result.connect(results.append)
    worker.start()
    worker.wait(3000)
    QCoreApplication.processEvents()

    assert results == [42]
