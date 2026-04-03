"""
Utility functions for asynchronous CSV writing in multi-animal tracking.
"""

import csv
import logging
import queue
import threading

logger = logging.getLogger(__name__)


class CSVWriterThread(threading.Thread):
    """
    Asynchronous CSV writer for high-performance data logging.

    This thread-safe class handles CSV writing in the background to prevent
    I/O operations from blocking the main tracking loop. It uses a queue-based
    system to buffer data and ensures data integrity during high-frequency writes.

    The CSV format includes:
    - TrackID: Track slot identifier (0-based index, reused across track losses)
    - TrajectoryID: Persistent trajectory identifier (increments when tracks are reassigned)
    - Index: Sequential count of detections for this track slot
    - X, Y: Pixel coordinates of object center
    - Theta: Orientation angle in radians
    - FrameID: Video frame number (1-based)
    - State: Tracking state ('active', 'occluded', 'lost')
    """

    def __init__(self, path: str, header=None):
        """
        Initialize CSV writer thread.

        Args:
            path (str): Output CSV file path
            header (list, optional): Column names for CSV header
        """
        super().__init__()
        self.csv_path = path
        self.header = header or []
        self.queue = queue.Queue()  # Thread-safe queue for data buffering
        self._stop_requested = False  # Shutdown flag

        # Open file and write header immediately
        self.f = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.f)
        if self.header:
            self.writer.writerow(self.header)

    def run(self: object) -> object:
        """
        Main thread loop for processing queued data.

        Continuously processes queued data rows until stop signal is received
        and all remaining data is flushed.
        """
        try:
            while not self._stop_requested or not self.queue.empty():
                try:
                    # Wait for data with timeout to allow periodic stop checks
                    row = self.queue.get(timeout=0.3)
                    self.writer.writerow(row)
                    self.queue.task_done()
                except queue.Empty:
                    continue  # Timeout occurred, check stop condition
        finally:
            # Ensure all data is written before closing
            self.f.flush()
            self.f.close()

    def enqueue(self: object, row: object) -> object:
        """
        Add a data row to the write queue.

        Args:
            row (list): Data row to write to CSV
        """
        self.queue.put(row)

    def stop(self: object) -> object:
        """Signal the thread to stop processing and shutdown gracefully."""
        self._stop_requested = True
