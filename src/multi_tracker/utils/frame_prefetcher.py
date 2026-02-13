"""
Frame prefetching utility for asynchronous video frame loading.

This module provides a thread-based frame prefetcher that reads video frames
in the background while the main tracking thread processes the current frame,
reducing I/O wait times and improving overall throughput.
"""

import logging
import queue
import threading

import cv2

logger = logging.getLogger(__name__)


class FramePrefetcher:
    """
    Asynchronous frame prefetcher for video processing.

    Uses a background thread to read frames ahead of time, reducing I/O blocking
    in the main processing loop. Maintains a small buffer of pre-read frames.

    Example:
        cap = cv2.VideoCapture("video.mp4")
        prefetcher = FramePrefetcher(cap, buffer_size=2)
        prefetcher.start()

        while True:
            ret, frame = prefetcher.read()
            if not ret:
                break
            # Process frame...

        prefetcher.stop()
    """

    def __init__(self, video_capture, buffer_size=2):
        """
        Initialize frame prefetcher.

        Args:
            video_capture: OpenCV VideoCapture object
            buffer_size (int): Number of frames to buffer (default: 2)
                              Higher = more memory, but better I/O tolerance
        """
        self.cap = video_capture
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_requested = threading.Event()
        self.exception = None  # Store exceptions from background thread
        self.thread = None
        self._started = False

    def start(self: object) -> object:
        """Start the background prefetching thread."""
        if self._started:
            logger.warning("FramePrefetcher already started")
            return

        self.stop_requested.clear()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()
        self._started = True
        logger.debug("FramePrefetcher started with buffer_size=%d", self.buffer_size)

    def _prefetch_loop(self):
        """Background thread loop that reads frames ahead of time."""
        try:
            while not self.stop_requested.is_set():
                # Read next frame from video
                ret, frame = self.cap.read()

                # Put frame in queue (blocks if queue is full)
                # Use timeout to allow periodic stop checks
                try:
                    self.frame_queue.put((ret, frame), timeout=0.1)
                except queue.Full:
                    # Queue is full, retry (allows checking stop_requested)
                    continue

                # If we've reached end of video, stop prefetching
                if not ret:
                    break

        except Exception as e:
            # Store exception to be raised in main thread
            self.exception = e
            logger.error("Exception in prefetcher thread: %s", e, exc_info=True)
            # Put sentinel to unblock main thread
            try:
                self.frame_queue.put((False, None), block=False)
            except queue.Full:
                pass

    def read(self: object) -> object:
        """
        Read the next frame (from prefetch buffer).

        Returns:
            tuple: (ret, frame) where ret is bool indicating success,
                   and frame is the numpy array or None
        """
        # Check for exceptions from background thread
        if self.exception is not None:
            raise RuntimeError("Prefetcher thread failed") from self.exception

        try:
            # Get frame from queue with timeout to detect stalls
            ret, frame = self.frame_queue.get(timeout=5.0)
            return ret, frame
        except queue.Empty:
            # Queue is empty after timeout - likely stalled
            logger.error("Frame prefetcher timeout - no frames available")
            return False, None

    def stop(self: object) -> object:
        """Stop the prefetching thread and clean up."""
        if not self._started:
            return

        logger.debug("Stopping FramePrefetcher...")
        self.stop_requested.set()

        # Wait for thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Prefetcher thread did not stop cleanly")

        # Clear any remaining frames from queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self._started = False
        logger.debug("FramePrefetcher stopped")

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()
        return False


class FramePrefetcherBackward(FramePrefetcher):
    """
    Frame prefetcher for backward (reverse) video iteration.

    Extends FramePrefetcher to support reading frames in reverse order
    by seeking backward through the video.
    """

    def __init__(self, video_capture, buffer_size=2, total_frames=None):
        """
        Initialize backward frame prefetcher.

        Args:
            video_capture: OpenCV VideoCapture object
            buffer_size (int): Number of frames to buffer
            total_frames (int): Total frames in video (required for backward seeking)
        """
        super().__init__(video_capture, buffer_size)
        self.total_frames = total_frames
        if self.total_frames is None:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = self.total_frames - 1

    def _prefetch_loop(self):
        """Background thread loop that reads frames in reverse order."""
        try:
            while not self.stop_requested.is_set() and self.current_frame_idx >= 0:
                # Seek to current frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.cap.read()

                # Put frame in queue
                try:
                    self.frame_queue.put((ret, frame), timeout=0.1)
                except queue.Full:
                    continue

                if not ret:
                    logger.warning(
                        "Failed to read frame %d in backward mode",
                        self.current_frame_idx,
                    )

                # Move to previous frame
                self.current_frame_idx -= 1

            # Signal end of video
            try:
                self.frame_queue.put((False, None), timeout=0.1)
            except queue.Full:
                pass

        except Exception as e:
            self.exception = e
            logger.error(
                "Exception in backward prefetcher thread: %s", e, exc_info=True
            )
            try:
                self.frame_queue.put((False, None), block=False)
            except queue.Full:
                pass
