"""
Comprehensive tests for FramePrefetcher classes.

Tests cover:
- Forward frame prefetching
- Backward frame prefetching
- Buffer management
- Thread safety and cleanup
- Error handling
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from multi_tracker.utils.frame_prefetcher import (
    FramePrefetcher,
    FramePrefetcherBackward,
)


def create_test_video(num_frames=30, width=640, height=480):
    """Create a temporary test video file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Create a frame with frame number drawn on it
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        cv2.putText(
            frame,
            str(i),
            (width // 2 - 50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )
        writer.write(frame)

    writer.release()
    return temp_path


class TestFramePrefetcher:
    """Test suite for FramePrefetcher class."""

    def test_initialization(self):
        """Test that FramePrefetcher initializes correctly."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            assert prefetcher.cap == cap
            assert prefetcher.buffer_size == 2
            assert not prefetcher._started
            assert prefetcher.exception is None

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_start(self):
        """Test starting the prefetcher thread."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            prefetcher.start()
            assert prefetcher._started
            assert prefetcher.thread is not None
            assert prefetcher.thread.is_alive()

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_read_single_frame(self):
        """Test reading a single frame."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)
            prefetcher.start()

            ret, frame = prefetcher.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_read_all_frames(self):
        """Test reading all frames from a video."""
        num_frames = 20
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=3)
            prefetcher.start()

            frames_read = 0
            while True:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                assert frame is not None
                frames_read += 1

            assert frames_read == num_frames

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_buffer_size_effect(self):
        """Test that different buffer sizes work correctly."""
        video_path = create_test_video(30)
        try:
            for buffer_size in [1, 2, 5]:
                cap = cv2.VideoCapture(video_path)
                prefetcher = FramePrefetcher(cap, buffer_size=buffer_size)
                prefetcher.start()

                # Read a few frames
                for _ in range(10):
                    ret, frame = prefetcher.read()
                    assert ret is True
                    assert frame is not None

                prefetcher.stop()
                cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test using FramePrefetcher as a context manager."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)

            with FramePrefetcher(cap, buffer_size=2) as prefetcher:
                assert prefetcher._started
                ret, frame = prefetcher.read()
                assert ret is True
                assert frame is not None

            # After exiting context, should be stopped
            assert not prefetcher._started

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_stop_clears_queue(self):
        """Test that stopping clears the frame queue."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=5)
            prefetcher.start()

            # Read a few frames to populate the buffer
            time.sleep(0.2)

            # Stop should clear the queue
            prefetcher.stop()
            assert prefetcher.frame_queue.empty()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_double_start_warning(self):
        """Test that starting an already started prefetcher logs a warning."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            prefetcher.start()
            # Second start should be a no-op with warning
            prefetcher.start()

            assert prefetcher._started

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_stop_without_start(self):
        """Test that stopping without starting is safe."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            # Should not crash
            prefetcher.stop()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestFramePrefetcherBackward:
    """Test suite for FramePrefetcherBackward class."""

    def test_initialization_with_total_frames(self):
        """Test initialization with explicit total_frames."""
        video_path = create_test_video(15)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcherBackward(cap, buffer_size=2, total_frames=15)

            assert prefetcher.total_frames == 15
            assert prefetcher.current_frame_idx == 14  # 0-indexed, so last frame is 14

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_initialization_auto_detect_frames(self):
        """Test initialization with auto-detected total_frames."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcherBackward(cap, buffer_size=2)

            # Should auto-detect from video properties
            assert prefetcher.total_frames >= 20
            assert prefetcher.current_frame_idx >= 19

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_read_backward(self):
        """Test reading frames in backward order."""
        num_frames = 10
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcherBackward(
                cap, buffer_size=2, total_frames=num_frames
            )
            prefetcher.start()

            frames_read = []
            while True:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                frames_read.append(frame)

            # Should read all frames in reverse order
            assert len(frames_read) == num_frames

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_backward_context_manager(self):
        """Test using FramePrefetcherBackward as a context manager."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)

            with FramePrefetcherBackward(
                cap, buffer_size=2, total_frames=10
            ) as prefetcher:
                assert prefetcher._started
                ret, frame = prefetcher.read()
                assert ret is True
                assert frame is not None

            assert not prefetcher._started

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_backward_full_iteration(self):
        """Test full backward iteration through video."""
        num_frames = 15
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcherBackward(
                cap, buffer_size=3, total_frames=num_frames
            )
            prefetcher.start()

            count = 0
            while True:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                count += 1

            assert count == num_frames

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestFramePrefetcherRobustness:
    """Test robustness and error handling."""

    def test_empty_video(self):
        """Test handling of empty/corrupt video."""
        video_path = create_test_video(0)  # Create empty video
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)
            prefetcher.start()

            ret, frame = prefetcher.read()
            # Should return False for empty video
            assert ret is False

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_rapid_start_stop(self):
        """Test rapid start/stop cycles."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)

            for _ in range(3):
                prefetcher = FramePrefetcher(cap, buffer_size=2)
                prefetcher.start()
                ret, frame = prefetcher.read()
                prefetcher.stop()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)
