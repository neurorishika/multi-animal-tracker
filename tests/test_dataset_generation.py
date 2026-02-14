"""
Tests for FrameQualityScorer class (dataset generation).

Tests cover:
- Frame quality scoring based on various metrics
- Low confidence detection
- Detection count mismatch
- High assignment costs
- Metric enabling/disabling
"""

import numpy as np

from multi_tracker.data.dataset_generation import FrameQualityScorer


class TestFrameQualityScorer:
    """Test suite for FrameQualityScorer class."""

    def test_initialization(self):
        """Test scorer initialization with default parameters."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
            "METRIC_TRACK_LOSS": True,
            "METRIC_HIGH_UNCERTAINTY": False,
        }

        scorer = FrameQualityScorer(params)

        assert scorer.max_targets == 4
        assert scorer.conf_threshold == 0.5
        assert scorer.use_confidence is True
        assert scorer.use_count_mismatch is True

    def test_score_frame_perfect_detections(self):
        """Test scoring of frame with perfect detections."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.9, 0.85, 0.88, 0.92],
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Perfect frame should have low score
        assert score == 0.0

    def test_score_frame_low_confidence(self):
        """Test scoring when detections have low confidence."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": False,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.3, 0.4, 0.5, 0.6],  # All below threshold
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Low confidence should increase score
        assert score > 0

    def test_score_frame_count_mismatch_under(self):
        """Test scoring when detection count is below expected."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.9],
            "count": 2,  # Only 2 instead of 4
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Under-detection should significantly increase score
        assert score > 0

    def test_score_frame_count_mismatch_over(self):
        """Test scoring when detection count is above expected."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.9, 0.85, 0.88, 0.82, 0.87],
            "count": 6,  # 6 instead of 4
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Over-detection should increase score (but less than under-detection)
        assert score > 0

    def test_score_frame_high_assignment_cost(self):
        """Test scoring when tracking assignment costs are high."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": False,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
        }

        scorer = FrameQualityScorer(params)

        tracking_data = {
            "assignment_costs": [80, 90, 75, 85],  # High costs
        }

        score = scorer.score_frame(frame_id=0, tracking_data=tracking_data)

        # High costs should increase score
        assert score > 0

    def test_score_frame_combined_metrics(self):
        """Test scoring with multiple problematic metrics."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.4, 0.5, 0.3],  # Low confidence
            "count": 3,  # Count mismatch
        }

        tracking_data = {
            "assignment_costs": [60, 70, 55],  # High costs
        }

        score = scorer.score_frame(
            frame_id=0, detection_data=detection_data, tracking_data=tracking_data
        )

        # Multiple issues should compound the score
        assert score > 0.3

    def test_score_frame_nan_confidences_ignored(self):
        """Test that NaN confidences (from background subtraction) are handled."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [np.nan, np.nan, np.nan, np.nan],
            "count": 4,
        }

        # Should not crash and should handle gracefully
        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        assert isinstance(score, (int, float))

    def test_score_frame_mixed_confidences(self):
        """Test scoring with mix of valid and NaN confidences."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.3, np.nan, 0.6],  # Mixed
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Should score based on valid confidences only
        assert score > 0  # 0.3 is below threshold

    def test_score_frame_no_data(self):
        """Test scoring with no detection or tracking data."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        score = scorer.score_frame(frame_id=0)

        # Should return 0 or low score when no data
        assert score == 0.0

    def test_score_frame_empty_detection_data(self):
        """Test scoring with empty detection data dict."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        score = scorer.score_frame(frame_id=0, detection_data={})

        assert score == 0.0

    def test_metrics_can_be_disabled(self):
        """Test that individual metrics can be disabled."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": False,  # Disabled
            "METRIC_COUNT_MISMATCH": False,  # Disabled
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.3, 0.4],  # Low confidence (but metric disabled)
            "count": 2,  # Count mismatch (but metric disabled)
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Should be 0 since all metrics are disabled
        assert score == 0.0

    def test_score_frame_zero_count(self):
        """Test scoring when no objects detected."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [],
            "count": 0,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Zero detections should be highly problematic
        assert score > 0.2

    def test_score_normalization(self):
        """Test that scores are in reasonable range."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
        }

        scorer = FrameQualityScorer(params)

        # Worst case scenario
        detection_data = {
            "confidences": [0.1, 0.2],
            "count": 2,
        }

        tracking_data = {
            "assignment_costs": [100, 120],
        }

        score = scorer.score_frame(
            frame_id=0, detection_data=detection_data, tracking_data=tracking_data
        )

        # Score should be finite and positive
        assert 0 <= score <= 5  # Allow some headroom for compounded scores

    def test_multiple_frames_independent(self):
        """Test that scoring multiple frames maintains independence."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        # Score frame 0
        score1 = scorer.score_frame(
            frame_id=0, detection_data={"confidences": [0.3, 0.4, 0.5, 0.6], "count": 4}
        )

        # Score frame 1 with different data
        score2 = scorer.score_frame(
            frame_id=1,
            detection_data={"confidences": [0.9, 0.8, 0.85, 0.87], "count": 4},
        )

        # Different inputs should produce different scores
        assert score1 != score2
        assert score1 > score2  # Frame 0 has lower confidence

    def test_empty_confidences_list(self):
        """Test handling of empty confidence list."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [],
            "count": 0,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Should handle gracefully
        assert isinstance(score, (int, float))

    def test_low_confidence_uses_frame_average_not_minimum(self):
        """Low-confidence score should use average confidence across detections."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": False,
        }

        scorer = FrameQualityScorer(params)

        # One low outlier but high frame-average confidence.
        detection_data = {
            "confidences": [0.1, 0.9, 0.9, 0.9],
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Average is 0.7 (>0.5), so low-confidence metric should not trigger.
        assert score == 0.0
