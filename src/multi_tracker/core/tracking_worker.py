"""
Core tracking engine running in separate thread for real-time performance.

This module contains the TrackingWorker class which implements the main
multi-object tracking pipeline using background subtraction, Kalman filtering,
and Hungarian algorithm for optimal track assignment.
"""

import sys, time, gc, math, logging, os, random
import numpy as np
import cv2
from collections import deque
from scipy.optimize import linear_sum_assignment
from PySide2.QtCore import QThread, Signal, QMutex

from ..utils.image_processing import apply_image_adjustments, stabilize_lighting
from ..utils.csv_writer import CSVWriterThread
from ..utils.video_io import create_reversed_video
from ..utils.geometry import wrap_angle_degs


logger = logging.getLogger(__name__)

class TrackingWorker(QThread):
    """
    Core tracking engine running in separate thread for real-time performance.
    
    This class implements a sophisticated multi-object tracking system combining:
    1. Background subtraction using "lightest pixel" method
    2. Kalman filtering for motion prediction and state estimation  
    3. Hungarian algorithm for optimal detection-to-track assignment
    4. State management for handling occlusions and track lifecycle
    
    The tracking pipeline processes each frame through these stages:
    - Preprocessing: Resize, color conversion, brightness/contrast adjustment
    - Background modeling: Maintain lightest-pixel background model
    - Foreground detection: Threshold difference image with morphological operations
    - Object detection: Find contours, fit ellipses, extract measurements
    - Prediction: Use Kalman filters to predict next positions/orientations
    - Assignment: Solve bipartite matching between predictions and detections
    - Update: Correct Kalman filters and update tracking states
    - Visualization: Draw trajectories and tracking information
    
    Signals:
        frame_signal (np.ndarray): Emits processed frame for display
        finished_signal (bool, list, list): Emits completion status, FPS data, and trajectories
    """
    
    # Qt signals for communication with main thread
    frame_signal = Signal(np.ndarray)      # Processed frame for display
    finished_signal = Signal(bool, list, list)  # (success, fps_list, trajectories)
    progress_signal = Signal(int, str)     # (percentage, status_text) for progress updates
    histogram_data_signal = Signal(dict)   # (histogram_data) for real-time parameter visualization
    
    def __init__(self, video_path, csv_writer_thread=None, video_output_path=None, backward_mode=False, parent=None):
        """
        Initialize tracking worker thread.
        
        Args:
            video_path (str): Path to input video file
            csv_writer_thread (CSVWriterThread, optional): CSV logger for data export
            video_output_path (str, optional): Path for output video with tracking overlays
            backward_mode (bool): If True, process video from end to start (backward tracking)
            parent (QObject, optional): Parent Qt object
        """
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.video_output_path = video_output_path
        self.video_writer = None  # OpenCV VideoWriter for output video
        self.backward_mode = backward_mode  # New: enable backward tracking
        
        # Thread-safe parameter access
        self.params_mutex = QMutex()
        self.parameters = {}
        self._stop_requested = False

        # Per-track state arrays (initialized based on MAX_TARGETS parameter)
        self.kalman_filters = []        # Kalman filter objects for motion prediction
        self.trajectories_pruned = []   # Recent trajectory points for visualization
        self.trajectories_full = []     # Complete trajectory history  
        self.position_deques = []       # Recent positions for velocity calculation
        self.orientation_last = []      # Previous orientation for smoothing
        self.last_shape_info = []       # Previous area/aspect ratio for cost calculation
        self.track_states = []          # Current state: 'active', 'occluded', 'lost'
        self.missed_frames = []         # Consecutive frames without detection
        self.tracking_continuity = []   # Consecutive frames successfully tracking
        
        # Trajectory ID system for persistent identity across track losses
        self.trajectory_ids = []        # Current trajectory ID for each track slot
        self.next_trajectory_id = 0     # Counter for assigning new trajectory IDs

        # Background subtraction and detection state
        self.background_model_lightest = None  # Lightest-pixel background model
        self.detection_initialized = False     # Whether detection system is stable
        self.tracking_stabilized = False       # Whether tracking system is stable
        self.detection_counts = 0              # Consecutive frames with correct detections
        self.tracking_counts = 0               # Consecutive frames with good tracking

        # Lighting stabilization components
        self.reference_intensity = None        # Global intensity reference from background
        self.intensity_history = deque(maxlen=50)  # Rolling history of frame intensities
        self.lighting_smooth_alpha = 0.95      # Smoothing factor for lighting adaptation
        self.adaptive_background = None        # Smoothly updated background model
        self.lighting_state = {}               # Dictionary to store smoothing state

        # Performance monitoring
        self.fps_list = []        # Frame processing rates over time
        self.frame_count = 0      # Current frame number
        self.start_time = 0       # Processing start timestamp

    def set_parameters(self, p: dict):
        """
        Thread-safe parameter update.
        
        Args:
            p (dict): Dictionary of tracking parameters
        """
        self.params_mutex.lock()
        self.parameters = p
        self.params_mutex.unlock()

    def get_current_params(self):
        """
        Thread-safe parameter retrieval.
        
        Returns:
            dict: Copy of current parameters
        """
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def stop(self):
        """Signal the tracking thread to stop processing."""
        self._stop_requested = True

    def init_kalman_filters(self, p):
        """
        Initialize Kalman filters for motion prediction and state estimation.
        
        Each filter tracks a 5-dimensional state vector:
        [x, y, theta, vx, vy] where:
        - x, y: Position coordinates (pixels)
        - theta: Orientation angle (radians) 
        - vx, vy: Velocity components (pixels/frame)
        
        The measurement vector is 3-dimensional: [x, y, theta]
        
        Args:
            p (dict): Parameters containing MAX_TARGETS and noise covariances
            
        Returns:
            list: List of initialized cv2.KalmanFilter objects
        """
        kfs = []
        for _ in range(p["MAX_TARGETS"]):
            # Create 5-state, 3-measurement Kalman filter
            kf = cv2.KalmanFilter(5, 3)
            
            # Measurement matrix: maps state [x,y,theta,vx,vy] to observation [x,y,theta]
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0, 0],  # x position
                [0, 1, 0, 0, 0],  # y position  
                [0, 0, 1, 0, 0]   # orientation
            ], np.float32)
            
            # Transition matrix: constant velocity model
            kf.transitionMatrix = np.array([
                [1, 0, 0, 1, 0],  # x(t+1) = x(t) + vx(t)
                [0, 1, 0, 0, 1],  # y(t+1) = y(t) + vy(t)
                [0, 0, 1, 0, 0],  # theta(t+1) = theta(t) (constant orientation)
                [0, 0, 0, 1, 0],  # vx(t+1) = vx(t) (constant velocity)
                [0, 0, 0, 0, 1]   # vy(t+1) = vy(t) (constant velocity)
            ], np.float32)
            
            # Process noise: models uncertainty in motion model
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
            
            # Measurement noise: models uncertainty in observations
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            
            # Initial error covariance
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            kfs.append(kf)
        return kfs

    def prime_lightest_background(self, cap, p):
        """
        Initialize background model using "lightest pixel" method with lighting reference.
        
        This method samples random frames from the video and builds a background
        model by taking the maximum intensity at each pixel across all samples.
        This approach works well when objects are darker than the background
        (e.g., dark flies on light surfaces).
        
        Additionally establishes reference intensity for lighting stabilization.
        
        Args:
            cap (cv2.VideoCapture): Video capture object
            p (dict): Parameters including BACKGROUND_PRIME_FRAMES, image adjustments
        """
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or p["BACKGROUND_PRIME_FRAMES"] < 1:
            return
            
        # Limit sampling to available frames
        count = min(p["BACKGROUND_PRIME_FRAMES"], total)
        br, ct, gm = p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"]
        ROI_mask = p.get("ROI_MASK", None)  # Get ROI mask for background priming
        resize_f = p.get("RESIZE_FACTOR", 1.0)
        
        # Sample random frame indices to avoid temporal bias
        idxs = random.sample(range(total), count)
        bg_temp = None
        intensity_samples = []
        
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: 
                continue
                
            # Apply resize early if needed
            if resize_f < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA)
                
            # Convert to grayscale and apply adjustments
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, br, ct, gm)
            
            # Apply ROI mask EARLY - before background modeling
            if ROI_mask is not None:
                # Resize ROI mask to match current frame dimensions
                if resize_f != 1.0:
                    roi_resized = cv2.resize(ROI_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    roi_resized = ROI_mask
                # Zero out everything outside ROI
                gray = cv2.bitwise_and(gray, gray, mask=roi_resized)
            
            # Collect intensity statistics for reference (robust mean excluding outliers)
            # Only from ROI area if mask is applied
            if ROI_mask is not None:
                # Calculate statistics only from ROI pixels
                roi_pixels = gray[roi_resized > 0]
                if len(roi_pixels) > 100:  # Ensure enough pixels
                    p25, p75 = np.percentile(roi_pixels, [25, 75])
                    mask = (roi_pixels >= p25) & (roi_pixels <= p75)
                    if np.sum(mask) > 0:
                        intensity_samples.append(np.mean(roi_pixels[mask]))
            else:
                # Use entire frame if no ROI
                frame_flat = gray.flatten()
                p25, p75 = np.percentile(frame_flat, [25, 75])
                mask = (frame_flat >= p25) & (frame_flat <= p75)
                if np.sum(mask) > 0:
                    intensity_samples.append(np.mean(frame_flat[mask]))
            
            # Build maximum intensity background model
            if bg_temp is None:
                bg_temp = gray.astype(np.float32)
            else:
                bg_temp = np.maximum(bg_temp, gray.astype(np.float32))
        
        if bg_temp is not None:
            self.background_model_lightest = bg_temp
            self.adaptive_background = bg_temp.copy()  # Initialize adaptive background
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Establish reference intensity from sampled frames
            if intensity_samples:
                self.reference_intensity = np.median(intensity_samples)
                logger.info(f"Reference intensity established: {self.reference_intensity:.1f}")
            else:
                # Fallback: use background model statistics
                if ROI_mask is not None:
                    roi_resized = cv2.resize(ROI_mask, (bg_temp.shape[1], bg_temp.shape[0]), interpolation=cv2.INTER_NEAREST) if resize_f != 1.0 else ROI_mask
                    roi_bg_pixels = bg_temp[roi_resized > 0]
                    self.reference_intensity = np.mean(roi_bg_pixels) if len(roi_bg_pixels) > 0 else np.mean(bg_temp)
                else:
                    self.reference_intensity = np.mean(bg_temp)
                logger.info(f"Fallback reference intensity: {self.reference_intensity:.1f}")

    def _local_conservative_split(self, sub, p):
        """
        Apply conservative morphological operations to split merged objects.
        
        When multiple objects are detected as a single large contour (due to
        proximity or lighting), this method attempts to split them using
        erosion followed by opening operations.
        
        Args:
            sub (np.ndarray): Binary mask region to process
            p (dict): Parameters with CONSERVATIVE_KERNEL_SIZE and CONSERVATIVE_ERODE_ITER
            
        Returns:
            np.ndarray: Processed binary mask with potential object separation
        """
        k = p["CONSERVATIVE_KERNEL_SIZE"]
        it = p["CONSERVATIVE_ERODE_ITER"]
        
        # Create elliptical structuring element for natural object shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        
        # Erode to separate touching objects
        out = cv2.erode(sub, kernel, iterations=it)
        
        # Open to remove noise and smooth boundaries
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    def post_process_trajectories(self, trajectories_full, params):
        """
        Post-process trajectories to clean up tracking artifacts and optimize trajectory quality.
        
        This function performs several trajectory improvements:
        1. Filter out very short trajectories (likely noise)
        2. Detect frame gaps and velocity/distance breaks
        3. Break trajectories at impossible velocity jumps or large distance gaps
        4. Re-sort trajectory IDs for cleaner output
        5. Provide trajectory quality statistics
        
        Args:
            trajectories_full (list): List of trajectory lists for each track slot
            params (dict): Parameters including post-processing thresholds
            
        Returns:
            tuple: (cleaned_trajectories, trajectory_stats)
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available, skipping trajectory post-processing")
            return trajectories_full, {}
        
        # Extract post-processing parameters with improved defaults from your example
        min_trajectory_length = params.get("MIN_TRAJECTORY_LENGTH", 10)  # Minimum frames to keep
        max_velocity_break = params.get("MAX_VELOCITY_BREAK", 10.0)  # px/frame to break trajectories (improved default)
        max_distance_break = params.get("MAX_DISTANCE_BREAK", 30.0)  # px to break trajectories (improved default)
        
        logger.info("Starting trajectory post-processing...")
        
        # Convert trajectories to pandas DataFrame matching CSV structure
        all_trajectory_data = []
        for track_id, trajectory in enumerate(trajectories_full):
            for idx, point in enumerate(trajectory):
                x, y, theta, frame = point
                all_trajectory_data.append({
                    'TrackID': track_id,
                    'TrajectoryID': track_id,  # Initially same as track ID
                    'Index': idx,
                    'X': int(x),
                    'Y': int(y),
                    'Theta': theta,
                    'FrameID': int(frame),
                    'State': 'active'  # Assume all points from trajectories_full are active
                })
        
        if not all_trajectory_data:
            logger.warning("No trajectory data to post-process")
            return trajectories_full, {}
        
        df = pd.DataFrame(all_trajectory_data)
        
        # Initialize trajectory statistics
        trajectory_stats = {
            'original_count': len([t for t in trajectories_full if len(t) > 0]),
            'removed_short': 0,
            'broken_velocity': 0,
            'broken_distance': 0,
            'broken_frame_gaps': 0,
            'final_count': 0,
            'avg_length': 0
        }
        
        # Process trajectories following the improved working example approach
        cleaned_trajectories_list = []
        
        for trajectory_id in df['TrajectoryID'].unique():
            traj = df[df['TrajectoryID'] == trajectory_id]
            
            # Skip short trajectories first (before processing)
            if len(traj) < min_trajectory_length:
                trajectory_stats['removed_short'] += 1
                logger.debug(f"Skipping trajectory {trajectory_id} due to insufficient length: {len(traj)} frames")
                continue
            
            # Keep only active frames (following the improved example)
            traj_active = traj[traj['State'] == 'active']
            
            if len(traj_active) < min_trajectory_length:
                trajectory_stats['removed_short'] += 1
                logger.debug(f"Skipping trajectory {trajectory_id} due to insufficient active length: {len(traj_active)} frames")
                continue
            
            # Check for breaks in the trajectory using improved logic
            breaks = np.where(np.diff(traj_active['FrameID']) > 1)[0]
            break_start_end = []
            last_valid_frame = traj_active.iloc[0]['FrameID']
            
            if len(breaks) > 0:
                for break_index in breaks:
                    # Get velocity and distance across the break
                    start_frame = traj_active.iloc[break_index]['FrameID']
                    end_frame = traj_active.iloc[break_index + 1]['FrameID']
                    start_pos = traj_active.iloc[break_index][['X', 'Y']].values
                    end_pos = traj_active.iloc[break_index + 1][['X', 'Y']].values
                    velocity = np.linalg.norm(end_pos - start_pos) / (end_frame - start_frame)
                    
                    # Check velocity break condition
                    if velocity > max_velocity_break:
                        logger.debug(f"Breaking trajectory {trajectory_id} at frame {start_frame} due to high velocity: {velocity:.2f} px/frame")
                        break_start_end.append((last_valid_frame, start_frame))
                        last_valid_frame = end_frame
                        trajectory_stats['broken_velocity'] += 1
                        continue
                    
                    # Check distance break condition
                    distance = np.linalg.norm(end_pos - start_pos)
                    if distance > max_distance_break:
                        logger.debug(f"Breaking trajectory {trajectory_id} at frame {start_frame} due to high distance: {distance:.2f} px")
                        break_start_end.append((last_valid_frame, start_frame))
                        last_valid_frame = end_frame
                        trajectory_stats['broken_distance'] += 1
                        continue
                    
                    # If we get here, it's just a frame gap but not a velocity/distance break
                    trajectory_stats['broken_frame_gaps'] += 1
            
            # Add the last segment
            if last_valid_frame <= traj_active['FrameID'].max():
                break_start_end.append((last_valid_frame, traj_active['FrameID'].max()))
            
            # Create segments for the trajectory using improved approach
            segments = []
            for start, end in break_start_end:
                segment = traj_active[(traj_active['FrameID'] >= start) & (traj_active['FrameID'] <= end)]
                if len(segment) > min_trajectory_length:
                    segments.append(segment)
                    logger.debug(f"Created trajectory segment: {len(segment)} points "
                               f"(frames {segment.iloc[0]['FrameID']}-{segment.iloc[-1]['FrameID']})")
            
            # Add the segments to the trajectories list
            cleaned_trajectories_list.extend(segments)
            trajectory_stats['final_count'] += len(segments)
        
        # Convert cleaned trajectories back to original format
        cleaned_trajectories = []
        for i, traj_df in enumerate(cleaned_trajectories_list):
            traj_list = []
            for _, row in traj_df.iterrows():
                traj_list.append((int(row['X']), int(row['Y']), row['Theta'], int(row['FrameID'])))
            cleaned_trajectories.append(traj_list)
        
        # Pad with empty trajectories to maintain list structure if needed
        while len(cleaned_trajectories) < len(trajectories_full):
            cleaned_trajectories.append([])
        
        # Calculate average trajectory length
        valid_lengths = [len(t) for t in cleaned_trajectories if len(t) > 0]
        trajectory_stats['avg_length'] = np.mean(valid_lengths) if valid_lengths else 0
        
        # Log summary statistics (keeping your nice reporting stats!)
        logger.info(f"Trajectory post-processing complete:")
        logger.info(f"  Original trajectories: {trajectory_stats['original_count']}")
        logger.info(f"  Removed (too short): {trajectory_stats['removed_short']}")
        logger.info(f"  Broken due to velocity: {trajectory_stats['broken_velocity']}")
        logger.info(f"  Broken due to distance: {trajectory_stats['broken_distance']}")
        logger.info(f"  Broken due to frame gaps: {trajectory_stats['broken_frame_gaps']}")
        logger.info(f"  Final trajectories: {trajectory_stats['final_count']}")
        logger.info(f"  Average length: {trajectory_stats['avg_length']:.1f} frames")
        
        return cleaned_trajectories, trajectory_stats

    def write_cleaned_trajectories_csv(self, cleaned_trajectories, output_path):
        """
        Write cleaned trajectories to a new CSV file with proper trajectory IDs.
        
        Args:
            cleaned_trajectories (list): List of cleaned trajectory lists
            output_path (str): Path for the cleaned CSV file
        """
        try:
            import csv
            
            header = ["TrackID", "TrajectoryID", "Index", "X", "Y", "Theta", "FrameID", "State"]
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
                for traj_id, trajectory in enumerate(cleaned_trajectories):
                    if not trajectory:  # Skip empty trajectories
                        continue
                        
                    for index, (x, y, theta, frame) in enumerate(trajectory):
                        writer.writerow([
                            traj_id,      # TrackID (cleaned, re-sorted)
                            traj_id,      # TrajectoryID (same as TrackID for cleaned data)
                            index,        # Index within this trajectory
                            x, y, theta,  # Position and orientation
                            frame,        # Frame number
                            'active'      # State (all cleaned trajectories are active)
                        ])
                        
            logger.info(f"Cleaned trajectories written to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write cleaned trajectories CSV: {e}")

    def _forward_frame_iterator(self, cap):
        """
        Generator to yield frames in forward order (normal processing).
        
        Args:
            cap (cv2.VideoCapture): Video capture object
            
        Yields:
            tuple: (frame, frame_number) in forward order
        """
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            yield frame, frame_num

    def run(self):
        """
        Main tracking loop executing the complete multi-object tracking pipeline.
        
        This method implements the core tracking algorithm with the following stages:
        
        1. **Initialization**: Set up video capture, background model, and tracking state
        2. **Frame Processing Loop**:
           - Load and preprocess frame (resize, adjust brightness/contrast/gamma)
           - Apply ROI mask if defined
           - Update background model using maximum intensity projection
           - Generate foreground mask using adaptive thresholding
           - Apply morphological operations to clean up mask
           - Detect objects by finding contours and fitting ellipses
           - Track objects using Kalman prediction + Hungarian assignment
           - Update tracking states and handle occlusions
           - Export data to CSV and visualize results
        3. **Cleanup**: Release resources and emit final results
        
        The tracking system handles multiple challenging scenarios:
        - Object occlusion and re-identification
        - Variable lighting conditions
        - Object merging and splitting
        - Orientation tracking with anti-flip logic
        - Track spawning and termination
        """
        # Memory cleanup and initialization
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()
        LOST_T = p["LOST_THRESHOLD_FRAMES"]  # Frames before declaring track lost

        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return

        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None  # Unknown length video (e.g., live streams)

        # Initialize video writer for output if specified
        if self.video_output_path:
            # Get video properties from input
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Apply resize factor if specified
            resize_f = p.get("RESIZE_FACTOR", 1.0)
            if resize_f < 1.0:
                width = int(width * resize_f)
                height = int(height * resize_f)
            
            # Modify output filename for backward tracking
            if self.backward_mode:
                base, ext = os.path.splitext(self.video_output_path)
                self.video_output_path = f"{base}_backward{ext}"
            
            # Initialize video writer with MP4 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_output_path, fourcc, fps, (width, height)
            )
            
            if not self.video_writer.isOpened():
                logger.warning(f"Could not initialize video writer for {self.video_output_path}")
                self.video_writer = None
            else:
                mode_str = "backward" if self.backward_mode else "forward"
                logger.info(f"Video output initialized ({mode_str}): {self.video_output_path} ({width}x{height} @ {fps:.1f}fps)")

        # Initialize background model and Kalman filters
        self.prime_lightest_background(cap, p)
        self.kalman_filters = self.init_kalman_filters(p)

        # Initialize per-track data structures
        N = p["MAX_TARGETS"]
        self.track_states = ['active'] * N               # Track lifecycle states
        self.missed_frames = [0] * N                     # Consecutive detection misses
        self.trajectories_pruned = [[] for _ in range(N)]  # Recent trajectory for display
        self.trajectories_full = [[] for _ in range(N)]    # Complete trajectory history
        self.position_deques = [deque(maxlen=2) for _ in range(N)]  # Recent positions for velocity
        self.orientation_last = [None] * N               # Previous orientation for smoothing
        self.last_shape_info = [None] * N                # Previous area/aspect for cost function
        self.tracking_continuity = [0] * N               # Consecutive frames successfully tracking
        
        # Initialize trajectory ID system - each track slot starts with a unique trajectory ID
        # When tracks are lost and reassigned, they get new trajectory IDs to maintain distinct identity
        self.trajectory_ids = list(range(N))             # Assign initial trajectory IDs 0, 1, 2, ..., N-1
        self.next_trajectory_id = N                      # Next available trajectory ID for reassignments

        # Performance monitoring
        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()
        local_counts = [0] * N  # Per-track detection counters for CSV indexing

        # Extract frequently used parameters to avoid repeated dictionary lookups
        vT = p["VELOCITY_THRESHOLD"]          # Speed threshold for orientation logic
        inst_flip = p["INSTANT_FLIP_ORIENTATION"]  # Enable velocity-based orientation correction
        Wp, Wo, Wa, Wasp = p["W_POSITION"], p["W_ORIENTATION"], p["W_AREA"], p["W_ASPECT"]  # Cost weights
        use_maha = p["USE_MAHALANOBIS"]       # Use Mahalanobis distance vs Euclidean
        ROI_mask = p.get("ROI_MASK", None)    # Optional circular ROI mask
        resize_f = p["RESIZE_FACTOR"]         # Video resize factor for performance
        
        # Set lighting stabilization parameters
        self.lighting_smooth_alpha = p.get("LIGHTING_SMOOTH_FACTOR", 0.95)

        # Setup frame iteration - both forward and backward use normal iteration
        # For backward mode, the video file itself is already reversed by FFmpeg
        frame_iterator = self._forward_frame_iterator(cap)
        mode_str = "backward" if self.backward_mode else "forward"
            
        logger.info(f"Starting {mode_str} tracking pass")

        # Main frame processing loop
        for frame, logical_frame_num in frame_iterator:
            if self._stop_requested:
                break
                
            self.frame_count += 1
            params = self.get_current_params()  # Get latest parameters (may change during runtime)

            # === PROGRESS UPDATES ===
            # Emit progress updates every 10 frames to avoid UI flooding
            if self.frame_count % 10 == 0 or self.frame_count == 1:
                if total_frames and total_frames > 0:
                    progress_pct = min(100, int((self.frame_count / total_frames) * 100))
                    status_text = f"{mode_str.capitalize()} tracking: frame {self.frame_count}/{total_frames} ({progress_pct}%)"
                else:
                    progress_pct = 0  # Unknown total frames
                    status_text = f"Processing frame {self.frame_count}"
                self.progress_signal.emit(progress_pct, status_text)

            # === FRAME PREPROCESSING ===
            # Resize frame for performance if specified
            if resize_f < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA)

            # Convert to grayscale and apply image adjustments
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, params["BRIGHTNESS"], params["CONTRAST"], params["GAMMA"])

            # === EARLY ROI APPLICATION ===
            # Apply circular ROI mask IMMEDIATELY after preprocessing to exclude all outside regions
            if ROI_mask is not None:
                # Resize ROI mask to match current frame dimensions
                if resize_f != 1.0:
                    roi_resized = cv2.resize(ROI_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    roi_resized = ROI_mask
                # Zero out everything outside ROI - this ensures no processing on outside areas
                gray = cv2.bitwise_and(gray, gray, mask=roi_resized)
                # Also store resized ROI for later use
                ROI_mask_current = roi_resized
            else:
                ROI_mask_current = None

            # === LIGHTING STABILIZATION ===
            # Apply lighting stabilization to compensate for changing illumination
            # This now works only on ROI pixels if ROI is defined
            if params.get("ENABLE_LIGHTING_STABILIZATION", True):
                median_window = params.get("LIGHTING_MEDIAN_WINDOW", 5)
                gray, self.intensity_history, current_intensity = stabilize_lighting(
                    gray, self.reference_intensity, self.intensity_history, 
                    self.lighting_smooth_alpha, ROI_mask_current, median_window, self.lighting_state
                )

            # === BACKGROUND MODEL UPDATE ===
            # Initialize background model with first frame
            if self.background_model_lightest is None:
                self.background_model_lightest = gray.astype(np.float32)
                self.adaptive_background = gray.astype(np.float32)
                self.emit_frame(frame)
                continue
                
            # Update background using maximum intensity (lightest pixel method)
            # Only update within ROI if mask is defined
            if ROI_mask_current is not None:
                # Update only ROI regions, preserve background outside ROI
                roi_mask_bool = ROI_mask_current > 0
                self.background_model_lightest[roi_mask_bool] = np.maximum(
                    self.background_model_lightest[roi_mask_bool], 
                    gray.astype(np.float32)[roi_mask_bool]
                )
            else:
                # Update entire background if no ROI
                self.background_model_lightest = np.maximum(self.background_model_lightest, gray.astype(np.float32))
            
            # Smooth adaptive background update for lighting changes
            if params.get("ENABLE_ADAPTIVE_BACKGROUND", True) and self.adaptive_background is not None:
                learning_rate = params.get("BACKGROUND_LEARNING_RATE", 0.001)
                if ROI_mask_current is not None:
                    # Update only ROI regions for adaptive background
                    roi_mask_bool = ROI_mask_current > 0
                    self.adaptive_background[roi_mask_bool] = (
                        (1 - learning_rate) * self.adaptive_background[roi_mask_bool] + 
                        learning_rate * gray.astype(np.float32)[roi_mask_bool]
                    )
                else:
                    # Update entire adaptive background if no ROI
                    self.adaptive_background = (1 - learning_rate) * self.adaptive_background + learning_rate * gray.astype(np.float32)
                
                # Use adaptive background for subtraction if tracking is stabilized
                if self.tracking_stabilized:
                    bg_u8 = cv2.convertScaleAbs(self.adaptive_background)
                else:
                    bg_u8 = cv2.convertScaleAbs(self.background_model_lightest)
            else:
                bg_u8 = cv2.convertScaleAbs(self.background_model_lightest)

            # === FOREGROUND DETECTION ===
            # Generate difference image between current frame and background
            # Use directional subtraction based on animal/background contrast
            dark_on_light = params.get("DARK_ON_LIGHT_BACKGROUND", True)
            
            if dark_on_light:
                # Dark animals on light background: background - current_frame
                # This highlights areas where the current frame is darker than background
                diff = cv2.subtract(bg_u8, gray)
            else:
                # Light animals on dark background: current_frame - background  
                # This highlights areas where the current frame is lighter than background
                diff = cv2.subtract(gray, bg_u8)
            
            # Apply binary thresholding to detect foreground objects
            _, fg_mask = cv2.threshold(diff, params["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

            # === MORPHOLOGICAL PROCESSING ===
            # Apply morphological operations to clean up foreground mask
            ksz = params["MORPH_KERNEL_SIZE"]
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, ker)   # Remove noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, ker)  # Fill holes
            
            # === ADDITIONAL DILATION FOR THIN ANIMALS ===
            # Apply additional dilation to connect split parts of thin animals
            if params.get("ENABLE_ADDITIONAL_DILATION", False):
                dil_ksz = params.get("DILATION_KERNEL_SIZE", 3)
                dil_iter = params.get("DILATION_ITERATIONS", 2)
                dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksz, dil_ksz))
                fg_mask = cv2.dilate(fg_mask, dil_kernel, iterations=dil_iter)

            # === CONSERVATIVE OBJECT SPLITTING ===
            # Attempt to split merged objects after detection system is initialized
            if self.detection_initialized and params.get("ENABLE_CONSERVATIVE_SPLIT", True):
                cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Identify suspicious contours (too large or wrong count)
                suspicious = [
                    cv2.boundingRect(c) for c in cnts 
                    if cv2.contourArea(c) > params["MERGE_AREA_THRESHOLD"]
                    or sum(1 for cc in cnts if cv2.contourArea(cc) > 0) < N
                ]
                # Apply conservative splitting to suspicious regions
                for bx, by, bw, bh in suspicious:
                    sub = fg_mask[by:by+bh, bx:bx+bw]
                    fg_mask[by:by+bh, bx:bx+bw] = self._local_conservative_split(sub, params)

            # === OBJECT DETECTION AND MEASUREMENT ===
            # Find contours in cleaned foreground mask
            cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # === FRAME QUALITY CHECK ===
            # If there are way too many contours, the frame is likely too noisy/corrupted
            # Better to skip entirely than try to process hundreds of noise blobs
            N = params["MAX_TARGETS"]
            max_contour_multiplier = params.get("MAX_CONTOUR_MULTIPLIER", 20)  # Default: 20x target count
            max_allowed_contours = N * max_contour_multiplier
            
            if len(cnts) > max_allowed_contours:
                logger.debug(f"Frame {self.frame_count}: Too many contours ({len(cnts)} > {max_allowed_contours}), treating as no detections")
                meas, sizes, shapes = [], [], []  # Treat as no detections
            else:
                meas, sizes, shapes = [], [], []
                
                for c in cnts:
                    area = cv2.contourArea(c)
                    # Filter out small noise and invalid contours
                    if area < params["MIN_CONTOUR_AREA"] or len(c) < 5: 
                        continue
                        
                    # Fit ellipse to contour for position and orientation
                    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
                    
                    # Ensure major axis is ax1, adjust angle accordingly
                    if ax1 < ax2:
                        ax1, ax2 = ax2, ax1
                        ang = (ang + 90) % 180
                        
                    # Convert angle to radians
                    rad = np.deg2rad(ang)
                    
                    # Store measurement: [x, y, orientation]
                    meas.append(np.array([cx, cy, rad], np.float32))
                    sizes.append(area)
                    # Store shape info: (ellipse_area, aspect_ratio)
                    shapes.append((np.pi * (ax1/2) * (ax2/2), ax1/ax2))            # === SIZE-BASED FILTERING ===
            # Filter detections by size range before sorting
            if meas and params.get("ENABLE_SIZE_FILTERING", False):
                min_size = params.get("MIN_OBJECT_SIZE", 0)
                max_size = params.get("MAX_OBJECT_SIZE", float('inf'))
                original_count = len(meas)
                
                # Filter out detections outside the acceptable size range
                filtered_meas, filtered_sizes, filtered_shapes = [], [], []
                for i, size in enumerate(sizes):
                    if min_size <= size <= max_size:
                        filtered_meas.append(meas[i])
                        filtered_sizes.append(sizes[i])
                        filtered_shapes.append(shapes[i])
                
                # Update lists with filtered results
                meas, sizes, shapes = filtered_meas, filtered_sizes, filtered_shapes
                
                # Debug message about filtering results
                if len(meas) != original_count:
                    logger.debug(f"Size filtering: {original_count} detections -> {len(meas)} detections (range: {min_size}-{max_size})")

            # === DETECTION FILTERING ===
            # Keep only the N largest detections to match expected target count
            if len(meas) > N:
                idxs = np.argsort(sizes)[::-1][:N]  # Sort by size, take largest N
                meas = [meas[i] for i in idxs]
                shapes = [shapes[i] for i in idxs]

            # === DETECTION INITIALIZATION LOGIC ===
            # Track consecutive frames with reasonable detection count for stability
            # Now we start tracking when we have ANY detections (not necessarily all N targets)
            min_detections_to_start = params.get("MIN_DETECTIONS_TO_START", 1)  # Minimum detections to begin tracking
            
            if len(meas) >= min_detections_to_start:
                self.detection_counts += 1
            else:
                self.detection_counts = 0
                
            # Initialize detection system when stable detection achieved
            # Reduce the required consecutive frames since we're more flexible now
            min_detection_frames = max(1, params["MIN_DETECTION_COUNTS"] // 2)  # Half the original requirement
            if self.detection_counts >= min_detection_frames:
                self.detection_initialized = True
                if self.detection_counts == min_detection_frames:  # Log only once when first initialized
                    logger.info(f"Tracking initialized with {len(meas)} detections (minimum required: {min_detections_to_start})")

            # Create copy of frame for visualization overlay
            overlay = frame.copy()

            # === TRACKING AND DATA ASSOCIATION ===
            if self.detection_initialized and meas:
                # === KALMAN FILTER INITIALIZATION (First Frame) ===
                # Initialize Kalman filter states - now more flexible for partial detections
                if self.frame_count == 1:
                    h, w = gray.shape
                    
                    # Initialize filters for detected objects with their actual positions
                    for i, measurement in enumerate(meas[:N]):  # Only initialize up to available measurements
                        if i < len(self.kalman_filters):
                            kf = self.kalman_filters[i]
                            # Initialize with actual detection position
                            init = np.array([measurement[0], measurement[1], measurement[2], 0, 0], np.float32)
                            kf.statePre = init.copy()
                            kf.statePost = init.copy()
                            # Clear history for new tracking session
                            self.position_deques[i].clear()
                            self.orientation_last[i] = None
                            self.last_shape_info[i] = None
                            self.tracking_continuity[i] = 0
                            # Set track as active since we have a detection
                            self.track_states[i] = 'active'
                            self.missed_frames[i] = 0
                    
                    # Initialize remaining filters with random positions (for future detections)
                    for i in range(len(meas), N):
                        if i < len(self.kalman_filters):
                            kf = self.kalman_filters[i]
                            # Random initial state for tracks without detections
                            init = np.array([random.randint(0, w), random.randint(0, h), 0, 0, 0], np.float32)
                            kf.statePre = init.copy()
                            kf.statePost = init.copy()
                            # Clear history for new tracking session
                            self.position_deques[i].clear()
                            self.orientation_last[i] = None
                            self.last_shape_info[i] = None
                            self.tracking_continuity[i] = 0
                            # Set track as lost since no detection available
                            self.track_states[i] = 'lost'
                            self.missed_frames[i] = 0
                    
                    logger.info(f"Initialized tracking: {len(meas)} active tracks, {N - len(meas)} lost tracks")

                # === PREDICTION STEP ===
                # Use Kalman filters to predict next positions/orientations
                preds = [kf.predict()[:3].flatten() for kf in self.kalman_filters]
                PREDS = np.array(preds, np.float32)  # Shape: (N, 3) [x, y, theta]

                # === COST MATRIX COMPUTATION ===
                # Build cost matrix for bipartite matching between tracks and detections
                M = len(meas)  # Number of current detections
                cost = np.zeros((N, M), np.float32)
                
                for i in range(N):  # For each track
                    for j in range(M):  # For each detection
                        # === POSITION COST ===
                        if use_maha:
                            # Use Mahalanobis distance considering prediction uncertainty
                            Pcov = self.kalman_filters[i].errorCovPre[:2, :2]  # Position covariance
                            diff = meas[j][:2] - PREDS[i][:2]
                            try: 
                                invP = np.linalg.inv(Pcov)
                                posc = np.sqrt(diff.T @ invP @ diff)
                            except: 
                                posc = np.linalg.norm(diff)  # Fallback to Euclidean
                        else:
                            # Simple Euclidean distance
                            posc = np.linalg.norm(meas[j][:2] - PREDS[i][:2])
                        
                        # === ORIENTATION COST ===
                        # Handle angular wraparound for orientation difference
                        odiff = abs(PREDS[i][2] - meas[j][2])
                        odiff = min(odiff, 2*np.pi - odiff)  # Minimum angular distance
                        
                        # === SHAPE CONSISTENCY COST ===
                        # Use previous shape info or current detection if no history
                        prev_a, prev_as = self.last_shape_info[i] if self.last_shape_info[i] else shapes[j]
                        area_diff = abs(shapes[j][0] - prev_a)      # Area difference
                        asp_diff = abs(shapes[j][1] - prev_as)      # Aspect ratio difference
                    
                        
                        # === COMBINED COST ===
                        # Weighted combination of all cost components
                        cost[i, j] = Wp * posc + Wo * odiff + Wa * area_diff + Wasp * asp_diff

                # === HYBRID ASSIGNMENT SYSTEM ===
                # Define continuity threshold for established vs new tracks
                CONTINUITY_THRESHOLD = params.get("CONTINUITY_THRESHOLD", 10)  # Frames of continuous tracking
                
                # Separate tracks into established (high continuity) and new/unstable (low continuity)
                established_tracks = [i for i in range(N) if self.tracking_continuity[i] >= CONTINUITY_THRESHOLD and self.track_states[i] != 'lost']
                unstable_tracks = [i for i in range(N) if self.tracking_continuity[i] < CONTINUITY_THRESHOLD and self.track_states[i] != 'lost']
                lost_tracks = [i for i in range(N) if self.track_states[i] == 'lost']
                
                logger.debug(f"Frame {self.frame_count}: Established={len(established_tracks)}, Unstable={len(unstable_tracks)}, Lost={len(lost_tracks)}")
                
                assigned_detections = set()
                assigned_tracks = set()
                all_assignments = []
                
                # === PHASE 1: HUNGARIAN FOR ESTABLISHED TRACKS ===
                if established_tracks and M > 0:
                    # Create sub-cost matrix for established tracks only
                    est_cost = np.zeros((len(established_tracks), M), np.float32)
                    for i, track_idx in enumerate(established_tracks):
                        for j in range(M):
                            est_cost[i, j] = cost[track_idx, j]
                    
                    # Apply Hungarian algorithm to established tracks
                    est_rows, est_cols = linear_sum_assignment(est_cost)
                    
                    # Process assignments for established tracks
                    for i, j in zip(est_rows, est_cols):
                        track_idx = established_tracks[i]
                        det_idx = j  # j is already the detection index from est_cols
                        
                        # Only accept assignment if cost is reasonable
                        if cost[track_idx, det_idx] < params["MAX_DISTANCE_THRESHOLD"]:
                            all_assignments.append((track_idx, det_idx))
                            assigned_detections.add(det_idx)
                            assigned_tracks.add(track_idx)
                            logger.debug(f"Hungarian assignment: Track {track_idx} (continuity={self.tracking_continuity[track_idx]}) -> Detection {det_idx} (cost={cost[track_idx, det_idx]:.2f})")
                        else:
                            logger.debug(f"Rejected Hungarian assignment: Track {track_idx} -> Detection {det_idx} (cost too high: {cost[track_idx, det_idx]:.2f})")
                
                # === PHASE 2: PRIORITY-BASED FOR UNSTABLE TRACKS ===
                # Sort unstable tracks by continuity (longest first, even if below threshold)
                unstable_tracks_sorted = sorted(unstable_tracks, key=lambda i: self.tracking_continuity[i], reverse=True)
                
                for track_idx in unstable_tracks_sorted:
                    best_detection = None
                    best_cost = float('inf')
                    
                    # Find best available detection for this track
                    for det_idx in range(M):
                        if det_idx in assigned_detections:
                            continue  # Detection already taken by established track
                        
                        track_cost = cost[track_idx, det_idx]
                        if track_cost < params["MAX_DISTANCE_THRESHOLD"] and track_cost < best_cost:
                            best_cost = track_cost
                            best_detection = det_idx
                    
                    # Assign best detection to this track if found
                    if best_detection is not None:
                        all_assignments.append((track_idx, best_detection))
                        assigned_detections.add(best_detection)
                        assigned_tracks.add(track_idx)
                        logger.debug(f"Priority assignment: Track {track_idx} (continuity={self.tracking_continuity[track_idx]}) -> Detection {best_detection} (cost={best_cost:.2f})")
                
                # === PHASE 3: RESPAWN LOST TRACKS WITH REMAINING DETECTIONS ===
                unassigned_detections = [i for i in range(M) if i not in assigned_detections]
                MIN_RESPAWN_DISTANCE = params.get("MIN_RESPAWN_DISTANCE", params["MAX_DISTANCE_THRESHOLD"] * 0.8)
                
                for det_idx in unassigned_detections:
                    if not lost_tracks:
                        break  # No more lost tracks available
                    
                    # Check if this detection is far enough from all assigned detections
                    detection_pos = meas[det_idx][:2]  # [x, y] position
                    is_good_detection = True
                    
                    # Calculate minimum distance to any assigned detection
                    min_distance_to_assigned = float('inf')
                    for assigned_det_idx in assigned_detections:
                        assigned_pos = meas[assigned_det_idx][:2]
                        distance = np.linalg.norm(detection_pos - assigned_pos)
                        min_distance_to_assigned = min(min_distance_to_assigned, distance)
                    
                    # Only respawn if detection is sufficiently far from assigned detections
                    if min_distance_to_assigned < MIN_RESPAWN_DISTANCE:
                        is_good_detection = False
                        logger.debug(f"Rejected respawn: Detection {det_idx} too close to assigned detection (distance={min_distance_to_assigned:.2f} < {MIN_RESPAWN_DISTANCE:.2f})")
                    
                    # Also check if detection is within reasonable bounds for any lost track
                    if is_good_detection:
                        best_lost_track = None
                        best_respawn_cost = float('inf')
                        
                        # Find the best lost track for this detection (if any)
                        for track_idx in lost_tracks:
                            # Use last known position if available, otherwise skip cost check
                            if self.kalman_filters[track_idx].statePost is not None:
                                last_known_pos = self.kalman_filters[track_idx].statePost[:2].flatten()
                                respawn_distance = np.linalg.norm(detection_pos - last_known_pos)
                                
                                # Only consider if within reasonable respawn distance
                                if respawn_distance < params["MAX_DISTANCE_THRESHOLD"] * 2.0:  # More lenient for respawning
                                    if respawn_distance < best_respawn_cost:
                                        best_respawn_cost = respawn_distance
                                        best_lost_track = track_idx
                            else:
                                # If no last known position, any lost track is eligible
                                best_lost_track = track_idx
                                break
                        
                        # Respawn the best lost track if found
                        if best_lost_track is not None:
                            all_assignments.append((best_lost_track, det_idx))
                            assigned_tracks.add(best_lost_track)
                            lost_tracks.remove(best_lost_track)
                            # Assign new trajectory ID for respawned track
                            self.trajectory_ids[best_lost_track] = self.next_trajectory_id
                            self.next_trajectory_id += 1
                            logger.debug(f"Respawn assignment: Track {best_lost_track} -> Detection {det_idx} (distance from assigned={min_distance_to_assigned:.2f}, respawn_cost={best_respawn_cost:.2f}) - New trajectory ID: {self.trajectory_ids[best_lost_track]}")
                        else:
                            logger.debug(f"No suitable lost track for detection {det_idx}")
                    
                    # If detection wasn't good enough for respawning, it remains unassigned
                    if not is_good_detection or best_lost_track is None:
                        logger.debug(f"Detection {det_idx} left unassigned")
                
                # Convert assignments back to the format expected by rest of code
                if all_assignments:
                    rows, cols = zip(*all_assignments)
                    rows, cols = list(rows), list(cols)
                else:
                    rows, cols = [], []

                # === TRACK STATE MANAGEMENT ===
                # Categorize tracks based on assignment results
                # Filter out assignments with costs that are too high
                valid_assignments = []
                high_cost_tracks = []
                
                for r, c in zip(rows, cols):
                    if cost[r, c] < params["MAX_DISTANCE_THRESHOLD"]:
                        valid_assignments.append((r, c))
                    else:
                        high_cost_tracks.append(r)
                        logger.debug(f"Track {r} assignment rejected due to high cost ({cost[r, c]:.2f})")
                
                # Update rows and cols to only include valid assignments
                if valid_assignments:
                    rows, cols = zip(*valid_assignments)
                    rows, cols = list(rows), list(cols)
                else:
                    rows, cols = [], []
                
                all_rows = set(range(N))
                matched = set(rows)
                # Include high-cost tracks as unmatched
                unmatched = list((all_rows - matched) | set(high_cost_tracks))
                all_dets = set(range(M))
                matched_dets = set(cols)
                free_dets = list(all_dets - matched_dets)   # Detections without tracks

                # Reset state for successfully matched tracks
                for r in matched:
                    self.missed_frames[r] = 0
                    self.track_states[r] = 'active'
                    
                # Handle unmatched tracks (missed detections)
                for r in unmatched:
                    self.missed_frames[r] += 1
                    # Transition: active -> occluded -> lost
                    if self.missed_frames[r] < LOST_T:
                        self.track_states[r] = 'occluded'
                        # Keep continuity during occlusion - don't reset it
                    else:
                        self.track_states[r] = 'lost'
                        # Only reset continuity when track is truly lost
                        self.tracking_continuity[r] = 0
                        logger.debug(f"Track {r} declared lost, resetting continuity")

                # === KALMAN FILTER CORRECTION AND STATE UPDATE ===
                avg_cost = 0
                for r, c in zip(rows, cols):
                    # All remaining assignments have reasonable costs (filtered above)
                    # === KALMAN CORRECTION ===
                    kf = self.kalman_filters[r]
                    kf.correct(meas[c].reshape(3, 1))  # Update filter with measurement
                    x, y, theta = meas[c]
                    
                    # === INCREMENT TRACKING CONTINUITY ===
                    # Track how long this track has been continuously following a target
                    self.tracking_continuity[r] += 1
                        
                    # === VELOCITY CALCULATION ===
                    # Track recent positions for speed estimation
                    self.position_deques[r].append((x, y))
                    speed = 0
                    if len(self.position_deques[r]) == 2:
                        (x1, y1), (x2, y2) = self.position_deques[r]
                        speed = math.hypot(x2 - x1, y2 - y1)  # Euclidean distance
                    
                    # === ORIENTATION SMOOTHING AND ANTI-FLIP LOGIC ===
                    final_theta = theta
                    old = self.orientation_last[r]
                    
                    # For slow-moving objects: smooth orientation changes
                    if speed < vT and old is not None:
                        old_deg, new_deg = math.degrees(old), math.degrees(theta)
                        delta = wrap_angle_degs(new_deg - old_deg)
                        
                        # Handle large orientation jumps (likely 180° flips)
                        if abs(delta) > 90:
                            new_deg = (new_deg + 180) % 360
                        elif abs(delta) > params["MAX_ORIENT_DELTA_STOPPED"]:
                            # Limit orientation change rate for stationary objects
                            new_deg = old_deg + math.copysign(params["MAX_ORIENT_DELTA_STOPPED"], delta)
                        final_theta = math.radians(new_deg)
                        
                    # For fast-moving objects: align orientation with motion direction
                    elif speed >= vT and inst_flip:
                        (x1, y1), (x2, y2) = self.position_deques[r]
                        # Calculate motion direction
                        ang = math.atan2(y2 - y1, x2 - x1)
                        # Check if orientation is opposite to motion (180° off)
                        diff = (ang - theta + math.pi) % (2 * math.pi) - math.pi
                        if abs(diff) > math.pi / 2:
                            final_theta = (theta + math.pi) % (2 * math.pi)
                    
                    # Update tracking state
                    self.orientation_last[r] = final_theta
                    self.last_shape_info[r] = shapes[c]
                    ts = self.frame_count  # Use frame count for consistent CSV output
                    
                    # === TRAJECTORY RECORDING ===
                    # Store complete trajectory for export
                    self.trajectories_full[r].append((int(x), int(y), final_theta, ts))
                    # Store recent trajectory for display
                    self.trajectories_pruned[r].append((int(x), int(y), final_theta, ts))
                    
                    # Prune old trajectory points to limit memory usage
                    self.trajectories_pruned[r] = [
                        pt for pt in self.trajectories_pruned[r]
                        if self.frame_count - pt[3] <= params["TRAJECTORY_HISTORY_SECONDS"]
                    ]
                    
                    # === CSV DATA EXPORT ===
                    if self.csv_writer_thread:
                        idx = local_counts[r]
                        traj_id = self.trajectory_ids[r]  # Use trajectory ID instead of track ID
                        self.csv_writer_thread.enqueue([
                            r, traj_id, idx, int(x), int(y), final_theta, ts, self.track_states[r]
                        ])
                        local_counts[r] += 1
                        
                    avg_cost += cost[r, c] / N

                # === REAL-TIME HISTOGRAM DATA COLLECTION ===
                # Collect histogram data every few frames to avoid overwhelming the GUI
                if params.get("enable_histograms", False) and self.frame_count % 3 == 0:
                    histogram_data = {
                        'velocities': [],
                        'sizes': [],  
                        'orientations': [],
                        'assignment_costs': []
                    }
                    
                    # Collect velocity data from successful assignments
                    for r, c in zip(rows, cols):
                        if len(self.position_deques[r]) == 2:
                            (x1, y1), (x2, y2) = self.position_deques[r]
                            velocity = math.hypot(x2 - x1, y2 - y1)
                            histogram_data['velocities'].append(velocity)
                        
                        # Collect orientation data
                        histogram_data['orientations'].append(meas[c][2])
                        
                        # Collect assignment cost data
                        histogram_data['assignment_costs'].append(cost[r, c])
                    
                    # Collect size data from all detections
                    histogram_data['sizes'].extend(sizes)
                    
                    # Emit histogram data signal
                    self.histogram_data_signal.emit(histogram_data)

                # === CSV EXPORT FOR ALL TRACKS ===
                # Export current state for ALL tracks, including occluded and lost ones
                if self.csv_writer_thread:
                    ts = self.frame_count  # Use frame count for consistent CSV output
                    for track_id in range(N):
                        # For successfully matched tracks, we already exported above
                        if track_id in matched:
                            continue
                        
                        # For unmatched tracks (occluded/lost), export last known position with current state
                        if len(self.trajectories_full[track_id]) > 0:
                            # Use last known position
                            last_x, last_y, last_theta, _ = self.trajectories_full[track_id][-1]
                            idx = local_counts[track_id]
                            traj_id = self.trajectory_ids[track_id]  # Use trajectory ID instead of track ID
                            self.csv_writer_thread.enqueue([
                                track_id, traj_id, idx, last_x, last_y, last_theta, ts, self.track_states[track_id]
                            ])
                            local_counts[track_id] += 1
                        # If no trajectory history yet, export NaN values
                        else:
                            idx = local_counts[track_id]
                            traj_id = self.trajectory_ids[track_id]  # Use trajectory ID instead of track ID
                            self.csv_writer_thread.enqueue([
                                track_id, traj_id, idx, float('nan'), float('nan'), float('nan'), ts, self.track_states[track_id]
                            ])
                            local_counts[track_id] += 1

                # === TRACK RESPAWNING ===
                # Assign unmatched detections to lost tracks for re-identification
                for d in free_dets:
                    for r in range(N):
                        if self.track_states[r] == 'lost':
                            kf = self.kalman_filters[r]
                            # Reinitialize Kalman filter with new detection
                            ns = np.array([meas[d][0], meas[d][1], meas[d][2], 0, 0], np.float32)
                            kf.statePre = ns.copy()
                            kf.statePost = ns.copy()
                            self.track_states[r] = 'active'
                            self.missed_frames[r] = 0
                            # Reset continuity for respawned track
                            self.tracking_continuity[r] = 0
                            # Assign new trajectory ID for respawned track
                            self.trajectory_ids[r] = self.next_trajectory_id
                            self.next_trajectory_id += 1
                            logger.debug(f"Final respawn: Track {r} -> Detection {d} - New trajectory ID: {self.trajectory_ids[r]}")
                            break  # Only respawn one track per detection

                # === TRACKING STABILIZATION DETECTION ===
                # Monitor tracking quality for system stability assessment
                if avg_cost < params["MAX_DISTANCE_THRESHOLD"]:
                    self.tracking_counts += 1
                else:
                    self.tracking_counts = 0
                    
                # Declare tracking stabilized after consistent good performance
                if self.tracking_counts >= params["MIN_TRACKING_COUNTS"] and not self.tracking_stabilized:
                    self.tracking_stabilized = True
                    logger.info(f"Tracking stabilized (avg cost={avg_cost:.2f})")

                # === VISUALIZATION OVERLAY ===
                # Check if any tracking visualization is enabled
                show_any_tracking = (params.get("SHOW_CIRCLES", False) or 
                                   params.get("SHOW_ORIENTATION", False) or 
                                   params.get("SHOW_TRAJECTORIES", False) or 
                                   params.get("SHOW_LABELS", False) or 
                                   params.get("SHOW_STATE", False))
                
                if show_any_tracking:
                    for i, tr in enumerate(self.trajectories_pruned):
                        if not tr: 
                            continue
                        
                        # Only show animals that are NOT lost
                        if self.track_states[i] == 'lost':
                            continue
                            
                        x, y, th, _ = tr[-1]  # Latest position
                        if math.isnan(x): 
                            continue
                            
                        # Get unique color for this track
                        col = tuple(int(c) for c in params["TRAJECTORY_COLORS"][i % len(params["TRAJECTORY_COLORS"])])
                        
                        # Draw current position as filled circle
                        if params.get("SHOW_CIRCLES", False):
                            cv2.circle(overlay, (x, y), 8, col, -1)
                        
                        # Draw orientation line
                        if params.get("SHOW_ORIENTATION", False):
                            end_x = int(x + 20 * math.cos(th))
                            end_y = int(y + 20 * math.sin(th))
                            cv2.line(overlay, (x, y), (end_x, end_y), col, 2)
                        
                        # Draw trajectory trail
                        if params.get("SHOW_TRAJECTORIES", False):
                            for a, b in zip(tr, tr[1:]):
                                p1 = (a[0], a[1])
                                p2 = (b[0], b[1])
                                # Check for valid coordinates
                                if not any(math.isnan(v) for v in (*p1, *p2)):
                                    cv2.line(overlay, p1, p2, col, 2)
                        
                        # Draw track ID and continuity label
                        if params.get("SHOW_LABELS", False) or params.get("SHOW_STATE", False):
                            label_parts = []
                            if params.get("SHOW_LABELS", False):
                                traj_id = self.trajectory_ids[i]  # Show trajectory ID instead of track ID
                                label_parts.append(f"T{traj_id} C:{self.tracking_continuity[i]}")
                            if params.get("SHOW_STATE", False):
                                label_parts.append(f"[{self.track_states[i]}]")
                            
                            if label_parts:
                                label_text = " ".join(label_parts)
                                cv2.putText(overlay, label_text, (x + 15, y - 15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            # === VIDEO OUTPUT WRITING ===
            # Write frame to output video BEFORE adding debug overlays
            # This ensures the output video has tracking overlays but no debug masks
            if self.video_writer is not None:
                self.video_writer.write(overlay)

            # === DEBUG VISUALIZATIONS ===
            # Show foreground mask in top-left corner
            if p["SHOW_FG"]:
                small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
                overlay[0:small_fg.shape[0], 0:small_fg.shape[1]] = cv2.cvtColor(small_fg, cv2.COLOR_GRAY2BGR)
                
            # Show background model in top-right corner
            if p["SHOW_BG"]:
                bg_bgr = cv2.cvtColor(bg_u8, cv2.COLOR_GRAY2BGR)
                small_bg = cv2.resize(bg_bgr, (0, 0), fx=0.3, fy=0.3)
                overlay[0:small_bg.shape[0], -small_bg.shape[1]:] = small_bg

            # Show lighting information if stabilization is enabled
            if params.get("ENABLE_LIGHTING_STABILIZATION", True) and hasattr(self, 'reference_intensity'):
                if self.reference_intensity is not None and len(self.intensity_history) > 0:
                    current_intensity = self.intensity_history[-1]
                    # Display lighting info on the frame
                    info_text = f"Ref: {self.reference_intensity:.1f}, Curr: {current_intensity:.1f}"
                    cv2.putText(overlay, info_text, (10, overlay.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # === PERFORMANCE MONITORING ===
            # Calculate and store current FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps_list.append(self.frame_count / elapsed)

            # === FRAME OUTPUT ===
            # Send processed frame to GUI for display
            self.emit_frame(overlay)
            
            # === MEMORY MANAGEMENT ===
            # Clean up large objects to prevent memory leaks
            del overlay, fg_mask, bg_u8, diff
            gc.collect()

        # === CLEANUP AND COMPLETION ===
        # Emit final progress update
        mode_str = "backward" if self.backward_mode else "forward"
        if total_frames and total_frames > 0:
            self.progress_signal.emit(100, f"Completed {mode_str} processing: {self.frame_count}/{total_frames} frames")
        else:
            self.progress_signal.emit(100, f"Completed {mode_str} processing: {self.frame_count} frames")
        
        cap.release()
        
        # Release video writer if it was initialized
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video output saved: {self.video_output_path}")
        
        # Apply post-processing if enabled
        final_trajectories = self.trajectories_full
        if params.get("enable_postprocessing", True):
            try:
                final_trajectories, trajectory_stats = self.post_process_trajectories(self.trajectories_full, params)
                logger.info("Trajectory post-processing completed successfully")
                
                # Write cleaned trajectories to a separate CSV if requested
                if self.csv_writer_thread and hasattr(self.csv_writer_thread, 'csv_path'):
                    cleaned_csv_path = self.csv_writer_thread.csv_path.replace('.csv', '_cleaned.csv')
                    self.write_cleaned_trajectories_csv(final_trajectories, cleaned_csv_path)
                    logger.info(f"Cleaned trajectories saved to: {cleaned_csv_path}")
                    
            except Exception as e:
                logger.error(f"Post-processing failed: {e}")
                logger.info("Using original trajectories without post-processing")
                final_trajectories = self.trajectories_full
        
        # Signal completion with success status, FPS data, and complete trajectories
        self.finished_signal.emit(not self._stop_requested, self.fps_list, final_trajectories)

    def emit_frame(self, bgr):
        """
        Convert BGR frame to RGB and emit signal for GUI display.
        
        Args:
            bgr (np.ndarray): BGR color frame from OpenCV
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb)
