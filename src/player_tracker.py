"""
Hockey Player Tracking Module

This module provides functionality to track individual hockey players across
video frames using multi-object tracking algorithms.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

# Import tracking algorithms
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Import headless utilities
from src.utils.headless_utils import HeadlessSafeVideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Track:
    """
    Represents a single player track across frames.
    """
    
    def __init__(self, track_id: int, initial_detection: Dict):
        self.track_id = track_id
        self.detections = [initial_detection]
        self.kalman_filter = self._init_kalman_filter(initial_detection)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.on_ice_time = 0.0  # Time on ice in seconds
        self.goals = 0
        self.assists = 0
        
    def _init_kalman_filter(self, detection: Dict) -> KalmanFilter:
        """Initialize Kalman filter for tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # w
            [0, 0, 0, 1, 0, 0, 0],  # h
            [0, 0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1]   # vw
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R[2:, 2:] *= 10.0
        
        # Process noise
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        bbox = detection['bbox']
        kf.x[:4] = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]).reshape((4, 1))
        
        return kf
    
    def predict(self):
        """Predict next state using Kalman filter."""
        if self.kalman_filter.x[6] + self.kalman_filter.x[2] <= 0:
            self.kalman_filter.x[6] *= 0.0
        self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection: Dict):
        """Update track with new detection."""
        self.time_since_update = 0
        self.hits += 1
        
        bbox = detection['bbox']
        measurement = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        self.kalman_filter.update(measurement)
        
        self.detections.append(detection)
        
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current bounding box prediction."""
        if self.kalman_filter.x[6] + self.kalman_filter.x[2] <= 0:
            return tuple(map(int, self.kalman_filter.x[:4].flatten()))
        
        x, y, w, h = self.kalman_filter.x[:4].flatten()
        return (int(x), int(y), int(x + w), int(y + h))


class PlayerTracker:
    """
    Multi-object tracker for hockey players using Kalman filters and Hungarian algorithm.
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize the player tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without updates
            min_hits: Minimum number of hits to consider a track valid
            iou_threshold: IoU threshold for associating detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Associate detections to existing tracks using Hungarian algorithm.
        
        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        # Build cost matrix based on IoU
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            track_bbox = track.get_state()
            for d, detection in enumerate(detections):
                detection_bbox = detection['bbox']
                iou = self._calculate_iou(track_bbox, detection_bbox)
                cost_matrix[t, d] = 1 - iou  # Convert IoU to cost
        
        # Apply Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for t, d in zip(track_indices, detection_indices):
            if cost_matrix[t, d] <= (1 - self.iou_threshold):
                matches.append((t, d))
                unmatched_detections.remove(d)
                unmatched_tracks.remove(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections: List[Dict], fps: float = 30.0) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of player detections for current frame
            fps: Frames per second for ice time calculation
            
        Returns:
            List of active tracks with IDs
        """
        self.frame_count += 1
        
        # Predict next state for all tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
            # Update ice time (assuming player is on ice if detected)
            self.tracks[track_idx].on_ice_time += 1.0 / fps
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[detection_idx])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update <= self.max_age]
        
        # Return active tracks that meet minimum hit requirement
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = track.get_state()
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'on_ice_time': track.on_ice_time,
                    'goals': track.goals,
                    'assists': track.assists,
                    'hits': track.hits
                })
        
        return active_tracks
    
    def track_video(self, video_path: str, detector, output_path: Optional[str] = None) -> Dict:
        """
        Track players throughout entire video.
        
        Args:
            video_path: Path to input video
            detector: PlayerDetector instance
            output_path: Optional path to save annotated video
            
        Returns:
            Dictionary containing tracking results and statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Tracking video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_tracks = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players in current frame
                detections = detector.detect_players(frame)
                
                # Update tracker
                tracks = self.update(detections, fps)
                all_tracks.append(tracks)
                
                # Draw tracking results if saving output
                if out is not None:
                    annotated_frame = self._draw_tracks(frame, tracks)
                    out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Tracked {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            HeadlessSafeVideoProcessor.destroy_all_windows()
        
        # Compile final statistics
        final_stats = self._compile_statistics(all_tracks, fps)
        
        logger.info("Tracking complete!")
        return {
            'frame_tracks': all_tracks,
            'player_statistics': final_stats,
            'video_info': {
                'fps': fps,
                'total_frames': frame_count,
                'duration': frame_count / fps
            }
        }
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """Draw tracking results on frame."""
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            
            # Draw bounding box
            color = self._get_track_color(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track info
            label = f"Player {track_id}"
            cv2.putText(annotated_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw ice time
            ice_time_str = f"Ice: {track['on_ice_time']:.1f}s"
            cv2.putText(annotated_frame, ice_time_str, (x1, y2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        return colors[track_id % len(colors)]
    
    def _compile_statistics(self, all_tracks: List[List[Dict]], fps: float) -> Dict:
        """Compile final player statistics from tracking data."""
        player_stats = defaultdict(lambda: {
            'total_ice_time': 0.0,
            'goals': 0,
            'assists': 0,
            'frames_detected': 0
        })
        
        # Track maximum ice time per player across all frames
        max_ice_times = defaultdict(float)
        
        for frame_tracks in all_tracks:
            for track in frame_tracks:
                player_id = track['track_id']
                player_stats[player_id]['frames_detected'] += 1
                player_stats[player_id]['goals'] = track['goals']
                player_stats[player_id]['assists'] = track['assists']
                
                # Keep track of maximum ice time seen
                max_ice_times[player_id] = max(max_ice_times[player_id], track['on_ice_time'])
        
        # Set final ice times to maximum observed
        for player_id in player_stats:
            player_stats[player_id]['total_ice_time'] = max_ice_times[player_id]
        
        return dict(player_stats)


if __name__ == "__main__":
    # Example usage
    from player_detector import PlayerDetector
    
    detector = PlayerDetector()
    tracker = PlayerTracker()
    
    video_path = "data/videos/sample_game.mp4"
    output_path = "data/output/tracked_players.mp4"
    
    try:
        results = tracker.track_video(video_path, detector, output_path)
        
        print("\nTracking Results:")
        print(f"Total frames processed: {results['video_info']['total_frames']}")
        print(f"Video duration: {results['video_info']['duration']:.2f} seconds")
        print(f"Detected {len(results['player_statistics'])} unique players")
        
        print("\nPlayer Statistics:")
        for player_id, stats in results['player_statistics'].items():
            print(f"Player {player_id}:")
            print(f"  Ice time: {stats['total_ice_time']:.2f} seconds")
            print(f"  Goals: {stats['goals']}")
            print(f"  Assists: {stats['assists']}")
            print(f"  Frames detected: {stats['frames_detected']}")
            print()
        
    except Exception as e:
        logger.error(f"Error tracking video: {e}")
