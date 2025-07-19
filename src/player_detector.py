"""
Hockey Player Detection using YOLO v8

This module provides functionality to detect hockey players in video frames
using the YOLO v8 object detection model.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    Hockey player detector using YOLO v8 model.
    
    Detects players in hockey game videos and returns bounding box coordinates
    along with confidence scores.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the player detector.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in a single frame.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of player detections with bounding boxes and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold)
        
        players = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter for person class (class 0 in COCO dataset)
                    if int(box.cls) == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Additional filtering for hockey players
                        if self._is_likely_hockey_player(frame, (x1, y1, x2, y2)):
                            players.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            })
        
        return players
    
    def _is_likely_hockey_player(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Additional filtering to determine if detection is likely a hockey player.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Boolean indicating if detection is likely a hockey player
        """
        x1, y1, x2, y2 = bbox
        
        # Check aspect ratio (hockey players are typically taller than wide)
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 0
        
        # Hockey players typically have aspect ratio between 1.5 and 4.0
        if not (1.5 <= aspect_ratio <= 4.0):
            return False
        
        # Check size relative to frame (too small or too large detections are unlikely)
        frame_area = frame.shape[0] * frame.shape[1]
        detection_area = width * height
        relative_size = detection_area / frame_area
        
        # Hockey players should be between 0.01% and 25% of frame area
        if not (0.0001 <= relative_size <= 0.25):
            return False
        
        return True
    
    def detect_players_in_video(self, video_path: str, output_path: Optional[str] = None) -> List[List[Dict]]:
        """
        Detect players in entire video.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            
        Returns:
            List of player detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players in current frame
                players = self.detect_players(frame)
                all_detections.append(players)
                
                # Draw bounding boxes if saving output video
                if out is not None:
                    annotated_frame = self._draw_detections(frame, players)
                    out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Detection complete. Processed {frame_count} frames")
        return all_detections
    
    def _draw_detections(self, frame: np.ndarray, players: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame
            players: List of player detections
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, player in enumerate(players):
            x1, y1, x2, y2 = player['bbox']
            confidence = player['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID and confidence
            label = f"Player {i+1}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_frame


if __name__ == "__main__":
    # Example usage
    detector = PlayerDetector()
    
    # Test with a sample video (replace with your video path)
    video_path = "data/videos/sample_game.mp4"
    output_path = "data/output/detected_players.mp4"
    
    try:
        detections = detector.detect_players_in_video(video_path, output_path)
        print(f"Detection complete! Found players in {len(detections)} frames")
        
        # Print summary statistics
        total_detections = sum(len(frame_detections) for frame_detections in detections)
        avg_players_per_frame = total_detections / len(detections) if detections else 0
        print(f"Average players per frame: {avg_players_per_frame:.2f}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
