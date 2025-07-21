"""
Video Processing Utilities

This module provides utilities for video processing, frame extraction,
and preprocessing for hockey game analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
import os
import logging
from pathlib import Path

# Import headless utilities
from src.utils.headless_utils import HeadlessSafeVideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video processing operations for hockey game analysis.
    """
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def validate_video(self, video_path: str) -> bool:
        """
        Validate if video file exists and is readable.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Boolean indicating if video is valid
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported video format: {file_ext}")
            return False
        
        # Try to open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return False
        
        cap.release()
        return True
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30, max_frames: Optional[int] = None) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frame file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {extracted_count} frames to {output_dir}")
        return frame_paths
    
    def resize_video(self, input_path: str, output_path: str, 
                    target_width: int, target_height: int) -> None:
        """
        Resize video to specified dimensions.
        
        Args:
            input_path: Path to input video
            output_path: Path to save resized video
            target_width: Target width in pixels
            target_height: Target height in pixels
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized_frame = cv2.resize(frame, (target_width, target_height))
                out.write(resized_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video resized and saved to {output_path}")
    
    def crop_video(self, input_path: str, output_path: str, 
                   crop_region: Tuple[int, int, int, int]) -> None:
        """
        Crop video to specified region.
        
        Args:
            input_path: Path to input video
            output_path: Path to save cropped video
            crop_region: (x, y, width, height) of crop region
        """
        x, y, width, height = crop_region
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cropped_frame = frame[y:y+height, x:x+width]
                out.write(cropped_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Video cropped and saved to {output_path}")
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better detection.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_frame, -1, kernel)
        
        # Blend original and sharpened (subtle sharpening)
        result = cv2.addWeighted(enhanced_frame, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def detect_scene_changes(self, video_path: str, threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes in video (useful for period/play detection).
        
        Args:
            video_path: Path to input video
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame numbers where scene changes occur
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        scene_changes = []
        prev_frame = None
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate histogram difference
                    hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
                    
                    # Compare histograms
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    if correlation < (1 - threshold):
                        scene_changes.append(frame_count)
                        logger.info(f"Scene change detected at frame {frame_count}")
                
                prev_frame = gray_frame
                frame_count += 1
        
        finally:
            cap.release()
        
        return scene_changes
    
    def create_video_summary(self, video_path: str, key_frames: List[int], 
                           output_path: str, frame_duration: float = 2.0) -> None:
        """
        Create a summary video from key frames.
        
        Args:
            video_path: Path to input video
            key_frames: List of frame numbers to include in summary
            output_path: Path to save summary video
            frame_duration: Duration to display each frame (seconds)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Fixed FPS for summary video
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_per_keyframe = int(fps * frame_duration)
        
        try:
            for frame_num in key_frames:
                # Seek to specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # Write frame multiple times to achieve desired duration
                    for _ in range(frames_per_keyframe):
                        out.write(frame)
        
        finally:
            cap.release()
            out.release()
        
        logger.info(f"Summary video created: {output_path}")
    
    def frame_generator(self, video_path: str, start_frame: int = 0, 
                       end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None for end of video)
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if end_frame is not None and frame_count >= end_frame:
                    break
                
                yield frame_count, frame
                frame_count += 1
        
        finally:
            cap.release()


class RinkCalibrator:
    """
    Helper class for calibrating hockey rink dimensions in video.
    """
    
    def __init__(self):
        self.points = []
        self.window_name = "Rink Calibration"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for manual point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
    
    def calibrate_manual(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Manual calibration by clicking on rink features.
        
        Args:
            frame: Sample frame from video
            
        Returns:
            List of calibration points
        """
        print("Click on the following points in order:")
        print("1. Left goal center")
        print("2. Right goal center")
        print("3. Center ice (optional)")
        print("Press 'q' when done")
        
        # Skip GUI operations if in headless environment
        if HeadlessSafeVideoProcessor.is_headless_environment():
            logger.warning("Running in headless environment. Calibration points cannot be collected via GUI.")
            # Return default calibration points
            return [(100, 250), (540, 250)]  # Default left and right goal centers
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        display_frame = frame.copy()
        
        while True:
            # Draw existing points
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"P{i+1}", 
                           (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(self.points) >= 2:
                break
        
        cv2.destroyAllWindows()
        return self.points.copy()


if __name__ == "__main__":
    # Example usage
    processor = VideoProcessor()
    
    video_path = "data/videos/sample_game.mp4"
    
    if processor.validate_video(video_path):
        # Get video information
        info = processor.get_video_info(video_path)
        print(f"Video info: {info}")
        
        # Extract sample frames
        processor.extract_frames(video_path, "data/frames/", frame_interval=300, max_frames=10)
        
        # Detect scene changes
        scene_changes = processor.detect_scene_changes(video_path)
        print(f"Scene changes detected at frames: {scene_changes}")
        
    else:
        print("Invalid video file")
