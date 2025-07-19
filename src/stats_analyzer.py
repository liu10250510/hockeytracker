"""
Hockey Statistics Analyzer

This module analyzes tracking data to extract meaningful hockey statistics
including goals, assists, and ice time analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import logging
from collections import defaultdict
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsAnalyzer:
    """
    Analyzes player tracking data to extract hockey statistics.
    """
    
    def __init__(self):
        self.goal_zones = self._define_goal_zones()
        self.rink_dimensions = None
        
    def _define_goal_zones(self) -> Dict:
        """Define goal zone areas (will be calibrated per video)."""
        return {
            'left_goal': None,   # Will be set during calibration
            'right_goal': None,  # Will be set during calibration
            'goal_threshold': 50  # Pixels within goal area
        }
    
    def calibrate_rink(self, frame: np.ndarray, manual_points: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        Calibrate rink dimensions and goal locations.
        
        Args:
            frame: Sample frame from video
            manual_points: Optional manually selected goal locations [(x1,y1), (x2,y2)]
            
        Returns:
            Dictionary with rink calibration data
        """
        height, width = frame.shape[:2]
        
        if manual_points:
            # Use manually provided goal locations
            left_goal = manual_points[0]
            right_goal = manual_points[1]
        else:
            # Auto-detect goal locations (simplified approach)
            # In a real implementation, you'd use more sophisticated goal detection
            left_goal = (int(width * 0.05), int(height * 0.5))  # Left side
            right_goal = (int(width * 0.95), int(height * 0.5))  # Right side
        
        self.goal_zones['left_goal'] = left_goal
        self.goal_zones['right_goal'] = right_goal
        
        self.rink_dimensions = {
            'width': width,
            'height': height,
            'center_line': width // 2,
            'left_goal': left_goal,
            'right_goal': right_goal
        }
        
        logger.info(f"Rink calibrated: {width}x{height}, goals at {left_goal} and {right_goal}")
        return self.rink_dimensions
    
    def analyze_tracking_data(self, tracking_results: Dict) -> Dict:
        """
        Analyze complete tracking data to extract statistics.
        
        Args:
            tracking_results: Output from PlayerTracker.track_video()
            
        Returns:
            Comprehensive statistics analysis
        """
        frame_tracks = tracking_results['frame_tracks']
        video_info = tracking_results['video_info']
        
        # Extract player movements and events
        player_movements = self._extract_player_movements(frame_tracks)
        
        # Detect goals and assists
        goals_assists = self._detect_goals_and_assists(player_movements, video_info['fps'])
        
        # Calculate advanced statistics
        advanced_stats = self._calculate_advanced_stats(player_movements, video_info)
        
        # Compile final analysis
        analysis = {
            'player_movements': player_movements,
            'goals_and_assists': goals_assists,
            'advanced_statistics': advanced_stats,
            'video_summary': self._create_video_summary(tracking_results),
            'ice_time_analysis': self._analyze_ice_time(tracking_results)
        }
        
        return analysis
    
    def _extract_player_movements(self, frame_tracks: List[List[Dict]]) -> Dict:
        """Extract movement patterns for each player."""
        movements = defaultdict(list)
        
        for frame_idx, tracks in enumerate(frame_tracks):
            for track in tracks:
                player_id = track['track_id']
                bbox = track['bbox']
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                
                movements[player_id].append({
                    'frame': frame_idx,
                    'position': center,
                    'bbox': bbox,
                    'ice_time': track['on_ice_time']
                })
        
        return dict(movements)
    
    def _detect_goals_and_assists(self, movements: Dict, fps: float) -> Dict:
        """
        Detect potential goals and assists based on movement patterns.
        
        This is a simplified implementation. In practice, you'd use more
        sophisticated analysis including puck tracking, player proximity, etc.
        """
        goals_assists = defaultdict(lambda: {'goals': [], 'assists': []})
        
        if not self.rink_dimensions:
            logger.warning("Rink not calibrated. Cannot detect goals accurately.")
            return dict(goals_assists)
        
        left_goal = self.rink_dimensions['left_goal']
        right_goal = self.rink_dimensions['right_goal']
        threshold = self.goal_zones['goal_threshold']
        
        for player_id, positions in movements.items():
            for i, pos_data in enumerate(positions):
                position = pos_data['position']
                frame = pos_data['frame']
                
                # Check if player is near goal area
                dist_left = np.sqrt((position[0] - left_goal[0])**2 + (position[1] - left_goal[1])**2)
                dist_right = np.sqrt((position[0] - right_goal[0])**2 + (position[1] - right_goal[1])**2)
                
                # Simple goal detection based on proximity
                if dist_left < threshold or dist_right < threshold:
                    # Check for rapid movement towards goal (potential scoring play)
                    if i > 0:
                        prev_pos = positions[i-1]['position']
                        movement_vector = (position[0] - prev_pos[0], position[1] - prev_pos[1])
                        movement_magnitude = np.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
                        
                        # If significant movement towards goal area
                        if movement_magnitude > 20:  # Adjust threshold as needed
                            goal_time = frame / fps
                            goals_assists[player_id]['goals'].append({
                                'time': goal_time,
                                'frame': frame,
                                'position': position,
                                'goal_side': 'left' if dist_left < dist_right else 'right'
                            })
                            
                            # Look for potential assists (other players nearby)
                            self._detect_assists(movements, player_id, frame, position, goals_assists)
        
        return dict(goals_assists)
    
    def _detect_assists(self, movements: Dict, scoring_player: int, frame: int, 
                       goal_position: Tuple[int, int], goals_assists: Dict):
        """Detect potential assists for a goal."""
        assist_radius = 100  # Pixels
        frame_window = 30    # Frames before goal
        
        start_frame = max(0, frame - frame_window)
        
        for player_id, positions in movements.items():
            if player_id == scoring_player:
                continue
            
            # Check if this player was near the goal area recently
            for pos_data in positions:
                if start_frame <= pos_data['frame'] <= frame:
                    distance = np.sqrt((pos_data['position'][0] - goal_position[0])**2 + 
                                     (pos_data['position'][1] - goal_position[1])**2)
                    
                    if distance < assist_radius:
                        goals_assists[player_id]['assists'].append({
                            'time': frame / 30.0,  # Assuming 30 fps
                            'frame': frame,
                            'assisted_player': scoring_player,
                            'position': pos_data['position']
                        })
                        break
    
    def _calculate_advanced_stats(self, movements: Dict, video_info: Dict) -> Dict:
        """Calculate advanced hockey statistics."""
        advanced_stats = {}
        
        for player_id, positions in movements.items():
            if not positions:
                continue
            
            # Calculate distance covered
            total_distance = 0
            speeds = []
            
            for i in range(1, len(positions)):
                curr_pos = positions[i]['position']
                prev_pos = positions[i-1]['position']
                
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                 (curr_pos[1] - prev_pos[1])**2)
                total_distance += distance
                
                # Calculate speed (pixels per frame)
                frame_diff = positions[i]['frame'] - positions[i-1]['frame']
                if frame_diff > 0:
                    speed = distance / frame_diff
                    speeds.append(speed)
            
            # Calculate statistics
            avg_speed = np.mean(speeds) if speeds else 0
            max_speed = np.max(speeds) if speeds else 0
            
            # Zone analysis (simplified)
            zone_time = self._analyze_zone_time(positions)
            
            advanced_stats[player_id] = {
                'total_distance_pixels': total_distance,
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'zone_time': zone_time,
                'total_frames': len(positions),
                'ice_time_seconds': positions[-1]['ice_time'] if positions else 0
            }
        
        return advanced_stats
    
    def _analyze_zone_time(self, positions: List[Dict]) -> Dict:
        """Analyze time spent in different zones of the rink."""
        if not self.rink_dimensions:
            return {'offensive': 0, 'defensive': 0, 'neutral': 0}
        
        center_line = self.rink_dimensions['center_line']
        zone_counts = {'offensive': 0, 'defensive': 0, 'neutral': 0}
        
        for pos_data in positions:
            x = pos_data['position'][0]
            
            # Simple zone classification
            if x < center_line * 0.3:
                zone_counts['defensive'] += 1
            elif x > center_line * 1.7:
                zone_counts['offensive'] += 1
            else:
                zone_counts['neutral'] += 1
        
        # Convert to percentages
        total = sum(zone_counts.values())
        if total > 0:
            for zone in zone_counts:
                zone_counts[zone] = (zone_counts[zone] / total) * 100
        
        return zone_counts
    
    def _create_video_summary(self, tracking_results: Dict) -> Dict:
        """Create a summary of the video analysis."""
        player_stats = tracking_results['player_statistics']
        video_info = tracking_results['video_info']
        
        total_players = len(player_stats)
        total_ice_time = sum(stats['total_ice_time'] for stats in player_stats.values())
        avg_ice_time = total_ice_time / total_players if total_players > 0 else 0
        
        return {
            'total_players_detected': total_players,
            'video_duration': video_info['duration'],
            'total_frames': video_info['total_frames'],
            'fps': video_info['fps'],
            'average_ice_time_per_player': avg_ice_time,
            'total_combined_ice_time': total_ice_time
        }
    
    def _analyze_ice_time(self, tracking_results: Dict) -> Dict:
        """Detailed ice time analysis."""
        player_stats = tracking_results['player_statistics']
        
        ice_times = [stats['total_ice_time'] for stats in player_stats.values()]
        
        if not ice_times:
            return {}
        
        return {
            'min_ice_time': min(ice_times),
            'max_ice_time': max(ice_times),
            'average_ice_time': np.mean(ice_times),
            'median_ice_time': np.median(ice_times),
            'std_ice_time': np.std(ice_times),
            'ice_time_distribution': {
                f'player_{i+1}': time for i, time in enumerate(ice_times)
            }
        }
    
    def export_statistics(self, analysis: Dict, output_path: str, format: str = 'json'):
        """
        Export analysis results to file.
        
        Args:
            analysis: Analysis results from analyze_tracking_data()
            output_path: Path to save the analysis
            format: Export format ('json', 'csv', 'excel')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        
        elif format == 'csv':
            # Create a DataFrame with key statistics
            stats_data = []
            for player_id, stats in analysis['advanced_statistics'].items():
                row = {
                    'player_id': player_id,
                    'ice_time_seconds': stats['ice_time_seconds'],
                    'total_distance_pixels': stats['total_distance_pixels'],
                    'average_speed': stats['average_speed'],
                    'max_speed': stats['max_speed'],
                    'offensive_zone_percent': stats['zone_time']['offensive'],
                    'defensive_zone_percent': stats['zone_time']['defensive'],
                    'neutral_zone_percent': stats['zone_time']['neutral']
                }
                
                # Add goals and assists if available
                if 'goals_and_assists' in analysis:
                    ga = analysis['goals_and_assists'].get(player_id, {'goals': [], 'assists': []})
                    row['goals'] = len(ga['goals'])
                    row['assists'] = len(ga['assists'])
                
                stats_data.append(row)
            
            df = pd.DataFrame(stats_data)
            df.to_csv(output_path, index=False)
        
        elif format == 'excel':
            # Similar to CSV but save as Excel
            stats_data = []
            for player_id, stats in analysis['advanced_statistics'].items():
                row = {
                    'player_id': player_id,
                    'ice_time_seconds': stats['ice_time_seconds'],
                    'total_distance_pixels': stats['total_distance_pixels'],
                    'average_speed': stats['average_speed'],
                    'max_speed': stats['max_speed'],
                    'offensive_zone_percent': stats['zone_time']['offensive'],
                    'defensive_zone_percent': stats['zone_time']['defensive'],
                    'neutral_zone_percent': stats['zone_time']['neutral']
                }
                
                if 'goals_and_assists' in analysis:
                    ga = analysis['goals_and_assists'].get(player_id, {'goals': [], 'assists': []})
                    row['goals'] = len(ga['goals'])
                    row['assists'] = len(ga['assists'])
                
                stats_data.append(row)
            
            df = pd.DataFrame(stats_data)
            df.to_excel(output_path, index=False)
        
        logger.info(f"Statistics exported to {output_path} in {format} format")


if __name__ == "__main__":
    # Example usage
    from player_detector import PlayerDetector
    from player_tracker import PlayerTracker
    
    # Initialize components
    detector = PlayerDetector()
    tracker = PlayerTracker()
    analyzer = StatsAnalyzer()
    
    video_path = "data/videos/sample_game.mp4"
    
    try:
        # Track players in video
        tracking_results = tracker.track_video(video_path, detector)
        
        # Analyze the tracking data
        analysis = analyzer.analyze_tracking_data(tracking_results)
        
        # Print summary
        print("\n=== HOCKEY STATISTICS ANALYSIS ===")
        summary = analysis['video_summary']
        print(f"Players detected: {summary['total_players_detected']}")
        print(f"Video duration: {summary['video_duration']:.2f} seconds")
        print(f"Average ice time per player: {summary['average_ice_time_per_player']:.2f} seconds")
        
        print("\n=== INDIVIDUAL PLAYER STATS ===")
        for player_id, stats in analysis['advanced_statistics'].items():
            print(f"\nPlayer {player_id}:")
            print(f"  Ice time: {stats['ice_time_seconds']:.2f}s")
            print(f"  Distance covered: {stats['total_distance_pixels']:.0f} pixels")
            print(f"  Average speed: {stats['average_speed']:.2f} pixels/frame")
            print(f"  Zone distribution: O:{stats['zone_time']['offensive']:.1f}% "
                  f"N:{stats['zone_time']['neutral']:.1f}% "
                  f"D:{stats['zone_time']['defensive']:.1f}%")
        
        # Export results
        analyzer.export_statistics(analysis, "data/output/hockey_analysis.json", "json")
        analyzer.export_statistics(analysis, "data/output/hockey_stats.csv", "csv")
        
        print(f"\nAnalysis complete! Results saved to data/output/")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
