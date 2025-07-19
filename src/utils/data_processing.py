"""
Data processing utilities for hockey tracking analysis.

This module provides functions for data cleaning, transformation,
and preparation for analysis and machine learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data processing operations for hockey tracking data.
    """
    
    def __init__(self):
        self.processed_data = {}
    
    def tracking_to_dataframe(self, tracking_results: Dict) -> pd.DataFrame:
        """
        Convert tracking results to pandas DataFrame for analysis.
        
        Args:
            tracking_results: Output from PlayerTracker.track_video()
            
        Returns:
            DataFrame with tracking data
        """
        frame_tracks = tracking_results['frame_tracks']
        video_info = tracking_results['video_info']
        
        data_rows = []
        
        for frame_idx, tracks in enumerate(frame_tracks):
            timestamp = frame_idx / video_info['fps']  # Convert to seconds
            
            for track in tracks:
                bbox = track['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                
                row = {
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'player_id': track['track_id'],
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox_x1': bbox[0],
                    'bbox_y1': bbox[1],
                    'bbox_x2': bbox[2],
                    'bbox_y2': bbox[3],
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'ice_time_cumulative': track['on_ice_time'],
                    'goals': track['goals'],
                    'assists': track['assists'],
                    'hits': track['hits']
                }
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        return df
    
    def calculate_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate velocity and acceleration for each player.
        
        Args:
            df: DataFrame with tracking data
            
        Returns:
            DataFrame with velocity and acceleration columns added
        """
        df_with_velocity = df.copy()
        
        # Sort by player and frame
        df_with_velocity = df_with_velocity.sort_values(['player_id', 'frame'])
        
        # Calculate velocities for each player
        velocity_data = []
        
        for player_id in df_with_velocity['player_id'].unique():
            player_data = df_with_velocity[df_with_velocity['player_id'] == player_id].copy()
            
            # Calculate position differences
            player_data['dx'] = player_data['center_x'].diff()
            player_data['dy'] = player_data['center_y'].diff()
            player_data['dt'] = player_data['timestamp'].diff()
            
            # Calculate velocity components
            player_data['velocity_x'] = player_data['dx'] / player_data['dt']
            player_data['velocity_y'] = player_data['dy'] / player_data['dt']
            
            # Calculate velocity magnitude
            player_data['velocity_magnitude'] = np.sqrt(
                player_data['velocity_x']**2 + player_data['velocity_y']**2
            )
            
            # Calculate acceleration
            player_data['acceleration_x'] = player_data['velocity_x'].diff() / player_data['dt']
            player_data['acceleration_y'] = player_data['velocity_y'].diff() / player_data['dt']
            player_data['acceleration_magnitude'] = np.sqrt(
                player_data['acceleration_x']**2 + player_data['acceleration_y']**2
            )
            
            # Calculate direction of movement
            player_data['movement_angle'] = np.arctan2(player_data['dy'], player_data['dx']) * 180 / np.pi
            
            velocity_data.append(player_data)
        
        result_df = pd.concat(velocity_data, ignore_index=True)
        
        # Fill NaN values for first frames
        result_df = result_df.fillna(0)
        
        return result_df
    
    def smooth_trajectories(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """
        Apply smoothing to player trajectories to reduce noise.
        
        Args:
            df: DataFrame with tracking data
            window_size: Size of smoothing window
            
        Returns:
            DataFrame with smoothed positions
        """
        df_smoothed = df.copy()
        
        for player_id in df_smoothed['player_id'].unique():
            mask = df_smoothed['player_id'] == player_id
            
            # Apply rolling mean to positions
            df_smoothed.loc[mask, 'center_x_smooth'] = (
                df_smoothed.loc[mask, 'center_x'].rolling(window=window_size, center=True).mean()
            )
            df_smoothed.loc[mask, 'center_y_smooth'] = (
                df_smoothed.loc[mask, 'center_y'].rolling(window=window_size, center=True).mean()
            )
        
        # Fill NaN values at edges
        df_smoothed['center_x_smooth'] = df_smoothed['center_x_smooth'].fillna(df_smoothed['center_x'])
        df_smoothed['center_y_smooth'] = df_smoothed['center_y_smooth'].fillna(df_smoothed['center_y'])
        
        return df_smoothed
    
    def detect_events(self, df: pd.DataFrame, rink_dimensions: Optional[Dict] = None) -> Dict:
        """
        Detect hockey events from tracking data.
        
        Args:
            df: DataFrame with tracking data
            rink_dimensions: Rink calibration data
            
        Returns:
            Dictionary with detected events
        """
        events = {
            'goals': [],
            'assists': [],
            'fast_breaks': [],
            'zone_entries': [],
            'collisions': []
        }
        
        if rink_dimensions is None:
            logger.warning("No rink dimensions provided. Event detection will be limited.")
            return events
        
        # Detect fast breaks (high speed movements)
        if 'velocity_magnitude' in df.columns:
            high_speed_threshold = df['velocity_magnitude'].quantile(0.9)
            fast_breaks = df[df['velocity_magnitude'] > high_speed_threshold]
            
            for _, row in fast_breaks.iterrows():
                events['fast_breaks'].append({
                    'timestamp': row['timestamp'],
                    'player_id': row['player_id'],
                    'speed': row['velocity_magnitude'],
                    'position': (row['center_x'], row['center_y'])
                })
        
        # Detect zone entries (simplified)
        center_line = rink_dimensions.get('center_line', rink_dimensions['width'] / 2)
        
        for player_id in df['player_id'].unique():
            player_data = df[df['player_id'] == player_id].sort_values('frame')
            
            prev_zone = None
            for _, row in player_data.iterrows():
                current_zone = 'left' if row['center_x'] < center_line else 'right'
                
                if prev_zone and prev_zone != current_zone:
                    events['zone_entries'].append({
                        'timestamp': row['timestamp'],
                        'player_id': player_id,
                        'from_zone': prev_zone,
                        'to_zone': current_zone,
                        'position': (row['center_x'], row['center_y'])
                    })
                
                prev_zone = current_zone
        
        # Detect potential collisions (players very close together)
        collision_threshold = 50  # pixels
        
        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]
            
            if len(frame_data) < 2:
                continue
            
            # Check distance between all pairs of players
            for i, row1 in frame_data.iterrows():
                for j, row2 in frame_data.iterrows():
                    if i >= j:  # Avoid duplicate pairs
                        continue
                    
                    distance = np.sqrt(
                        (row1['center_x'] - row2['center_x'])**2 + 
                        (row1['center_y'] - row2['center_y'])**2
                    )
                    
                    if distance < collision_threshold:
                        events['collisions'].append({
                            'timestamp': row1['timestamp'],
                            'player1_id': row1['player_id'],
                            'player2_id': row2['player_id'],
                            'distance': distance,
                            'position': ((row1['center_x'] + row2['center_x'])/2,
                                       (row1['center_y'] + row2['center_y'])/2)
                        })
        
        return events
    
    def calculate_player_statistics(self, df: pd.DataFrame, video_duration: float) -> Dict:
        """
        Calculate comprehensive player statistics from tracking data.
        
        Args:
            df: DataFrame with tracking data
            video_duration: Total video duration in seconds
            
        Returns:
            Dictionary with player statistics
        """
        stats = {}
        
        for player_id in df['player_id'].unique():
            player_data = df[df['player_id'] == player_id]
            
            # Basic statistics
            ice_time = len(player_data) / len(df) * video_duration if len(df) > 0 else 0
            total_distance = 0
            
            if 'velocity_magnitude' in player_data.columns:
                # Calculate total distance (approximate)
                velocities = player_data['velocity_magnitude'].dropna()
                if len(velocities) > 0:
                    avg_velocity = velocities.mean()
                    max_velocity = velocities.max()
                    total_distance = velocities.sum() * (1/30)  # Assuming 30 fps
                else:
                    avg_velocity = max_velocity = 0
            else:
                avg_velocity = max_velocity = 0
            
            # Position-based statistics
            positions_x = player_data['center_x']
            positions_y = player_data['center_y']
            
            avg_position = (positions_x.mean(), positions_y.mean())
            position_variance = (positions_x.var(), positions_y.var())
            
            # Time-based statistics
            first_appearance = player_data['timestamp'].min()
            last_appearance = player_data['timestamp'].max()
            active_duration = last_appearance - first_appearance
            
            stats[player_id] = {
                'ice_time_seconds': ice_time,
                'total_frames': len(player_data),
                'total_distance_estimate': total_distance,
                'average_velocity': avg_velocity,
                'max_velocity': max_velocity,
                'average_position': avg_position,
                'position_variance': position_variance,
                'first_appearance': first_appearance,
                'last_appearance': last_appearance,
                'active_duration': active_duration,
                'goals': player_data['goals'].iloc[-1] if len(player_data) > 0 else 0,
                'assists': player_data['assists'].iloc[-1] if len(player_data) > 0 else 0
            }
        
        return stats
    
    def export_to_formats(self, df: pd.DataFrame, base_filename: str) -> Dict[str, str]:
        """
        Export DataFrame to multiple formats.
        
        Args:
            df: DataFrame to export
            base_filename: Base filename without extension
            
        Returns:
            Dictionary mapping format to filepath
        """
        exported_files = {}
        
        # CSV
        csv_path = f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        exported_files['csv'] = csv_path
        
        # Excel
        excel_path = f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data
            df.to_excel(writer, sheet_name='Tracking_Data', index=False)
            
            # Summary statistics
            summary_stats = df.groupby('player_id').agg({
                'ice_time_cumulative': 'max',
                'velocity_magnitude': ['mean', 'max'] if 'velocity_magnitude' in df.columns else 'count',
                'goals': 'max',
                'assists': 'max'
            })
            summary_stats.to_excel(writer, sheet_name='Summary_Stats')
        
        exported_files['excel'] = excel_path
        
        # JSON (sample data only to avoid huge files)
        json_path = f"{base_filename}_sample.json"
        sample_data = df.head(1000).to_dict('records')  # First 1000 rows
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
        exported_files['json'] = json_path
        
        logger.info(f"Data exported to: {list(exported_files.values())}")
        return exported_files
    
    def create_time_series_data(self, df: pd.DataFrame, interval_seconds: int = 10) -> pd.DataFrame:
        """
        Create time series data aggregated over intervals.
        
        Args:
            df: DataFrame with tracking data
            interval_seconds: Time interval for aggregation
            
        Returns:
            DataFrame with time series data
        """
        # Create time bins
        max_time = df['timestamp'].max()
        time_bins = np.arange(0, max_time + interval_seconds, interval_seconds)
        df['time_bin'] = pd.cut(df['timestamp'], time_bins, include_lowest=True)
        
        # Aggregate data by time bins and player
        agg_functions = {
            'center_x': 'mean',
            'center_y': 'mean',
            'velocity_magnitude': 'mean' if 'velocity_magnitude' in df.columns else 'count',
            'ice_time_cumulative': 'max',
            'goals': 'max',
            'assists': 'max'
        }
        
        time_series = df.groupby(['player_id', 'time_bin']).agg(agg_functions).reset_index()
        
        # Add time bin midpoint
        time_series['time_midpoint'] = time_series['time_bin'].apply(
            lambda x: x.mid if pd.notna(x) else 0
        )
        
        return time_series
    
    def filter_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Filter outliers from specified columns.
        
        Args:
            df: DataFrame to filter
            columns: List of column names to check for outliers
            method: Method to use ('iqr', 'zscore')
            factor: Factor for outlier detection
            
        Returns:
            DataFrame with outliers filtered
        """
        df_filtered = df.copy()
        
        for column in columns:
            if column not in df_filtered.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_filtered[column].quantile(0.25)
                Q3 = df_filtered[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)
                df_filtered = df_filtered[mask]
            
            elif method == 'zscore':
                z_scores = np.abs((df_filtered[column] - df_filtered[column].mean()) / df_filtered[column].std())
                mask = z_scores <= factor
                df_filtered = df_filtered[mask]
        
        logger.info(f"Filtered {len(df) - len(df_filtered)} outlier rows")
        return df_filtered


if __name__ == "__main__":
    # Example usage with dummy data
    processor = DataProcessor()
    
    # Create dummy tracking results
    dummy_tracking = {
        'frame_tracks': [
            [{'track_id': 1, 'bbox': (100, 100, 150, 200), 'on_ice_time': 1.0, 'goals': 0, 'assists': 0, 'hits': 1}],
            [{'track_id': 1, 'bbox': (105, 105, 155, 205), 'on_ice_time': 2.0, 'goals': 0, 'assists': 0, 'hits': 2}],
            [{'track_id': 1, 'bbox': (110, 110, 160, 210), 'on_ice_time': 3.0, 'goals': 1, 'assists': 0, 'hits': 3}]
        ],
        'video_info': {'fps': 30.0, 'duration': 0.1, 'total_frames': 3}
    }
    
    # Convert to DataFrame
    df = processor.tracking_to_dataframe(dummy_tracking)
    print("Tracking DataFrame:")
    print(df.head())
    
    # Calculate velocities
    df_with_velocity = processor.calculate_velocities(df)
    print("\nWith velocity calculations:")
    print(df_with_velocity[['player_id', 'velocity_magnitude']].head())
    
    # Calculate statistics
    stats = processor.calculate_player_statistics(df_with_velocity, 0.1)
    print("\nPlayer statistics:")
    for player_id, player_stats in stats.items():
        print(f"Player {player_id}: {player_stats}")
