"""
Visualization utilities for hockey tracking data.

This module provides functions to create plots, charts, and visual representations
of player tracking statistics and game analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HockeyVisualizer:
    """
    Creates visualizations for hockey tracking data and statistics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_player_ice_time(self, player_stats: Dict, save_path: Optional[str] = None) -> None:
        """
        Create bar plot of player ice time.
        
        Args:
            player_stats: Dictionary with player statistics
            save_path: Optional path to save the plot
        """
        players = list(player_stats.keys())
        ice_times = [stats['total_ice_time'] for stats in player_stats.values()]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(range(len(players)), ice_times, color=self.colors[:len(players)])
        
        plt.title('Player Ice Time Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Player ID', fontsize=12)
        plt.ylabel('Ice Time (seconds)', fontsize=12)
        plt.xticks(range(len(players)), [f'Player {p}' for p in players], rotation=45)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, ice_times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_player_movement_heatmap(self, movements: Dict, frame_width: int, 
                                   frame_height: int, save_path: Optional[str] = None) -> None:
        """
        Create heatmap of player movement patterns.
        
        Args:
            movements: Player movement data from tracking
            frame_width: Width of video frame
            frame_height: Height of video frame
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        player_ids = list(movements.keys())[:6]  # Show up to 6 players
        
        for i, player_id in enumerate(player_ids):
            positions = movements[player_id]
            x_coords = [pos['position'][0] for pos in positions]
            y_coords = [pos['position'][1] for pos in positions]
            
            # Create 2D histogram for heatmap
            heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, 
                                                   bins=[frame_width//20, frame_height//20],
                                                   range=[[0, frame_width], [0, frame_height]])
            
            im = axes[i].imshow(heatmap.T, origin='lower', aspect='auto', 
                              extent=[0, frame_width, 0, frame_height], 
                              cmap='hot', alpha=0.8)
            
            axes[i].set_title(f'Player {player_id} Movement Heatmap', fontweight='bold')
            axes[i].set_xlabel('X Position (pixels)')
            axes[i].set_ylabel('Y Position (pixels)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Hide unused subplots
        for i in range(len(player_ids), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Player Movement Heatmaps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_player_trajectories(self, movements: Dict, frame_width: int, 
                               frame_height: int, save_path: Optional[str] = None) -> None:
        """
        Plot player movement trajectories on ice.
        
        Args:
            movements: Player movement data
            frame_width: Width of video frame
            frame_height: Height of video frame
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(16, 10))
        
        # Draw rink outline (simplified)
        rink_x = [0, frame_width, frame_width, 0, 0]
        rink_y = [0, 0, frame_height, frame_height, 0]
        plt.plot(rink_x, rink_y, 'k-', linewidth=3, label='Rink')
        
        # Draw center line
        plt.axvline(x=frame_width/2, color='red', linestyle='--', alpha=0.7, label='Center Line')
        
        # Plot trajectories for each player
        for i, (player_id, positions) in enumerate(movements.items()):
            if i >= len(self.colors):
                break
                
            x_coords = [pos['position'][0] for pos in positions]
            y_coords = [pos['position'][1] for pos in positions]
            
            plt.plot(x_coords, y_coords, color=self.colors[i], 
                    linewidth=2, alpha=0.7, label=f'Player {player_id}')
            
            # Mark start and end points
            if x_coords and y_coords:
                plt.scatter(x_coords[0], y_coords[0], color=self.colors[i], 
                          s=100, marker='o', edgecolor='black', linewidth=2)
                plt.scatter(x_coords[-1], y_coords[-1], color=self.colors[i], 
                          s=100, marker='s', edgecolor='black', linewidth=2)
        
        plt.title('Player Movement Trajectories', fontsize=16, fontweight='bold')
        plt.xlabel('X Position (pixels)', fontsize=12)
        plt.ylabel('Y Position (pixels)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_zone_analysis(self, advanced_stats: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot zone time distribution for players.
        
        Args:
            advanced_stats: Advanced statistics data
            save_path: Optional path to save the plot
        """
        players = list(advanced_stats.keys())
        zones = ['offensive', 'neutral', 'defensive']
        
        # Prepare data for stacked bar chart
        zone_data = {zone: [] for zone in zones}
        for player_id in players:
            zone_time = advanced_stats[player_id]['zone_time']
            for zone in zones:
                zone_data[zone].append(zone_time[zone])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bottom = np.zeros(len(players))
        colors = ['#ff4444', '#ffaa44', '#4444ff']  # Red, Yellow, Blue
        
        for i, zone in enumerate(zones):
            ax.bar(range(len(players)), zone_data[zone], bottom=bottom, 
                  label=f'{zone.title()} Zone', color=colors[i], alpha=0.8)
            bottom += zone_data[zone]
        
        ax.set_title('Player Zone Time Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Player ID', fontsize=12)
        ax.set_ylabel('Time Percentage (%)', fontsize=12)
        ax.set_xticks(range(len(players)))
        ax.set_xticklabels([f'Player {p}' for p in players], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_speed_analysis(self, advanced_stats: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot player speed analysis.
        
        Args:
            advanced_stats: Advanced statistics data
            save_path: Optional path to save the plot
        """
        players = list(advanced_stats.keys())
        avg_speeds = [stats['average_speed'] for stats in advanced_stats.values()]
        max_speeds = [stats['max_speed'] for stats in advanced_stats.values()]
        
        x = np.arange(len(players))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars1 = ax.bar(x - width/2, avg_speeds, width, label='Average Speed', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, max_speeds, width, label='Max Speed', 
                      color='orange', alpha=0.8)
        
        ax.set_title('Player Speed Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Player ID', fontsize=12)
        ax.set_ylabel('Speed (pixels/frame)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Player {p}' for p in players], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, analysis: Dict, save_path: Optional[str] = None) -> None:
        """
        Create interactive Plotly dashboard.
        
        Args:
            analysis: Complete analysis results
            save_path: Optional path to save HTML file
        """
        advanced_stats = analysis['advanced_statistics']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ice Time Distribution', 'Speed Analysis', 
                          'Zone Time Distribution', 'Player Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        players = list(advanced_stats.keys())
        
        # Ice time plot
        ice_times = [stats['ice_time_seconds'] for stats in advanced_stats.values()]
        fig.add_trace(
            go.Bar(x=[f'Player {p}' for p in players], y=ice_times, 
                  name='Ice Time', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Speed analysis
        avg_speeds = [stats['average_speed'] for stats in advanced_stats.values()]
        max_speeds = [stats['max_speed'] for stats in advanced_stats.values()]
        
        fig.add_trace(
            go.Bar(x=[f'Player {p}' for p in players], y=avg_speeds, 
                  name='Avg Speed', marker_color='orange'),
            row=1, col=2
        )
        
        # Zone distribution (stacked bar)
        offensive = [stats['zone_time']['offensive'] for stats in advanced_stats.values()]
        neutral = [stats['zone_time']['neutral'] for stats in advanced_stats.values()]
        defensive = [stats['zone_time']['defensive'] for stats in advanced_stats.values()]
        
        fig.add_trace(
            go.Bar(x=[f'Player {p}' for p in players], y=offensive, 
                  name='Offensive Zone', marker_color='red'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=[f'Player {p}' for p in players], y=neutral, 
                  name='Neutral Zone', marker_color='yellow'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=[f'Player {p}' for p in players], y=defensive, 
                  name='Defensive Zone', marker_color='blue'),
            row=2, col=1
        )
        
        # Performance scatter (ice time vs speed)
        fig.add_trace(
            go.Scatter(x=ice_times, y=avg_speeds, 
                      mode='markers+text',
                      text=[f'P{p}' for p in players],
                      textposition='top center',
                      marker=dict(size=10, color='green'),
                      name='Performance'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Hockey Player Tracking Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Players", row=1, col=1)
        fig.update_yaxes(title_text="Ice Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Players", row=1, col=2)
        fig.update_yaxes(title_text="Speed", row=1, col=2)
        fig.update_xaxes(title_text="Players", row=2, col=1)
        fig.update_yaxes(title_text="Zone Time (%)", row=2, col=1)
        fig.update_xaxes(title_text="Ice Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Avg Speed", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def create_tracking_video_overlay(self, frame: np.ndarray, tracks: List[Dict], 
                                    frame_number: int) -> np.ndarray:
        """
        Create video frame with tracking overlays.
        
        Args:
            frame: Original video frame
            tracks: List of player tracks for this frame
            frame_number: Current frame number
            
        Returns:
            Annotated frame with tracking information
        """
        annotated_frame = frame.copy()
        
        # Add frame number
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for track in tracks:
            player_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get consistent color for player
            color_idx = player_id % len(self.colors)
            color_hex = self.colors[color_idx].lstrip('#')
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))  # Convert to BGR
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Player info box
            info_y = y1 - 40
            cv2.rectangle(annotated_frame, (x1, info_y), (x1 + 150, y1), color_bgr, -1)
            
            # Player text
            cv2.putText(annotated_frame, f"Player {player_id}", (x1 + 5, info_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Ice: {track['on_ice_time']:.1f}s", (x1 + 5, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 3, color_bgr, -1)
        
        return annotated_frame


if __name__ == "__main__":
    # Example usage with dummy data
    
    # Create sample data
    dummy_stats = {
        1: {'total_ice_time': 245.5, 'frames_detected': 450},
        2: {'total_ice_time': 198.2, 'frames_detected': 380},
        3: {'total_ice_time': 312.8, 'frames_detected': 520},
        4: {'total_ice_time': 156.4, 'frames_detected': 290}
    }
    
    dummy_advanced = {
        1: {
            'ice_time_seconds': 245.5,
            'average_speed': 12.5,
            'max_speed': 35.2,
            'zone_time': {'offensive': 35.0, 'neutral': 40.0, 'defensive': 25.0}
        },
        2: {
            'ice_time_seconds': 198.2,
            'average_speed': 15.8,
            'max_speed': 42.1,
            'zone_time': {'offensive': 45.0, 'neutral': 30.0, 'defensive': 25.0}
        }
    }
    
    # Create visualizer
    viz = HockeyVisualizer()
    
    # Create plots
    viz.plot_player_ice_time(dummy_stats)
    viz.plot_zone_analysis(dummy_advanced)
    viz.plot_speed_analysis(dummy_advanced)
