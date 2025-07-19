"""
Main application script for hockey player tracking.

This script provides a command-line interface to run the complete
hockey player tracking pipeline.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from player_detector import PlayerDetector
from player_tracker import PlayerTracker
from stats_analyzer import StatsAnalyzer
from video_processor import VideoProcessor
from utils.visualization import HockeyVisualizer
from utils.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Hockey Player Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video data/videos/game.mp4
  python main.py --video data/videos/game.mp4 --output data/output/ --visualize
  python main.py --video data/videos/game.mp4 --confidence 0.6 --export-format csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to input hockey game video'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        default='data/output/',
        help='Output directory for results (default: data/output/)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help='YOLO model path (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--max-age',
        type=int,
        default=30,
        help='Maximum age for tracks without updates (default: 30)'
    )
    
    parser.add_argument(
        '--min-hits',
        type=int,
        default=3,
        help='Minimum hits to consider track valid (default: 3)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--export-format',
        choices=['json', 'csv', 'excel', 'all'],
        default='json',
        help='Export format for statistics (default: json)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Create interactive dashboard'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save annotated video with tracking results'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing tracking components...")
    
    video_processor = VideoProcessor()
    detector = PlayerDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    tracker = PlayerTracker(
        max_age=args.max_age,
        min_hits=args.min_hits
    )
    analyzer = StatsAnalyzer()
    visualizer = HockeyVisualizer()
    data_processor = DataProcessor()
    
    # Validate video
    if not video_processor.validate_video(args.video):
        logger.error("Invalid video file")
        return 1
    
    # Get video info
    video_info = video_processor.get_video_info(args.video)
    logger.info(f"Processing video: {video_info['width']}x{video_info['height']} "
                f"@ {video_info['fps']:.1f}fps, {video_info['duration']:.2f}s")
    
    try:
        # Run tracking
        logger.info("Starting player tracking...")
        output_video_path = os.path.join(args.output, "tracked_players.mp4") if args.save_video else None
        
        tracking_results = tracker.track_video(
            args.video,
            detector,
            output_path=output_video_path
        )
        
        logger.info("Tracking completed successfully!")
        
        # Analyze results
        logger.info("Analyzing tracking data...")
        analysis = analyzer.analyze_tracking_data(tracking_results)
        
        # Print summary
        print_summary(tracking_results, analysis)
        
        # Export statistics
        export_statistics(analyzer, analysis, args.output, args.export_format)
        
        # Create visualizations
        if args.visualize:
            create_visualizations(visualizer, tracking_results, analysis, args.output)
        
        # Create interactive dashboard
        if args.interactive:
            create_dashboard(visualizer, analysis, args.output)
        
        # Process and export data
        export_processed_data(data_processor, tracking_results, args.output)
        
        logger.info(f"All results saved to: {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


def print_summary(tracking_results: dict, analysis: dict):
    """Print analysis summary to console."""
    print("\n" + "="*60)
    print("HOCKEY PLAYER TRACKING RESULTS")
    print("="*60)
    
    # Video summary
    video_info = tracking_results['video_info']
    print(f"Video Duration: {video_info['duration']:.2f} seconds")
    print(f"Total Frames: {video_info['total_frames']}")
    print(f"Frame Rate: {video_info['fps']:.1f} fps")
    
    # Player summary
    player_stats = tracking_results['player_statistics']
    print(f"\nPlayers Detected: {len(player_stats)}")
    
    if 'video_summary' in analysis:
        summary = analysis['video_summary']
        print(f"Average Ice Time: {summary['average_ice_time_per_player']:.2f}s")
    
    # Individual player stats
    print("\nINDIVIDUAL PLAYER STATISTICS:")
    print("-" * 40)
    
    for player_id, stats in player_stats.items():
        print(f"Player {player_id}:")
        print(f"  Ice Time: {stats['total_ice_time']:.2f}s")
        print(f"  Goals: {stats['goals']}")
        print(f"  Assists: {stats['assists']}")
        print(f"  Detections: {stats['frames_detected']}")
        
        # Add advanced stats if available
        if 'advanced_statistics' in analysis and player_id in analysis['advanced_statistics']:
            adv_stats = analysis['advanced_statistics'][player_id]
            print(f"  Avg Speed: {adv_stats['average_speed']:.2f} px/frame")
            zone_time = adv_stats['zone_time']
            print(f"  Zone Distribution: O:{zone_time['offensive']:.1f}% "
                  f"N:{zone_time['neutral']:.1f}% D:{zone_time['defensive']:.1f}%")
        print()


def export_statistics(analyzer: StatsAnalyzer, analysis: dict, output_dir: str, export_format: str):
    """Export statistics in specified format(s)."""
    logger.info(f"Exporting statistics in {export_format} format...")
    
    if export_format == 'all':
        formats = ['json', 'csv', 'excel']
    else:
        formats = [export_format]
    
    for fmt in formats:
        output_path = os.path.join(output_dir, f"hockey_analysis.{fmt}")
        analyzer.export_statistics(analysis, output_path, fmt)
        logger.info(f"Statistics exported to: {output_path}")


def create_visualizations(visualizer: HockeyVisualizer, tracking_results: dict, 
                         analysis: dict, output_dir: str):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    # Ice time plot
    ice_time_path = os.path.join(output_dir, "ice_time_analysis.png")
    visualizer.plot_player_ice_time(tracking_results['player_statistics'], ice_time_path)
    
    # Advanced statistics plots
    if 'advanced_statistics' in analysis:
        # Speed analysis
        speed_path = os.path.join(output_dir, "speed_analysis.png")
        visualizer.plot_speed_analysis(analysis['advanced_statistics'], speed_path)
        
        # Zone analysis
        zone_path = os.path.join(output_dir, "zone_analysis.png")
        visualizer.plot_zone_analysis(analysis['advanced_statistics'], zone_path)
    
    # Movement visualizations
    if 'player_movements' in analysis:
        movements = analysis['player_movements']
        
        # Assume standard HD dimensions (update as needed)
        frame_width = 1920
        frame_height = 1080
        
        # Heatmap
        heatmap_path = os.path.join(output_dir, "movement_heatmap.png")
        visualizer.plot_player_movement_heatmap(movements, frame_width, frame_height, heatmap_path)
        
        # Trajectories
        trajectory_path = os.path.join(output_dir, "player_trajectories.png")
        visualizer.plot_player_trajectories(movements, frame_width, frame_height, trajectory_path)
    
    logger.info("Visualizations created successfully!")


def create_dashboard(visualizer: HockeyVisualizer, analysis: dict, output_dir: str):
    """Create interactive dashboard."""
    logger.info("Creating interactive dashboard...")
    
    dashboard_path = os.path.join(output_dir, "interactive_dashboard.html")
    visualizer.create_interactive_dashboard(analysis, dashboard_path)
    
    logger.info(f"Interactive dashboard created: {dashboard_path}")


def export_processed_data(data_processor: DataProcessor, tracking_results: dict, output_dir: str):
    """Export processed tracking data."""
    logger.info("Processing and exporting tracking data...")
    
    # Convert to DataFrame
    df = data_processor.tracking_to_dataframe(tracking_results)
    
    # Calculate velocities
    df_with_velocity = data_processor.calculate_velocities(df)
    
    # Smooth trajectories
    df_smooth = data_processor.smooth_trajectories(df_with_velocity)
    
    # Export processed data
    base_path = os.path.join(output_dir, "processed_tracking_data")
    exported_files = data_processor.export_to_formats(df_smooth, base_path)
    
    # Create time series data
    time_series = data_processor.create_time_series_data(df_smooth, interval_seconds=30)
    time_series_path = os.path.join(output_dir, "time_series_data.csv")
    time_series.to_csv(time_series_path, index=False)
    
    logger.info("Processed data exported successfully!")


if __name__ == "__main__":
    sys.exit(main())
