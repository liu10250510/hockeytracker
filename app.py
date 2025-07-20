"""
Streamlit app for Hockey Player Tracking.

This app provides a web interface for uploading hockey videos,
processing them, and downloading the results.
"""

import os
import streamlit as st
import tempfile
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from player_detector import PlayerDetector
from player_tracker import PlayerTracker
from stats_analyzer import StatsAnalyzer
from video_processor import VideoProcessor
from utils.visualization import HockeyVisualizer
from utils.data_processing import DataProcessor
from utils.file_export import (
    create_temp_output_folder,
    create_zip_download_button,
    cleanup_temp_folder
)

st.set_page_config(
    page_title="Hockey Player Tracker",
    page_icon="🏒",
    layout="wide"
)

def main():
    st.title("Hockey Player Tracking System 🏒")
    st.write("Upload a hockey game video to track players and analyze their statistics.")
    
    # Set up sidebar
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    create_visualizations = st.sidebar.checkbox("Create Visualizations", True)
    save_video = st.sidebar.checkbox("Save Tracked Video", False)
    export_format = st.sidebar.selectbox(
        "Export Format", 
        options=["json", "csv", "excel", "all"],
        index=1
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Hockey Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create temp directory for output
        output_dir = create_temp_output_folder()
        
        # Save uploaded video to temporary file
        temp_video_path = os.path.join(output_dir, "input_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.success(f"Video uploaded successfully!")
        
        # Initialize components
        st.info("Initializing tracking components...")
        
        video_processor = VideoProcessor()
        detector = PlayerDetector(
            model_path="yolov8n.pt",
            confidence_threshold=confidence
        )
        tracker = PlayerTracker(
            max_age=30,
            min_hits=3
        )
        analyzer = StatsAnalyzer()
        visualizer = HockeyVisualizer()
        data_processor = DataProcessor()
        
        # Display video info
        if video_processor.validate_video(temp_video_path):
            video_info = video_processor.get_video_info(temp_video_path)
            st.write("Video Information:")
            st.write(f"- Resolution: {video_info['width']}x{video_info['height']}")
            st.write(f"- Duration: {video_info['duration']:.2f} seconds")
            st.write(f"- Frame Rate: {video_info['fps']:.1f} fps")
            
            # Process button
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    try:
                        # Set up output video path if needed
                        output_video_path = os.path.join(output_dir, "tracked_players.mp4") if save_video else None
                        
                        # Run tracking
                        progress_bar = st.progress(0)
                        st.info("Tracking players...")
                        
                        tracking_results = tracker.track_video(
                            temp_video_path,
                            detector,
                            output_path=output_video_path
                        )
                        
                        progress_bar.progress(50)
                        
                        # Analyze results
                        st.info("Analyzing tracking data...")
                        analysis = analyzer.analyze_tracking_data(tracking_results)
                        
                        progress_bar.progress(75)
                        
                        # Export statistics
                        st.info("Exporting results...")
                        
                        if export_format == 'all':
                            formats = ['json', 'csv', 'excel']
                        else:
                            formats = [export_format]
                        
                        for fmt in formats:
                            output_path = os.path.join(output_dir, f"hockey_analysis.{fmt}")
                            analyzer.export_statistics(analysis, output_path, fmt)
                        
                        # Create visualizations
                        if create_visualizations:
                            st.info("Creating visualizations...")
                            
                            # Ice time plot
                            ice_time_path = os.path.join(output_dir, "ice_time_analysis.png")
                            visualizer.plot_player_ice_time(tracking_results['player_statistics'], ice_time_path)
                            
                            if 'advanced_statistics' in analysis:
                                # Speed analysis
                                speed_path = os.path.join(output_dir, "speed_analysis.png")
                                visualizer.plot_speed_analysis(analysis['advanced_statistics'], speed_path)
                                
                                # Zone analysis
                                zone_path = os.path.join(output_dir, "zone_analysis.png")
                                visualizer.plot_zone_analysis(analysis['advanced_statistics'], zone_path)
                        
                        progress_bar.progress(100)
                        
                        # Display results
                        st.success("Processing complete!")
                        
                        # Display summary
                        st.subheader("Player Statistics")
                        player_stats = tracking_results['player_statistics']
                        st.write(f"Players Detected: {len(player_stats)}")
                        
                        # Create columns for stats
                        col1, col2, col3 = st.columns(3)
                        
                        # Display player info
                        player_ids = list(player_stats.keys())
                        selected_player = col1.selectbox("Select Player", player_ids)
                        
                        if selected_player:
                            stats = player_stats[selected_player]
                            col2.metric("Ice Time (seconds)", f"{stats['total_ice_time']:.2f}")
                            col3.metric("Goals", stats['goals'])
                            col3.metric("Assists", stats['assists'])
                            
                            if 'advanced_statistics' in analysis and selected_player in analysis['advanced_statistics']:
                                adv_stats = analysis['advanced_statistics'][selected_player]
                                col2.metric("Average Speed", f"{adv_stats['average_speed']:.2f}")
                        
                        # Show tracked video if available
                        if save_video and os.path.exists(output_video_path):
                            st.subheader("Tracked Video")
                            st.video(output_video_path)
                        
                        # Add zip download option for all files
                        st.subheader("Download Results")
                        create_zip_download_button(
                            output_dir,
                            "Download All Results",
                            "hockey_tracking_results.zip"
                        )
                        
                        # Register cleanup function
                        import atexit
                        atexit.register(cleanup_temp_folder, output_dir)
                    
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        else:
            st.error("Invalid video file. Please upload a valid video.")

if __name__ == "__main__":
    main()
