# Hockey Player Tracking System

A computer vision-based system for tracking hockey players in game videos using open source tools.

## Features

- **Player Detection**: Identify and detect hockey players in video frames
- **Player Tracking**: Track individual players across video sequences
- **Statistics Extraction**: Automatically track goals, assists, and ice time
- **Video Analysis**: Process hockey game videos frame by frame
- **Real-time Processing**: Optimized for live game analysis
- **Streamlit Interface**: Web-based UI for easy uploading, processing, and results download
- **Export Options**: Download all results as a single ZIP file

## Technology Stack

- **Computer Vision**: OpenCV, YOLO v8
- **Object Tracking**: DeepSORT, ByteTrack
- **Pose Estimation**: MediaPipe
- **Machine Learning**: scikit-learn, ultralytics
- **Video Processing**: FFmpeg
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Visualization**: matplotlib, plotly
- **Web Interface**: Streamlit

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit Web Interface

The easiest way to use the Hockey Tracker is through the Streamlit web interface:

```bash
# Run the Streamlit app
streamlit run app.py
```

This will open a web browser with the interface where you can:
1. Upload hockey game videos
2. Adjust detection and analysis settings
3. Process videos and view results
4. Download all results as a single ZIP file

### Python API

You can also use the individual components programmatically:

## Project Structure

```
hockeytracker/
├── app.py                      # Streamlit web interface
├── src/
│   ├── player_detector.py      # YOLO-based player detection
│   ├── player_tracker.py       # Multi-object tracking
│   ├── stats_analyzer.py       # Statistics extraction
│   ├── video_processor.py      # Video processing utilities
│   └── utils/
│       ├── visualization.py    # Plotting and visualization
│       ├── data_processing.py  # Data manipulation utilities
│       └── file_export.py      # File export and download utilities
├── models/                     # Pre-trained models
├── data/                       # Sample videos and datasets
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
└── requirements.txt
```

## Getting Started

### Using the Web Interface

1. Start the Streamlit app: `streamlit run app.py`
2. Upload your hockey game video through the web interface
3. Adjust settings in the sidebar (detection confidence, visualization options, etc.)
4. Click "Process Video" to start the analysis
5. View player statistics and visualizations in the interface
6. Download all results as a ZIP file

### Using the Command Line

1. Place your hockey game videos in the `data/videos/` directory
2. Run the player detection: `python -m src.player_detector`
3. Analyze tracking results: `python -m src.stats_analyzer`
4. View results in the generated visualizations

## Python API Usage

### Player Detection

```python
from src.player_detector import PlayerDetector

# Initialize the detector with a YOLO model
detector = PlayerDetector(
    model_path="yolov8n.pt",
    confidence_threshold=0.5
)

# Detect players in a video frame
frame = video_processor.get_frame(video_path, frame_idx)
detections = detector.detect(frame)
```

### Player Tracking

```python
from src.player_tracker import PlayerTracker

# Initialize the tracker
tracker = PlayerTracker(
    max_age=30,
    min_hits=3
)

# Track players through a video
tracking_results = tracker.track_video(
    video_path,
    detector,
    output_path="tracked_video.mp4"  # Optional
)
```

### Statistics Analysis

```python
from src.stats_analyzer import StatsAnalyzer

# Initialize the analyzer
analyzer = StatsAnalyzer()

# Analyze tracking data
analysis = analyzer.analyze_tracking_data(tracking_results)

# Export statistics to different formats
analyzer.export_statistics(analysis, "stats.csv", format="csv")
analyzer.export_statistics(analysis, "stats.json", format="json")
```

### Visualization

```python
from src.utils.visualization import HockeyVisualizer

# Initialize the visualizer
visualizer = HockeyVisualizer()

# Create visualizations
visualizer.plot_player_ice_time(tracking_results['player_statistics'], "ice_time.png")
visualizer.plot_speed_analysis(analysis['advanced_statistics'], "speed.png")
visualizer.plot_zone_analysis(analysis['advanced_statistics'], "zones.png")
```

## File Export and Download

The system includes utilities for exporting and downloading results:

```python
from src.utils.file_export import (
    create_temp_output_folder,
    create_zip_download_button
)

# Create a temporary folder for output files
output_folder = create_temp_output_folder()

# Save results to the output folder
# ...

# Create a download button for all files as a ZIP
create_zip_download_button(
    output_folder,
    "Download All Results",
    "hockey_tracking_results.zip"
)
```



