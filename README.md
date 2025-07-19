# Hockey Player Tracking System

A computer vision-based system for tracking hockey players in game videos using open source tools.

## Features

- **Player Detection**: Identify and detect hockey players in video frames
- **Player Tracking**: Track individual players across video sequences
- **Statistics Extraction**: Automatically track goals, assists, and ice time
- **Video Analysis**: Process hockey game videos frame by frame
- **Real-time Processing**: Optimized for live game analysis

## Technology Stack

- **Computer Vision**: OpenCV, YOLO v8
- **Object Tracking**: DeepSORT, ByteTrack
- **Pose Estimation**: MediaPipe
- **Machine Learning**: scikit-learn, ultralytics
- **Video Processing**: FFmpeg
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Visualization**: matplotlib, plotly

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Player Detection**:
   ```python
   from src.player_detector import PlayerDetector
   detector = PlayerDetector()
   players = detector.detect_players(video_path)
   ```

2. **Player Tracking**:
   ```python
   from src.player_tracker import PlayerTracker
   tracker = PlayerTracker()
   tracked_players = tracker.track_players(video_path)
   ```

3. **Statistics Analysis**:
   ```python
   from src.stats_analyzer import StatsAnalyzer
   analyzer = StatsAnalyzer()
   stats = analyzer.extract_stats(tracked_data)
   ```

## Project Structure

```
hockeytracker/
├── src/
│   ├── player_detector.py      # YOLO-based player detection
│   ├── player_tracker.py       # Multi-object tracking
│   ├── stats_analyzer.py       # Statistics extraction
│   ├── video_processor.py      # Video processing utilities
│   └── utils/
│       ├── visualization.py    # Plotting and visualization
│       └── data_processing.py  # Data manipulation utilities
├── models/                     # Pre-trained models
├── data/                       # Sample videos and datasets
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
└── requirements.txt
```

## Getting Started

1. Place your hockey game videos in the `data/videos/` directory
2. Run the player detection: `python src/player_detector.py`
3. Analyze tracking results: `python src/stats_analyzer.py`
4. View results in the generated visualizations

