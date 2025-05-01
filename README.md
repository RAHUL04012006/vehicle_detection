# vehicle_detection
# Vehicle Detection Project  This project implements real-time vehicle detection from video using YOLOv5 and OpenCV. 

# Vehicle Detection Project

This project implements real-time vehicle detection from video using YOLOv5 and OpenCV. It can detect and count different types of vehicles in a video stream.

## Features

- Real-time vehicle detection and classification
- Vehicle counting by category
- Live display of detection results
- Support for both YOLOv5m and YOLOv5s models

## Prerequisites

- Python 3.6+
- OpenCV
- PyTorch
- NumPy

## Installation

1. Clone this repository
2. Install the required dependencies:
```sh
pip install opencv-python torch numpy
```

## Usage

1. Ensure you have a video file named `vehicle.mp4` in the project directory
2. Run the detection script:
```sh
python "vehicle detection.py"
```

## Controls

- Press 'q' to quit the detection program

## Files

- `vehicle detection.py` - Main detection script
- `vehicle.mp4` - Input video file
- `yolov5m.pt` - YOLOv5 medium model weights
- `yolov5s.pt` - YOLOv5 small model weights

## Output

The program displays:
- Real-time video feed with vehicle detection boxes
- Total vehicle count
- Individual counts for each vehicle category
