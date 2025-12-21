# Real-Time Object Detection (Python 3.13)

This project implements real-time object detection using OpenCV's DNN module with the YOLOv4-Tiny model.  
It is fully compatible with Python 3.13 and does not require PyTorch.

## Features
- Real-time webcam detection
- YOLOv4-Tiny for fast inference
- Bounding boxes with class labels
- Lightweight and CPU-friendly
- Python 3.13 compatible

## Technologies Used
- Python 3.13
- OpenCV
- YOLOv4-Tiny
- NumPy

## Project Structure
object_detection_tracking/
│
├── main.py
├── coco.names
├── yolov4-tiny.cfg
├── yolov4-tiny.weights
└── README.md

## How to Run
```bash
pip install opencv-python numpy
python main.py
Press Q to exit.

Output

Live webcam feed

Detected objects with bounding boxes

Class labels displayed in real time

Author
Pragya Rajwanshi
## STEP 5: Initialize Git in VS Code

Open **VS Code terminal** inside your project folder:

```bash
git init