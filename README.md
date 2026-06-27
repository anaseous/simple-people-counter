# Simple People Counter

A simple local web application that counts visible people from:

- RTSP camera streams
- Uploaded video files

Built with:

- Python
- Flask
- OpenCV
- Ultralytics YOLO

## Features

- RTSP stream input
- Video file upload
- Real-time browser preview
- Person detection bounding boxes
- Current people count
- Maximum people count
- FPS display
- Processed frame count

## Important Note

This tool counts visible people per frame. It does not identify people, track identity, or perform facial recognition.

For production use, make sure you follow privacy, security, and local legal requirements before processing camera footage.

## Repository Structure

```text
simple-people-counter/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    ├── app.js
    └── style.css
