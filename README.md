# 🎥 Simple People Counter

A modern web application that counts visible people from:

✅ RTSP Camera Streams

✅ Uploaded Video Files

✅ Real-Time Browser Dashboard

✅ YOLO AI Person Detection

✅ Live Statistics

---

## Features

- RTSP Camera Support
- Video Upload Support
- Real-Time People Detection
- Live People Count
- Maximum People Count
- FPS Monitoring
- Processed Frame Statistics
- Browser-Based Interface
- YOLOv8 AI Detection
- OpenCV Video Processing

---

## Screenshots

### Main Dashboard

> After running the application, you will see:

![Dashboard](https://via.placeholder.com/1200x700.png?text=Simple+People+Counter+Dashboard)

---

### RTSP Camera Example

![RTSP Example](https://via.placeholder.com/1200x700.png?text=RTSP+Camera+People+Counting)

---

### Video Upload Example

![Video Upload](https://via.placeholder.com/1200x700.png?text=Uploaded+Video+People+Detection)

---

### Detection Output

![Detection Output](https://via.placeholder.com/1200x700.png?text=People+Detected+with+Bounding+Boxes)

---

# Repository Structure

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
├── static/
│   ├── app.js
│   └── style.css
│
└── uploads/
```

---

# Prerequisites

Install:

- Python 3.10+
- Git

Verify installation:

```bash
python --version
git --version
```

Expected output:

```text
Python 3.12.x
git version 2.x.x
```

---

# Installation (Windows)

Open PowerShell:

```powershell
git clone https://github.com/anaseous/simple-people-counter.git

cd simple-people-counter
```

Create virtual environment:

```powershell
python -m venv venv
```

Activate virtual environment:

```powershell
venv\Scripts\activate
```

Expected:

```text
(venv) PS C:\simple-people-counter>
```

Install required packages:

```powershell
pip install -r requirements.txt
```

Verify installation:

```powershell
pip list
```

---

# Installation (Linux)

```bash
git clone https://github.com/anaseous/simple-people-counter.git

cd simple-people-counter

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

# Installation (macOS)

```bash
git clone https://github.com/anaseous/simple-people-counter.git

cd simple-people-counter

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

# Start the Application

Run:

```bash
python app.py
```

Expected:

```text
 * Running on http://127.0.0.1:5000
```

---

# Open Browser

Open:

```text
http://localhost:5000
```

or

```text
http://127.0.0.1:5000
```

---

# RTSP Camera Example

Example format:

```text
rtsp://username:password@192.168.1.100:554/stream1
```

Example only:

```text
rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/101
```

Example Hikvision:

```text
rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/101
```

Example Dahua:

```text
rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

Example Axis:

```text
rtsp://root:password123@192.168.1.100/axis-media/media.amp
```

⚠️ Do NOT upload real RTSP credentials to GitHub.

---

# Upload Video Example

Supported formats:

```text
mp4
avi
mov
mkv
webm
```

Example videos:

```text
people_walking.mp4

shopping_mall.mp4

office_entrance.mp4

airport_terminal.mp4

factory_workers.mp4
```

---

# Sample Test Data

Create a folder:

```text
test_videos
```

Add:

```text
test_videos/
│
├── airport.mp4
├── shopping_mall.mp4
├── office.mp4
└── warehouse.mp4
```

Use any public sample people-counting video.

---

# Configuration

Default values:

```bash
MODEL_PATH=yolov8n.pt

CONFIDENCE_THRESHOLD=0.35

FRAME_SKIP=2

MAX_DISPLAY_WIDTH=1280
```

---

# Run With Custom Configuration

Windows:

```powershell
set CONFIDENCE_THRESHOLD=0.45

set FRAME_SKIP=3

python app.py
```

Linux:

```bash
CONFIDENCE_THRESHOLD=0.45 FRAME_SKIP=3 python app.py
```

---

# Application Workflow

```text
User
  │
  ▼
Enter RTSP URL
OR
Upload Video
  │
  ▼
Flask Backend
  │
  ▼
OpenCV Reads Frames
  │
  ▼
YOLO Detects Persons
  │
  ▼
Count Visible Persons
  │
  ▼
Display Bounding Boxes
  │
  ▼
Stream Results To Browser
```

---

# How It Works

1. User enters an RTSP URL or uploads a video.

2. Flask receives the request.

3. OpenCV opens the stream or file.

