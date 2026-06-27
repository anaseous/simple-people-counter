import os
import time
import threading
from pathlib import Path

import cv2
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
MAX_DISPLAY_WIDTH = int(os.getenv("MAX_DISPLAY_WIDTH", "1280"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

model = YOLO(MODEL_PATH)

current_processor = None
processor_lock = threading.Lock()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class PeopleCounterProcessor:
    def __init__(self, source, source_type):
        self.source = source
        self.source_type = source_type
        self.stop_flag = False

        self.current_count = 0
        self.max_count = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.fps = 0.0
        self.status = "Starting"

        self.last_boxes = []
        self.last_confidences = []

        self.lock = threading.Lock()

    def stop(self):
        with self.lock:
            self.stop_flag = True
            self.status = "Stopped"

    def should_stop(self):
        with self.lock:
            return self.stop_flag

    def get_stats(self):
        with self.lock:
            return {
                "status": self.status,
                "source_type": self.source_type,
                "current_count": self.current_count,
                "max_count": self.max_count,
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "fps": round(self.fps, 2),
            }

    def resize_frame(self, frame):
        height, width = frame.shape[:2]

        if width <= MAX_DISPLAY_WIDTH:
            return frame

        scale = MAX_DISPLAY_WIDTH / width
        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(frame, (new_width, new_height))

    def detect_people(self, frame):
        results = model.predict(
            source=frame,
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        boxes = []
        confidences = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())

                boxes.append((x1, y1, x2, y2))
                confidences.append(confidence)

        return boxes, confidences

    def draw_overlay(self, frame, boxes, confidences):
        count = len(boxes)

        for index, (box, confidence) in enumerate(zip(boxes, confidences), start=1):
            x1, y1, x2, y2 = box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 80), 2)

            label = f"Person {index}: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 80),
                2,
                cv2.LINE_AA
            )

        header = f"People Count: {count}"
        cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            header,
            (25, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    def open_capture(self):
        if self.source_type == "rtsp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)

        return cap

    def generate_frames(self):
        cap = self.open_capture()

        if not cap.isOpened():
            with self.lock:
                self.status = "Error: Cannot open video source"
            return

        with self.lock:
            self.status = "Running"

        last_time = time.time()
        frame_counter_for_fps = 0

        while not self.should_stop():
            success, frame = cap.read()

            if not success:
                with self.lock:
                    if self.source_type == "upload":
                        self.status = "Completed"
                    else:
                        self.status = "Stream disconnected or unavailable"
                break

            self.total_frames += 1
            frame_counter_for_fps += 1

            frame = self.resize_frame(frame)

            if self.total_frames % FRAME_SKIP == 0:
                boxes, confidences = self.detect_people(frame)

                with self.lock:
                    self.last_boxes = boxes
                    self.last_confidences = confidences
                    self.current_count = len(boxes)
                    self.max_count = max(self.max_count, self.current_count)
                    self.processed_frames += 1
            else:
                boxes = self.last_boxes
                confidences = self.last_confidences

            now = time.time()
            elapsed = now - last_time

            if elapsed >= 1.0:
                with self.lock:
                    self.fps = frame_counter_for_fps / elapsed

                frame_counter_for_fps = 0
                last_time = now

            frame = self.draw_overlay(frame, boxes, confidences)

            ok, buffer = cv2.imencode(".jpg", frame)

            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start_rtsp", methods=["POST"])
def start_rtsp():
    global current_processor

    data = request.get_json(silent=True) or {}
    rtsp_url = data.get("rtsp_url", "").strip()

    if not rtsp_url:
        return jsonify({"success": False, "message": "RTSP URL is required"}), 400

    if not rtsp_url.lower().startswith("rtsp://"):
        return jsonify({"success": False, "message": "Invalid RTSP URL. It must start with rtsp://"}), 400

    with processor_lock:
        if current_processor:
            current_processor.stop()

        current_processor = PeopleCounterProcessor(rtsp_url, "rtsp")

    return jsonify({"success": True, "message": "RTSP stream started"})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    global current_processor

    if "video" not in request.files:
        return jsonify({"success": False, "message": "No video file uploaded"}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Unsupported file type. Use mp4, avi, mov, mkv, or webm."
        }), 400

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    saved_filename = f"{timestamp}_{filename}"
    saved_path = UPLOAD_DIR / saved_filename

    file.save(str(saved_path))

    with processor_lock:
        if current_processor:
            current_processor.stop()

        current_processor = PeopleCounterProcessor(str(saved_path), "upload")

    return jsonify({"success": True, "message": "Video uploaded and processing started"})


@app.route("/video_feed")
def video_feed():
    global current_processor

    if not current_processor:
        return "No active video source. Start RTSP or upload a video first.", 400

    return Response(
        current_processor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stats")
def stats():
    global current_processor

    if not current_processor:
        return jsonify({
            "status": "Idle",
            "source_type": "-",
            "current_count": 0,
            "max_count": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "fps": 0.0
        })

    return jsonify(current_processor.get_stats())


@app.route("/api/stop", methods=["POST"])
def stop():
    global current_processor

    with processor_lock:
        if current_processor:
            current_processor.stop()
            current_processor = None

    return jsonify({"success": True, "message": "Processing stopped"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
