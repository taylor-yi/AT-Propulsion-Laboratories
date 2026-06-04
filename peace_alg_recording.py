import time
import os
from datetime import datetime
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# --- frame setup ---
W, H = 640, 480
CENTER_X, CENTER_Y = W // 2, H // 2
DEADZONE_X = int(W * 0.12)
DEADZONE_Y = int(H * 0.12)
HOLD_TIME = 2.0
FPS = 10  # match this roughly to your real inference rate

# --- output file: new timestamped file every boot ---
SAVE_DIR = "/home/peace/flights"
os.makedirs(SAVE_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(SAVE_DIR, f"flight_{stamp}.mp4")

picam2 = Picamera2()
picam2.preview_configuration.main.size = (W, H)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(1)  # let the sensor settle before recording

model = YOLO("yolo26n.pt")
print("measuring capture rate...")
t0 = time.monotonic()
N = 20
for _ in range(N):
    frame = picam2.capture_array()
    _ = model(frame, classes=[0], conf=0.5, verbose=False)
measured_fps = N / (time.monotonic() - t0)
print(f"measured ~{measured_fps:.2f} FPS")
measured_fps = N / (time.monotonic() - t0)
measured_fps = max(1.0, min(measured_fps, 30.0))   # keep it sane: 1–30
print(f"measured ~{measured_fps:.2f} FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, measured_fps, (W, H))
if not writer.isOpened():
    print("mp4v failed, falling back to MJPG/.avi")
    out_path = out_path.replace(".mp4", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, measured_fps, (W, H))
print(f"Recording to: {out_path}  (writer open: {writer.isOpened()})")

centered_since = None

try:
    while True:
        frame = picam2.capture_array()
        results = model(frame, classes=[0], conf=0.5, verbose=False)
        boxes = results[0].boxes
        annotated = results[0].plot()

        cv2.drawMarker(annotated, (CENTER_X, CENTER_Y), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.rectangle(annotated,
                      (CENTER_X - DEADZONE_X, CENTER_Y - DEADZONE_Y),
                      (CENTER_X + DEADZONE_X, CENTER_Y + DEADZONE_Y),
                      (255, 255, 255), 1)

        if len(boxes) > 0:
            best = boxes[boxes.conf.argmax()]
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)
            dx, dy = px - CENTER_X, py - CENTER_Y

            cv2.line(annotated, (CENTER_X, CENTER_Y), (px, py), (0, 255, 255), 2)
            cv2.circle(annotated, (px, py), 6, (0, 255, 255), -1)

            horiz = "RIGHT" if dx > DEADZONE_X else "LEFT" if dx < -DEADZONE_X else ""
            vert  = "DOWN"  if dy > DEADZONE_Y else "UP"   if dy < -DEADZONE_Y else ""

            if horiz or vert:
                centered_since = None
                cmd = f"MOVE {vert} {horiz}".strip().replace("  ", " ")
                color = (0, 255, 0)
            else:
                if centered_since is None:
                    centered_since = time.monotonic()
                held = time.monotonic() - centered_since
                if held >= HOLD_TIME:
                    cmd = 'CENTERED, DELIVER "FOOD"'
                    color = (0, 200, 255)
                else:
                    cmd = f"CENTERED  ({held:.1f}s)"
                    color = (0, 255, 0)

            cv2.putText(annotated, cmd, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(annotated, f"dx={dx:+d} dy={dy:+d}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            centered_since = None
            cv2.putText(annotated, "no person", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # timestamp burned into the recording
        cv2.putText(annotated, datetime.now().strftime("%H:%M:%S"),
                    (W - 110, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        writer.write(annotated)
finally:
    writer.release()
    picam2.stop()
