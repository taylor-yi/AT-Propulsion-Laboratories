import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# --- frame setup ---
W, H = 640, 480
CENTER_X, CENTER_Y = W // 2, H // 2

# "close enough" zone (pixels) so it doesn't nag when roughly centered
DEADZONE_X = int(W * 0.12)   # ~77 px left/right
DEADZONE_Y = int(H * 0.12)   # ~58 px up/down

HOLD_TIME = 2.0   # seconds centered before we "deliver"

picam2 = Picamera2()
picam2.preview_configuration.main.size = (W, H)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

#model = YOLO("yolo26n.pt")
model = YOLO("/home/peace/model/yolo26n_ncnn_model")
centered_since = None   # timestamp when centering began, or None

while True:
    frame = picam2.capture_array()

    # detect people only (class 0), min 50% confidence
    #results = model(frame, classes=[0], conf=0.5, verbose=False)
    results = model(frame, classes=[0], conf=0.5, imgsz=320, verbose=False)
    boxes = results[0].boxes

    annotated = results[0].plot()

    # draw frame center crosshair + deadzone box
    cv2.drawMarker(annotated, (CENTER_X, CENTER_Y), (255, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.rectangle(annotated,
                  (CENTER_X - DEADZONE_X, CENTER_Y - DEADZONE_Y),
                  (CENTER_X + DEADZONE_X, CENTER_Y + DEADZONE_Y),
                  (255, 255, 255), 1)

    if len(boxes) > 0:
        # pick the most confident person as the target
        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        px, py = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # offset from center: +x = person is right, +y = person is down
        dx = px - CENTER_X
        dy = py - CENTER_Y

        # draw a line from center to the person
        cv2.line(annotated, (CENTER_X, CENTER_Y), (px, py), (0, 255, 255), 2)
        cv2.circle(annotated, (px, py), 6, (0, 255, 255), -1)

        # build the command text
        horiz = "RIGHT" if dx > DEADZONE_X else "LEFT" if dx < -DEADZONE_X else ""
        vert  = "DOWN"  if dy > DEADZONE_Y else "UP"   if dy < -DEADZONE_Y else ""

        if horiz or vert:
            # not centered -> reset the hold timer
            centered_since = None
            cmd = f"MOVE {vert} {horiz}".strip().replace("  ", " ")
            color = (0, 255, 0)
        else:
            # centered -> start the timer if it isn't already running
            if centered_since is None:
                centered_since = time.monotonic()

            held = time.monotonic() - centered_since
            if held >= HOLD_TIME:
                cmd = 'CENTERED, DELIVER "FOOD"'
                color = (0, 200, 255)   # orange-ish to stand out
            else:
                cmd = f"CENTERED  ({held:.1f}s)"
                color = (0, 255, 0)

        cv2.putText(annotated, cmd, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, f"dx={dx:+d} dy={dy:+d}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        # no person in frame -> reset the hold timer
        centered_since = None
        cv2.putText(annotated, "no person", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Drone view", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
