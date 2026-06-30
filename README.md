# OpenVisionKit

[![CI — Unit Tests](https://github.com/anurupborah2001/openvisionkit/actions/workflows/ci-unit.yml/badge.svg)](https://github.com/your-org/openvisionkit/actions/workflows/ci-unit.yml)
[![Security Scan](https://github.com/anurupborah2001/openvisionkit/actions/workflows/ci-security.yml/badge.svg)](https://github.com/your-org/openvisionkit/actions/workflows/ci-security.yml)
[![PyPI version](https://badge.fury.io/py/openvisionkit.svg)](https://pypi.org/p/openvisionkit)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OpenVisionKit** is a high-level Python computer vision library built on top of [MediaPipe](https://developers.google.com/mediapipe) and [OpenCV](https://opencv.org/). It provides production-ready detectors and segmentation utilities for face detection, face mesh, hand tracking, pose estimation, object detection, and background segmentation — wrapped in clean, developer-friendly APIs that eliminate boilerplate and let you focus on building.

Whether you are prototyping a gesture-controlled application, building a fitness tracker, adding AR effects, or conducting research, OpenVisionKit gives you the tools to go from camera frame to structured detections in a few lines of code.

---

## Features

| Module | Capability |
|---|---|
| `FaceDetector` | Bounding boxes, 6-point keypoints, confidence filtering, IoU, face cropping |
| `FaceMeshDetector` | 478 landmarks, blendshapes, head pose (yaw/pitch/roll), gaze direction, emotion, AR overlays |
| `HandDetector` | 21 landmarks, gesture recognition, finger-join detection, distance estimation, palm width |
| `PoseDetector` | 33 body landmarks, joint angle calculation, exercise detection, workout rep counter, segmentation |
| `ObjectDetector` | EfficientDet-based multi-class detection with bounding boxes and labels |
| `SelfieSegmentation` | Background removal, blur, replacement, virtual backgrounds, alpha blending |
| `HairSegmentation` | Hair region segmentation and recoloring |
| `ScreenCapture` | High-performance screen grabbing via `mss` |
| `video_capture_template` | Drop-in webcam loop with FPS overlay, recording, and screenshot support |
| `image_template` | Single-image processing template with auto-centering, resize, and custom logic hook |
| `TextDetector` | Tesseract OCR with character/word/digit/table detection, NLP entity extraction, image matching, handwriting support |

---

## Requirements

- Python >= 3.11.8
- A `.tflite` / `.task` model file for each MediaPipe detector (see [Model Downloads](#model-downloads))

### TextDetector additional requirements

`TextDetector` uses Tesseract OCR and optional NLP tooling that are **not** bundled with MediaPipe.

**1. Install Tesseract binary** (system-level):

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

**2. Install Python packages:**

```bash
# pip
pip install pytesseract imutils pandas scikit-image Pillow

# uv
uv add pytesseract imutils pandas scikit-image Pillow
```

**3. (Optional) spaCy for NLP features** (entity extraction, keyword extraction, summarization, relation extraction):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

Without spaCy, all NLP methods return empty results gracefully — the rest of `TextDetector` works without it.

---

## Installation

### pip

```bash
pip install openvisionkit
```

Or install directly from source:

```bash
pip install git+https://github.com/your-org/openvisionkit.git
```

### uv

```bash
uv add openvisionkit
```

Or from source:

```bash
uv add git+https://github.com/your-org/openvisionkit.git
```

For development (editable install with all dev dependencies):

```bash
git clone https://github.com/your-org/openvisionkit.git
cd openvisionkit
make setup
```

---

## Model Downloads

OpenVisionKit delegates inference to MediaPipe `.tflite` / `.task` model files. Download the models you need and place them in a `models/` directory at your project root.

| Detector | Model file | Download |
|---|---|---|
| `FaceDetector` | `face_detector.tflite` | [MediaPipe Face Detector](https://developers.google.com/mediapipe/solutions/vision/face_detector) |
| `FaceMeshDetector` | `face_landmarker_v2_with_blendshapes.task` | [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |
| `HandDetector` | `hand_landmarker.task` | [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |
| `PoseDetector` | `pose_landmarker.task` | [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| `ObjectDetector` | `efficientdet_lite.tflite` | [MediaPipe Object Detector](https://developers.google.com/mediapipe/solutions/vision/object_detector) |
| `SelfieSegmentation` | `deeplab_v3.tflite` | [MediaPipe Image Segmenter](https://developers.google.com/mediapipe/solutions/vision/image_segmenter) |
| `HairSegmentation` | `hair_segmenter.tflite` | [MediaPipe Hair Segmenter](https://developers.google.com/mediapipe/solutions/vision/image_segmenter) |

---

## Quick Start

```python
import cv2
from openvisionkit.capture.video_template import video_capture_template
from openvisionkit.lib.hand_detector import HandDetector

detector = HandDetector(model_path="./models/hand_landmarker.task")

def process(frame):
    frame = detector.draw_landmarks(frame)
    return frame

video_capture_template(custom_logic=process, window_name="Hand Tracking")
```

---

## Usage

### FaceDetector

Detects faces in an image or video stream and returns bounding boxes, keypoints, and confidence scores.

```python
import cv2
from openvisionkit.lib.face_detector import FaceDetector

detector = FaceDetector(
    model_path="./models/face_detector.tflite",
    max_faces=5,
    running_mode="IMAGE",           # "IMAGE" | "VIDEO"
    min_detection_confidence=0.5,
    min_suppression_threshold=0.3,
)

frame = cv2.imread("photo.jpg")

# Returns annotated frame + list of detection dicts
annotated, detections = detector.detect_faces(frame, to_draw_bounding_box=True, to_draw_landmarks=True)

for det in detections:
    print(det["id"])                    # face index
    print(det["score"])                 # confidence 0–1
    print(det["bbox"])                  # (x, y, w, h)
    print(det["bbox_xyxy"])             # (x1, y1, x2, y2)
    print(det["center"])                # (cx, cy)
    print(det["normalized_keypoints"]) # list of (x, y) pixel coords for 6 landmarks

cv2.imshow("Faces", annotated)
cv2.waitKey(0)
```

**Utility methods:**

```python
# Filter detections below a confidence threshold
confident = detector.filter_by_confidence(detections, threshold=0.7)

# Get the largest face by bounding-box area
biggest = detector.get_largest_face(detections)

# Crop face regions out of the image (optional pixel margin)
face_crops = detector.crop_faces(frame, detections, margin=10)

# Sort by area (descending) or any other detection key
sorted_faces = detector.sort_faces(detections, by="area")

# Intersection over Union — useful for NMS or tracking
iou = detector.get_iou(detections[0]["bbox_xyxy"], detections[1]["bbox_xyxy"])
```

---

### FaceMeshDetector

Detects 478 facial landmarks per face along with blendshape expressions and head-pose matrices.

```python
import cv2
from openvisionkit.lib.face_mesh_detector import FaceMeshDetector

detector = FaceMeshDetector(
    model_path="./models/face_landmarker_v2_with_blendshapes.task",
    num_faces=2,
    min_face_detection_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

frame = cv2.imread("face.jpg")

annotated, faces, blendshapes, matrices, bboxes = detector.face_mesh_detection(frame, drawLandMarks=True)

# faces[i]       -> list of [x, y] pixel coords for 478 landmarks
# blendshapes[i] -> dict of {blendshape_name: score}  (52 expressions)
# matrices[i]    -> 4x4 numpy head-pose matrix
# bboxes[i]      -> [min_x, min_y, max_x, max_y]

for i, blend in enumerate(blendshapes):
    # Rule-based emotion from blendshapes
    emotion = detector.get_emotion(blend)
    print(f"Face {i}: {emotion}")

    # Gaze direction for each eye
    gaze = detector.get_eye_gaze_direction(faces[i], is_left_eye=True)
    print(f"Left gaze: {gaze}")   # "Left" | "Center" | "Right"

    # Mouth openness ratio (0 = closed, 0.5+ = wide open)
    ratio = detector.get_mouth_openness_ratio(faces[i])
    print(f"Mouth ratio: {ratio:.2f}")

    # Head pose angles from transformation matrix
    if matrices[i] is not None:
        yaw, pitch, roll = detector.get_head_pose_angles(matrices[i])
        print(f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}")

    # Inter-pupillary distance
    ipd = detector.get_inter_pupillary_distance(faces[i], normalized=False)
    print(f"IPD: {ipd:.1f}px")
```

**AR overlay example:**

```python
# Overlay a PNG glasses filter (must have alpha channel)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)   # RGBA
frame_with_glasses = detector.overlay_ar_filter(frame, faces[0], glasses, filter_type="glasses")
```

---

### HandDetector

Tracks up to N hands with 21 landmarks each. Provides gesture recognition, finger-join detection, and distance estimation.

```python
import cv2
from openvisionkit.lib.hand_detector import HandDetector

detector = HandDetector(
    model_path="./models/hand_landmarker.task",
    running_mode="IMAGE",       # "IMAGE" | "VIDEO"
    max_hands=2,
    detection_confidence=0.5,
    tracking_confidence=0.5,
    smoothing_window=8,
)

frame = cv2.imread("hand.jpg")

# Draw landmarks, bounding box, and handedness label
annotated = detector.draw_landmarks(
    frame,
    to_draw_landmark=True,
    to_draw_bounding_box=True,
    to_put_handle_label=True,
)

# Get structured landmark data for all detected hands
all_hands = detector.get_landmarks(frame)

for hand in all_hands:
    print(hand["hand_type"])          # "Left" or "Right"
    print(hand["bounding_box"])       # (x, y, w, h)
    print(hand["center_point"])       # (cx, cy)
    lm = hand["landmarks_list"]       # list of [id, x, y, z]

    # Which fingers are raised?
    fingers = detector.fingers_up(lm)
    # [thumb, index, middle, ring, little] — 1=up, 0=down

    # Gesture shortcuts
    print(detector.is_fist())
    print(detector.is_thumbs_up())
    print(detector.is_peace_sign())
    print(detector.is_open_hand())

    # Distance between any two landmarks with visual feedback
    p1 = (lm[4][1], lm[4][2])   # thumb tip
    p2 = (lm[8][1], lm[8][2])   # index tip
    length, annotated, coords = detector.get_distance(p1, p2, annotated)
    print(f"Thumb-index distance: {length:.1f}px")

    # Detect if two finger tips are touching
    joined = detector.is_fingers_joined(4, 8, annotated, lm, threshold=0.25)

    # Palm width in pixels (stable reference)
    palm_px, idx_mcp, pinky_mcp = detector.palm_width_px(frame, lm)
    print(f"Palm width: {palm_px:.1f}px")
```

**Distance estimation (calibration-based):**

```python
# Provide (palm_width_px, distance_cm) pairs to calibrate
calibration = [(180, 20), (120, 35), (80, 55), (60, 75)]
detector_calibrated = HandDetector(
    model_path="./models/hand_landmarker.task",
    calibration_samples=calibration,
)

# After calibration, estimate distance from a new palm width
distance_cm = detector_calibrated.estimate_distance_cm(palm_width_px=110)
print(f"Estimated distance: {distance_cm:.1f} cm")
```

---

### PoseDetector

Detects 33 body landmarks. Supports joint angle calculation, exercise classification, workout rep counting, and body segmentation.

```python
import cv2
from openvisionkit.lib.pose_detector import PoseDetector
from mediapipe.tasks.python import vision

detector = PoseDetector(
    model_path="./models/pose_landmarker.task",
    running_mode=vision.RunningMode.VIDEO,   # VIDEO for webcam streams
    num_poses=1,
    min_pose_detection_confidence=0.5,
    output_segmentation_masks=True,
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and annotate
    annotated, result = detector.detect(frame, draw_landmarks=True)

    # All landmark positions as pixel dicts
    landmarks = detector.get_all_postion(frame, result)

    # Get a specific landmark (e.g. nose = id 0)
    nose = detector.get_landmark(result, pose_index=0, landmark_id=0)
    print(nose["x"], nose["y"], nose["visibility"])

    # Calculate joint angle — e.g. left elbow (shoulder=11, elbow=13, wrist=15)
    annotated, angle = detector.calculate_angle(annotated, result, p1=11, p2=13, p3=15)
    print(f"Left elbow angle: {angle:.1f} degrees")

    # Classify current exercise
    exercise = detector.detect_exercise(annotated, result)
    print(f"Exercise: {exercise}")

    # Workout rep counter (tracks bicep curls automatically)
    angle, percent, reps = detector.calculate_workout_percentage()
    stats = detector.get_workout_stats(annotated)
    print(f"Reps: {stats['reps']}  Calories: {stats['calories']:.1f}")

    # Body segmentation overlay (requires output_segmentation_masks=True)
    annotated = detector.draw_segmentation_mask(annotated, result, alpha=0.5, color=(0, 255, 0))

    cv2.imshow("Pose", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

**Auto-select the most visible arm for curl tracking:**

```python
p1, p2, p3 = detector.select_active_arm(result)
annotated, angle = detector.calculate_angle(annotated, result, p1, p2, p3)
```

---

### ObjectDetector

Detects multiple object classes in a frame using EfficientDet Lite.

```python
import cv2
from openvisionkit.lib.object_detector import ObjectDetector

detector = ObjectDetector(
    model_path="./models/efficientdet_lite.tflite",
    max_results=5,
    running_mode="IMAGE",           # "IMAGE" | "VIDEO"
    category_allowlist=None,        # e.g. ["person", "car"] to restrict classes
    category_denylist=None,
)

frame = cv2.imread("street.jpg")

# Returns annotated image with bounding boxes and labels drawn
annotated = detector.detect_objects(frame)

cv2.imshow("Objects", annotated)
cv2.waitKey(0)

# Or get raw detection result for custom processing
result, mp_image = detector.detect(frame)
for detection in result.detections:
    label = detection.categories[0].category_name
    score = detection.categories[0].score
    bbox  = detection.bounding_box
    print(f"{label}: {score:.2f} @ ({bbox.origin_x}, {bbox.origin_y})")
```

---

### SelfieSegmentation

Separates people from backgrounds using DeepLab V3. Multiple compositing modes available.

```python
import cv2
from openvisionkit.lib.selfie_segmentation import SelfieSegmentation

seg = SelfieSegmentation(
    model_path="./models/deeplab_v3.tflite",
    output_category_mask=True,
)

frame = cv2.imread("selfie.jpg")

# Remove background (black fill)
no_bg = seg.remove_background(frame)

# Blur background
blurred = seg.blur_background(frame, blur_strength=(55, 55))

# Replace background with an image
replaced = seg.replace_background(frame, background_path="./bg.jpg")

# Solid color background
colored = seg.color_background(frame, color=(0, 120, 255))

# Alpha-blend foreground over a custom background array
bg = cv2.imread("./bg.jpg")
blended = seg.alpha_blend(frame, bg)

# Optimized virtual background with temporal smoothing + edge refinement
# (best for real-time webcam use)
output = seg.optimize_virtual_background(frame, bg)

# Single-person isolation — removes other people in the background
output = seg.optimize_virtual_background_improved(frame, bg)

# Debug: visualize the raw segmentation heatmap
heatmap = seg.overlay_mask(frame)

cv2.imshow("Segmented", output)
cv2.waitKey(0)
```

---

### HairSegmentation

Segments hair regions for recoloring or styling effects.

```python
import cv2
import numpy as np
from openvisionkit.lib.hair_segmentation import HairSegmentation

seg = HairSegmentation(model_path="./models/hair_segmenter.tflite")

frame = cv2.imread("portrait.jpg")
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

result = seg.process(rgb_frame)
mask = result.category_mask.numpy_view()    # shape (H, W), values 0–1

# Recolor hair to blue
hair_color = np.zeros_like(frame)
hair_color[:] = (255, 0, 0)                 # BGR blue
hair_region = (mask > 0.5)[..., None]
output = np.where(hair_region, hair_color, frame)

cv2.imshow("Hair", output)
cv2.waitKey(0)
```

---

### ScreenCapture

Captures live frames from a monitor — useful for screen-based CV pipelines.

```python
from openvisionkit.capture.screen_capture import ScreenCapture
import cv2

cap = ScreenCapture(monitor_index=1)  # 1 = primary monitor

while True:
    frame = cap.grab()                # returns BGR numpy array
    cv2.imshow("Screen", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
```

---

### video_capture_template

A reusable webcam loop that handles window setup, FPS display, recording, and screenshots. Pass a `custom_logic` callback for your processing.

```python
import cv2
from openvisionkit.capture.video_template import video_capture_template
from openvisionkit.lib.face_detector import FaceDetector

detector = FaceDetector(model_path="./models/face_detector.tflite", running_mode="VIDEO")

def process(frame):
    annotated, _ = detector.detect_faces(frame)
    return annotated

video_capture_template(
    video_source=0,                       # webcam index or path to video file
    custom_logic=process,
    window_name="Face Detection",
    resolution=(1280, 720),
    draw_fps=True,
    enable_auto_recording=True,           # auto-saves .mp4 from first frame
    record_format="mp4",                  # "mp4" or "gif"
    enable_screenshot=True,               # press 's' to capture a frame
    auto_screenshot_after_seconds=10.0,   # also auto-capture after 10 s
    auto_screenshot_repeat=False,         # True = repeat every 10 s
)
```

**Key bindings (built-in):**

| Key | Action | Condition |
|---|---|---|
| `ESC` | Exit loop | always |
| `s` / `S` | Save screenshot | `enable_screenshot=True` |
| `r` / `R` | Toggle manual recording on/off | `enable_manual_recording=True` |

**Stateful key handlers with `KeyEventManager`:**

```python
from openvisionkit.capture.video_template import KeyEventManager, video_capture_template

state = {"score": 0}
km = KeyEventManager()
km.register(ord("p"), lambda frame, s: print(f"Score: {s['score']}"))
km.register(ord("+"), lambda frame, s: s.update({"score": s["score"] + 1}))

video_capture_template(
    video_source=0,
    state=state,
    key_manager=km,
    custom_logic=lambda frame: frame,
)
```

**Manual recording:**

```python
video_capture_template(
    video_source=0,
    enable_manual_recording=True,   # press R to start, R again to stop and save
    record_format="gif",
)
```

**Parameter reference:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_source` | `int \| str` | `0` | Camera index or path to video file |
| `loop_forever` | `bool` | `True` | Loop video file when it ends |
| `custom_logic` | `Callable[[ndarray], ndarray]` | `None` | Per-frame processing; receives and returns BGR image |
| `state` | `dict` | `None` | Shared state dict passed to every key handler |
| `key_manager` | `KeyEventManager` | `None` | Custom key-event dispatcher |
| `window_name` | `str` | `"Demo"` | OpenCV window title |
| `show_window` | `bool` | `True` | Display the OpenCV window |
| `resolution` | `tuple[int, int]` | `(1280, 720)` | Camera resolution `(width, height)` |
| `center_window` | `bool` | `True` | Auto-center window on screen via pyautogui |
| `draw_fps` | `bool` | `True` | Overlay FPS counter on frame |
| `fps` | `int` | `15` | Recording frame rate (auto-recording only) |
| `mouse_callback` | `Callable` | `None` | OpenCV mouse-event callback |
| `mouse_callback_params` | `dict` | `None` | Extra params passed to mouse callback |
| `enable_auto_recording` | `bool` | `False` | Record every frame automatically from start |
| `enable_manual_recording` | `bool` | `False` | Allow toggling recording with `R` key |
| `record_format` | `str` | `"mp4"` | `"mp4"` or `"gif"` |
| `enable_screenshot` | `bool` | `False` | Enable `s`-key and auto-screenshot |
| `screenshot_output_dir` | `str` | `"screenshots"` | Directory for saved screenshots |
| `screenshot_prefix` | `str` | `"capture"` | Filename prefix before timestamp |
| `auto_screenshot_after_seconds` | `float` | `None` | Trigger first screenshot after N seconds |
| `auto_screenshot_repeat` | `bool` | `False` | Repeat auto-screenshot every N seconds |

---

### image_template

A single-image equivalent of `video_capture_template`. Loads one image from disk, applies an optional processing callback, resizes to the target resolution, auto-centers the window on screen, and displays it.

```python
import cv2
from openvisionkit.capture.image_template import image_template
from openvisionkit.lib.face_detector import FaceDetector

detector = FaceDetector(model_path="./models/face_detector.tflite", running_mode="IMAGE")

def process(frame):
    annotated, _ = detector.detect_faces(frame)
    return annotated

image_template(
    image_path="photo.jpg",
    custom_logic=process,       # receives the loaded BGR image, must return BGR image
    window_name="Face Demo",
    resolution=(1280, 720),     # image is resized to this before display
    center_window=True,         # auto-centers window on screen via pyautogui
    show_window=True,           # set False to run headless (e.g. save to disk instead)
)
```

Without a `custom_logic` callback the image is loaded, resized, and displayed as-is:

```python
image_template(image_path="photo.jpg")
```

**Parameter reference:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_path` | `str` | required | Path to the image file |
| `custom_logic` | `Callable[[ndarray], ndarray]` | `None` | Processing function applied before display |
| `window_name` | `str` | `"Demo"` | OpenCV window title |
| `resolution` | `tuple[int, int]` | `(1280, 720)` | `(width, height)` to resize the image |
| `center_window` | `bool` | `True` | Move window to screen center via pyautogui |
| `show_window` | `bool` | `True` | Display the OpenCV window |

---

### TextDetector

Tesseract-backed OCR class with per-character, per-word, and per-digit detection, document boundary detection, table extraction, image-to-image feature matching, cursive/handwriting OCR, and optional NLP post-processing via spaCy.

#### Installation prerequisites

See [TextDetector additional requirements](#textdetector-additional-requirements) above before using this class.

#### Basic OCR

```python
import cv2
from openvisionkit.lib.text_detector import TextDetector

image = cv2.imread("document.jpg")

detector = TextDetector(
    image=image,
    lang="eng",         # Tesseract language code(s); multi-language: "eng+chi_sim"
    oem=3,              # OCR Engine Mode — 3 = default (LSTM preferred)
    psm=6,              # Page Segmentation Mode — 6 = single uniform text block
    preprocess=True,    # apply grayscale + histogram equalization + adaptive threshold
    use_gpu=False,      # enable OpenCL GPU acceleration for OpenCV ops
)

# Full text string from the image
text = detector.detect_text()
print(text)

# Switch language at runtime (no need to reinstantiate)
detector.set_language("eng+fra")

# Replace the image on an existing instance
new_image = cv2.imread("page2.jpg")
detector.set_image(new_image)
```

#### Word-level detection

```python
words, annotated = detector.detect_words(
    draw_boxes=True,
    bounding_box_color=(255, 0, 0),   # BGR
    text_color=(255, 0, 0),
    font_scale=1,
    font_thickness=2,
)

for word in words:
    print(word["text"])   # recognized word string
    print(word["conf"])   # Tesseract confidence 0–100
    print(word["x"], word["y"], word["w"], word["h"])  # bounding box

cv2.imshow("Words", annotated)
cv2.waitKey(0)

# Convenience accessors
word_strings = detector.get_words()          # List[str]
lines         = detector.get_lines()          # List[str] — full lines
avg_conf      = detector.get_confidence()     # float — mean confidence across all words
df            = detector.to_dataframe()       # pandas DataFrame of word detections
```

#### Character-level detection

```python
chars, annotated = detector.detect_characters(
    draw_boxes=True,
    is_dark_background=False,    # set True to invert image before OCR
    adjust_text_height=20,       # vertical offset for label above bounding box
    bounding_box_color=(255, 0, 0),
    text_color=(255, 0, 0),
)

for c in chars:
    print(c["char"])               # single character string
    print(c["x1"], c["y1"])        # top-left (OpenCV coords)
    print(c["x2"], c["y2"])        # bottom-right (OpenCV coords)
```

#### Digit-only detection

```python
digits, annotated = detector.detect_digits(image, draw_boxes=True)
print(digits)   # e.g. ['4', '2', '0']
```

#### Document & table detection

```python
# Detect document boundary (returns 4-corner numpy array, or None)
corners = detector.detect_document()
if corners is not None:
    print("Document corners:", corners)

# Extract text from table regions using morphological line detection
tables = detector.detect_tables()
for table_text in tables:
    print(table_text)
```

#### Orientation & script detection

```python
osd = detector.image_to_osd()
print(osd["Orientation in degrees"])   # e.g. '90'
print(osd["Script"])                   # e.g. 'Latin'
```

#### Export formats

```python
# PDF bytes
pdf_bytes = detector.image_to_pdf_or_hocr(extension="pdf")
with open("output.pdf", "wb") as f:
    f.write(pdf_bytes)

# hOCR HTML bytes
hocr_bytes = detector.image_to_pdf_or_hocr(extension="hocr")

# ALTO XML string (structured layout format for digital libraries)
alto_xml = detector.image_to_alto_xml()
```

#### Handwriting / cursive OCR

```python
text, preprocessed = detector.extract_cursive_text(image)
print(text)
# preprocessed is the adaptive-threshold binary image used for OCR
```

#### Image preprocessing utilities

```python
# Resize (uses imutils to preserve aspect ratio)
resized = detector.resize(width=800)

# Rotate (may clip corners)
rotated = detector.rotate(angle=45)

# Rotate without clipping
rotated_bound = detector.rotate_bound(angle=45)

# Auto deskew (corrects small rotation from skewed scans)
deskewed = detector.deskew()

# Auto Canny edge detection with sigma-based threshold
edges = detector.auto_canny(sigma=0.33)
```

#### ORB keypoint detection and image matching

These methods are useful for comparing a scanned form against a template to detect alignment, tampering, or form type.

```python
# Detect ORB keypoints and descriptors
keypoints, descriptors, annotated = detector.detect_keypoints(
    features=500,
    draw_keypoints=True,
    keypoint_color=(0, 255, 0),
)

# Compare two images using KNN feature matching + RANSAC homography
# Falls back to SSIM if not enough features are found
template = cv2.imread("template.jpg")
result = detector.compare_matches_knn_matcher(
    image2=template,
    form_name="Invoice",
    no_of_feature=500,
    matched_amount=50,
    percentage_of_matches=20,
    draw_matches=False,
    draw_aligned=False,
)
print(result["matches"])          # number of good matches
print(result["homography"])       # 3x3 transformation matrix
# result["aligned_image"]         # template warped to match the query
# result["matched_image"]         # side-by-side match visualization

# Brute-force matcher variant (no ratio test, faster but less selective)
result_bf = detector.compare_matches_bf_matcher(image2=template, form_name="Invoice")

# SSIM-based fallback (used automatically, also callable directly)
ssim_result = TextDetector.fallback_ssim(image, template, "Invoice")
print(ssim_result["ssim_score"])  # structural similarity 0.0–1.0
```

#### NLP methods (requires spaCy `en_core_web_sm`)

```python
raw_text = detector.detect_text()

# Clean whitespace and newlines
clean = detector.clean_text(raw_text)

# Named entity recognition — returns list of {text, label} dicts
entities = detector.extract_entities(raw_text)
# e.g. [{"text": "Singapore", "label": "GPE"}, {"text": "2026", "label": "DATE"}]

# Group entities by label
grouped = detector.group_entities(raw_text)
# e.g. {"GPE": ["Singapore"], "DATE": ["2026"]}

# Keyword extraction (nouns and proper nouns, stop-words filtered)
keywords = detector.extract_keywords(raw_text)

# Extractive summarization (top N sentences)
summary = detector.summarize(raw_text, max_sentences=3)

# Subject-verb-object relation extraction
relations = detector.extract_relations(raw_text)
# e.g. [{"subject": ["John"], "verb": "signed", "object": ["contract"]}]
```

#### GPU acceleration

```python
detector.enable_gpu()    # enables OpenCV OpenCL (requires compatible GPU)
detector.disable_gpu()   # revert to CPU
```

---

## Project Structure

```
openvisionkit/
├── __init__.py               # package version (__version__)
├── lib/
│   ├── face_detector.py          # FaceDetector
│   ├── face_mesh_detector.py     # FaceMeshDetector (478 landmarks)
│   ├── hand_detector.py          # HandDetector (21 landmarks)
│   ├── pose_detector.py          # PoseDetector (33 landmarks)
│   ├── object_detector.py        # ObjectDetector (EfficientDet)
│   ├── selfie_segmentation.py    # SelfieSegmentation
│   ├── hair_segmentation.py      # HairSegmentation
│   ├── fps_counter.py            # FPSCounter utility
│   ├── classifier.py             # Generic classifier
│   ├── form_detector.py          # Form / document detector
│   ├── form_roi_detector.py      # Form region-of-interest detector
│   ├── form_roi_annotator.py     # Form annotation utilities
│   ├── image_detector.py         # Image-based detector
│   ├── image_hsv_detector.py     # HSV color-range detector
│   └── text_detector.py          # Text detection
├── capture/
│   ├── video_template.py         # video_capture_template loop
│   ├── screen_capture.py         # ScreenCapture
│   ├── video_recorder.py         # VideoRecorder
│   ├── image_template.py         # Single-image processing template
│   └── draw_object.py            # Drawing helpers
└── utility/
    ├── vision_utilis.py          # Shared image utilities
    └── live_plot.py              # Real-time matplotlib plotting
```

---

## Running Modes

All detectors support three MediaPipe running modes:

| Mode | Use case | Notes |
|---|---|---|
| `IMAGE` | Static images | No timestamp needed |
| `VIDEO` | Webcam / pre-recorded video | Pass `timestamp_ms` or let detector auto-increment |
| `LIVE_STREAM` | Async streaming | Results delivered via callback |

---

## Contributing

### Dev setup

```bash
git clone https://github.com/your-org/openvisionkit.git
cd openvisionkit
make setup          # uv sync + install pre-commit hooks
```

### Useful Make targets

| Target | What it does |
|---|---|
| `make setup` | Install all deps + pre-commit hooks (run once after clone) |
| `make format` | Auto-format with black + isort |
| `make lint` | Run ruff + flake8 |
| `make lint-fix` | Auto-fix ruff-fixable issues |
| `make test` | Run all non-integration tests |
| `make test-cov` | Run tests with HTML coverage report |
| `make typecheck` | mypy static analysis |
| `make check` | format-check + lint + typecheck (pre-push sanity) |

### Commit convention

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/). The pre-commit hook enforces this.

| Prefix | Effect |
|---|---|
| `fix:`, `perf:`, `refactor:` | patch release |
| `feat:` | minor release |
| `feat!:` or `BREAKING CHANGE:` footer | major release |
| `chore:`, `docs:`, `test:`, `ci:` | no release |

### CI/CD

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci-unit.yml` | push / PR | Unit tests on Python 3.11 + 3.12 |
| `ci-integration.yml` | push/PR to main, manual | Integration tests (requires model files) |
| `ci-security.yml` | push/PR to main, daily 02:00 UTC | pip-audit, Trivy, CodeQL |
| `renovate.yml` | weekly Monday 01:00 UTC | Automated dependency updates |
| `semantic-release.yml` | push to main | Semantic version bump + GitHub Release |
| `publish.yml` | GitHub Release published | Build + publish to PyPI via OIDC |

Releases are fully automated — push commits to `main` and the semantic-release workflow handles version bumping, tagging, changelog generation, and PyPI publishing.

---

## License

MIT
