import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

class FaceDetector:
  """
  FaceDetector class that utilizes MediaPipe's Face Detection solution to detect faces in images or video frames. It provides options to draw bounding boxes and landmarks on the detected faces.
  """
  def __init__(
      self, 
      model_path: str="./models/face_detector.tflite",
      max_faces=5,
      running_mode="IMAGE",  # IMAGE | VIDEO | LIVE_STREAM
      min_detection_confidence: float=0.5,
      min_suppression_threshold: float=0.3
    ):
      self.running_mode = getattr(vision.RunningMode, running_mode)
      self.base_options = python.BaseOptions(
        model_asset_path=model_path
     )
      self.max_faces = max_faces
      self.min_detection_confidence = min_detection_confidence
      self.min_suppression_threshold = min_suppression_threshold  
      self.options = vision.FaceDetectorOptions(
          base_options=self.base_options, 
          running_mode=self.running_mode,  # IMAGE | VIDEO | LIVE_STREAM
          min_detection_confidence=self.min_detection_confidence, 
          min_suppression_threshold=self.min_suppression_threshold
      )
      self.detector = vision.FaceDetector.create_from_options(self.options)
      self.mp_drawing_utils = mp.tasks.vision.drawing_utils
      self.mp_drawing_styles = mp.tasks.vision.drawing_styles
      
  def _to_mp_image(self, image):
    """
    Convert a BGR image (as used by OpenCV) to an mp.Image format suitable for MediaPipe processing.
    Args:
      image: The input image in BGR format (as used by OpenCV).
    Returns:
      An mp.Image object in RGB format suitable for MediaPipe processing.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


  def detect(self, image, timestamp_ms=None):
    """
    Detect faces in the input image using MediaPipe's Face Detector.
    Args: 
      image: The input image in which to detect faces (BGR format).
      timestamp_ms: An optional timestamp in milliseconds for video processing (required for VIDEO and LIVE_STREAM modes).
    Returns:
      The raw detection result from MediaPipe's Face Detector, which includes information about detected faces such
    """
    mp_image = self._to_mp_image(image)
    if self.running_mode == vision.RunningMode.IMAGE:
        result = self.detector.detect(mp_image)
    else:
        result = self.detector.detect_for_video(mp_image, timestamp_ms or 0)

    return result

  def _parse_detections(self, result, shape):
    """
    Parse the raw detection results from MediaPipe and extract relevant information such as bounding boxes, keypoints, and confidence scores.
    Args:
      result: The raw detection result from MediaPipe's Face Detector.
      shape: The shape of the input image (height, width).
    Returns:
      A list of parsed detections, where each detection is a dictionary containing information about the detected face, including its bounding box, confidence score, keypoints, and other relevant attributes.
    """
    
    H, W = shape[:2]
    parsed = []
    bounding_boxes = []
    key_points = []
    categories = []
    if not result.detections:
      return parsed

    for i, detection in enumerate(result.detections):
        """
          Detection(bounding_box=BoundingBox(origin_x=180, origin_y=145, width=701, height=701), categories=[Category(index=0, score=0.9549353718757629, display_name=None, category_name=None)], keypoints=[NormalizedKeypoint(x=0.18383397161960602, y=0.2978437542915344, label=None, score=0.0), NormalizedKeypoint(x=0.33176130056381226, y=0.2957031726837158, label=None, score=0.0), NormalizedKeypoint(x=0.25055351853370667, y=0.4610801339149475, label=None, score=0.0), NormalizedKeypoint(x=0.2593384385108948, y=0.543393611907959, label=None, score=0.0), NormalizedKeypoint(x=0.12543150782585144, y=0.3002464771270752, label=None, score=0.0), NormalizedKeypoint(x=0.4346682131290436, y=0.2893249988555908, label=None, score=0.0)])
        """
        score = detection.categories[0].score if detection.categories else 0
        bbox = detection.bounding_box
        bounding_boxes.append(bbox)
        key_points.append(detection.keypoints)
        categories.append(detection.categories)
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        x2 = x + w
        y2 = y + h

        bounding_box_coordinates = (x, y, w, h)
        parsed.append({
            "id": i,
            "score": score,
            "bbox": (x, y, w, h),
            "bbox_xyxy": (x, y, x2, y2),
            "center": (x + w // 2, y + h // 2),
            "coordinates": bounding_box_coordinates,
            "area": w * h,
            "normalized_keypoints": self._normalize_keypoints(detection.keypoints, W, H),
            "bounding_boxes": bounding_boxes,
            "key_points": key_points,
            "categories": categories
        })
    return parsed

  def detect_faces(self, image, timestamp_ms=None, to_draw_bounding_box=True, to_draw_landmarks=True):
    """
    Detect faces in the input image and optionally draw bounding boxes and landmarks on the detected faces.
    Args:
      image: The input image in which to detect faces (BGR format).
      timestamp_ms: An optional timestamp in milliseconds for video processing (required for VIDEO and LIVE_STREAM modes).
      to_draw_bounding_box: Whether to draw bounding boxes around detected faces.
      to_draw_landmarks: Whether to draw facial landmarks on the detected faces.  
    Returns:
      The image with detected faces (and optionally drawn bounding boxes and landmarks).
    """
    # Implement face detection logic here
    detection_result = self.detect(image, timestamp_ms)
  
    detections = self._parse_detections(detection_result, image.shape)
    
    if self.options.min_detection_confidence is not None:
      detections = self.filter_by_confidence(detections, self.options.min_detection_confidence)

    if self.max_faces is not None:
      detections = self.sort_faces(detections)[:self.max_faces]

    if to_draw_bounding_box:
        image = self.draw_detections(image, detections, to_draw_landmarks)
         
    return image, detections

  def draw_detections(self, image, detections, draw_landmarks=True):
    """
    Draw bounding boxes and landmarks for detected faces on the input image.
    Args:
      image: The input image on which to draw detections (BGR format). 
      detections: A list of detected faces with their bounding box and landmark information.
      draw_landmarks: Whether to draw facial landmarks on the detected faces.
    Returns:
      The image with drawn bounding boxes and landmarks for detected faces.
    """
    for det in detections:
        x, y, x2, y2 = det["bbox_xyxy"]
        fontface = 2 if self.running_mode == vision.RunningMode.IMAGE else 0.8
        cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 255), 2)

        cv2.putText(
            image,
            f'{int(det["score"] * 100)}%',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontface,
            (0, 255, 255),
            2
        )

        if draw_landmarks:
            for (kx, ky) in det["normalized_keypoints"]:
                cv2.circle(image, (kx, ky), 2, (0, 255, 0), -1)

    return image

  def filter_by_confidence(self, detections, threshold=0.5):
    """
    Filter detected faces based on a confidence threshold.
    Args:
      detections: A list of detected faces with their confidence scores.
      threshold: The confidence threshold for filtering detections.
    Returns:
      A list of detections that have confidence scores above the specified threshold. 
    """  
    return [d for d in detections if d["score"] >= threshold]

  def get_largest_face(self, detections):
    """
    Get the largest detected face based on the area.
    Args:
      detections: A list of detected faces with their bounding box information.
    Returns:
      The detection with the largest area, or None if no detections are available.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d["area"])

  def crop_faces(self, image, detections, margin=0):
    """
    Crop detected faces from the input image based on their bounding boxes.
    Args:
      image: The input image from which to crop faces (BGR format).
      detections: A list of detected faces with their bounding box information.
      margin: An optional margin to add around the bounding box when cropping (default is 0).
    Returns:
      A list of cropped face images.
    """
    faces = []
    H, W = image.shape[:2]
    for det in detections:
      x, y, w, h = det["bbox"]
      x1 = max(0, x - margin)
      y1 = max(0, y - margin)
      x2 = min(W, x + w + margin)
      y2 = min(H, y + h + margin)
      faces.append(image[y1:y2, x1:x2])
    return faces

  def sort_faces(self, detections, by="area", descending=True):
    """
    Sort detected faces based on a specified attribute.
    Args:
      detections: A list of detected faces with their attributes.
      by: The attribute to sort by (default is "area").
      descending: Whether to sort in descending order (default is True).
    Returns:
      A list of sorted detections.
    """
    return sorted(detections, key=lambda x: x[by], reverse=descending)
  
  def get_iou(self, boxA, boxB):
    """
    useful for tracking / NMS
    A box is defined by its top-left corner (x1, y1) and bottom-right corner (x2, y2).
    
    Args:
        boxA: A tuple (x1, y1, x2, y2) representing the first bounding box.
        boxB: A tuple (x1, y1, x2, y2) representing the second bounding box.  
        
    Returns:
        The Intersection over Union (IoU) value between the two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)

  def _normalize_keypoints(self, keypoints, W, H):
    """
    Normalize keypoints to the image dimensions.
    Args:
      keypoints: A list of keypoints with x and y coordinates normalized between 0 and 1.
      W: The width of the image.
      H: The height of the image.
    Returns:
      A list of keypoints with coordinates scaled to the image dimensions.
    """
    if not keypoints:
        return []
    return [(int(k.x * W), int(k.y * H)) for k in keypoints]

  # ─────────────────────────── NEW METHODS ───────────────────────────

  def count_faces(self, detections):
    """Return total number of detected faces.

    Args:
      detections: List of detection dicts returned by detect_faces().
    Returns:
      int: number of detections.
    """
    return len(detections)

  def blur_faces(self, image, detections, blur_strength=(51, 51), margin=0):
    """Blur every detected face region in-place on a copy of the image.
    Useful for privacy masking before saving or streaming.

    Args:
      image: BGR numpy array.
      detections: List of detection dicts from detect_faces().
      blur_strength: (kW, kH) kernel size for GaussianBlur — must be odd.
      margin: Extra pixels to expand each face crop before blurring.
    Returns:
      BGR numpy array with blurred faces.
    """
    out = image.copy()
    H, W = out.shape[:2]
    for det in detections:
      x, y, w, h = det["bbox"]
      x1 = max(0, x - margin)
      y1 = max(0, y - margin)
      x2 = min(W, x + w + margin)
      y2 = min(H, y + h + margin)
      roi = out[y1:y2, x1:x2]
      if roi.size == 0:
        continue
      # Kernel sizes must be positive odd numbers
      kw = blur_strength[0] if blur_strength[0] % 2 == 1 else blur_strength[0] + 1
      kh = blur_strength[1] if blur_strength[1] % 2 == 1 else blur_strength[1] + 1
      out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)
    return out

  def is_face_in_zone(self, detection, zone_rect):
    """Check whether the face center falls inside a rectangular zone.
    Useful for attendance systems, door-unlock triggers, restricted-area alerts.

    Args:
      detection: Single detection dict from detect_faces().
      zone_rect: (x, y, w, h) defining the zone rectangle in pixel coords.
    Returns:
      bool: True if face center is inside the zone.
    """
    cx, cy = detection["center"]
    zx, zy, zw, zh = zone_rect
    return (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)

  def get_face_screen_position(self, detection, image_width):
    """Classify horizontal position of a face as 'left', 'center', or 'right'.
    Divides the frame into three equal vertical bands.

    Args:
      detection: Single detection dict from detect_faces().
      image_width: Width of the source image in pixels.
    Returns:
      str: 'left' | 'center' | 'right'
    """
    cx = detection["center"][0]
    third = image_width / 3
    if cx < third:
      return "left"
    elif cx < 2 * third:
      return "center"
    return "right"

  def track_faces(self, prev_detections, curr_detections, iou_threshold=0.3):
    """Associate current-frame faces to previous-frame faces via IoU.
    Returns matched pairs; unmatched current detections are marked as new.

    Args:
      prev_detections: List of detection dicts from the previous frame.
      curr_detections: List of detection dicts from the current frame.
      iou_threshold: Minimum IoU to consider two faces the same person.
    Returns:
      List of dicts: [{
        'prev': detection_or_None,
        'curr': detection,
        'is_new': bool
      }]
    """
    matched = []
    used_prev = set()

    for curr in curr_detections:
      best_iou = 0.0
      best_prev = None
      for i, prev in enumerate(prev_detections):
        if i in used_prev:
          continue
        iou = self.get_iou(curr["bbox_xyxy"], prev["bbox_xyxy"])
        if iou > best_iou:
          best_iou = iou
          best_prev = (i, prev)

      if best_prev and best_iou >= iou_threshold:
        used_prev.add(best_prev[0])
        matched.append({"prev": best_prev[1], "curr": curr, "is_new": False})
      else:
        matched.append({"prev": None, "curr": curr, "is_new": True})

    return matched

  def draw_zone(self, image, zone_rect, color=(0, 255, 255), label="Zone", thickness=2):
    """Draw a named rectangular zone on the image.

    Args:
      image: BGR numpy array.
      zone_rect: (x, y, w, h) zone coordinates.
      color: BGR color tuple.
      label: Text label drawn above the zone rectangle.
      thickness: Border thickness in pixels.
    Returns:
      Annotated BGR numpy array.
    """
    out = image.copy()
    x, y, w, h = zone_rect
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(out, label, (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

  # ─────────────────────── PRIVACY & CROP UTILITIES ───────────────────────

  def pixelate_faces(self, image, detections, block_size=10):
    """Pixelate every detected face region for privacy masking.

    Downscales the face ROI to a tiny tile grid then upscales back with
    nearest-neighbour interpolation, creating a mosaic / pixelation effect.

    Args:
      image: BGR numpy array.
      detections: List of detection dicts from detect_faces().
      block_size: Pixel block size; larger values = coarser mosaic.
    Returns:
      BGR numpy array with pixelated faces (copy of input).
    """
    out = image.copy()
    for det in detections:
      x, y, w, h = det["bbox"]
      x1 = max(0, x)
      y1 = max(0, y)
      x2 = min(image.shape[1], x + w)
      y2 = min(image.shape[0], y + h)
      if x2 <= x1 or y2 <= y1:
        continue
      roi = out[y1:y2, x1:x2]
      small_w = max(1, (x2 - x1) // block_size)
      small_h = max(1, (y2 - y1) // block_size)
      small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
      out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                      interpolation=cv2.INTER_NEAREST)
    return out

  def is_frontal(self, detection, threshold=0.8):
    """Heuristic frontal-face check based on detection confidence.

    MediaPipe Face Detection scores are higher for well-aligned, frontal
    faces, so a high-confidence score is a reasonable frontal proxy.

    Args:
      detection: Single detection dict from detect_faces().
      threshold: Minimum confidence score to consider the face frontal.
    Returns:
      bool: True if detection score >= threshold.
    """
    return detection["score"] >= threshold

  def get_padded_crop(self, image, detection, pad_ratio=0.2):
    """Crop a face with proportional padding on all sides.

    Adds padding relative to the face bounding-box dimensions, then clips
    to the image boundary so the crop is always valid.

    Args:
      image: BGR numpy array.
      detection: Single detection dict from detect_faces().
      pad_ratio: Fraction of face width/height to add as padding on each side.
    Returns:
      BGR numpy array crop (copy).
    """
    x, y, w, h = detection["bbox"]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    return image[y1:y2, x1:x2].copy()

  # ─────────────────── TRACKING & BATCH UTILITIES ──────────────────────────

  def draw_face_ids(self, image, tracked_faces):
    """Overlay persistent face IDs on the image.

    Draws a green bounding box and an "ID:<n>" label above each tracked face.

    Args:
      image: BGR numpy array.
      tracked_faces: List of dicts with keys ``bbox`` (x, y, w, h) and ``id``.
    Returns:
      Annotated BGR numpy array (copy of input).
    """
    out = image.copy()
    for face in tracked_faces:
      x, y, w, h = face["bbox"]
      face_id = face.get("id", 0)
      cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(out, f"ID:{face_id}", (x, max(y - 10, 10)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

  def get_attention_score(self, detections, img_w, img_h):
    """Estimate viewer attention as a 0–1 scalar.

    Combines two signals per detection and returns the best score across all
    detected faces:

    * **Area ratio** — larger face implies closer / more engaged viewer.
    * **Centrality** — face centred in the frame scores higher than one at
      the periphery.

    Args:
      detections: List of detection dicts from detect_faces().
      img_w: Frame width in pixels.
      img_h: Frame height in pixels.
    Returns:
      float in [0.0, 1.0]. 0.0 when no detections are present.
    """
    if not detections:
      return 0.0
    img_area = img_w * img_h
    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    max_dist = (img_cx ** 2 + img_cy ** 2) ** 0.5
    scores = []
    for det in detections:
      area_ratio = min(det["area"] / img_area, 1.0)
      cx, cy = det["center"]
      dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
      centrality = 1.0 - min(dist / max_dist, 1.0) if max_dist > 0 else 1.0
      scores.append(0.5 * area_ratio + 0.5 * centrality)
    return float(max(scores))

  def batch_detect(self, images):
    """Run detect_faces on a list of frames and return only the detection lists.

    Args:
      images: Iterable of BGR numpy arrays.
    Returns:
      List[List[dict]]: one detection list per input frame, in order.
    """
    return [self.detect_faces(img)[1] for img in images]

  def save_crops(self, image, detections, output_dir, prefix="face"):
    """Crop each detected face (with padding) and save to disk as PNG files.

    Files are named ``<prefix>_<index>.png`` and written to *output_dir*,
    which is created if it does not exist.

    Args:
      image: BGR numpy array.
      detections: List of detection dicts from detect_faces().
      output_dir: Destination directory path (created automatically).
      prefix: Filename prefix for saved crops.
    Returns:
      List[str]: absolute file paths of the written PNG files, in order.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, det in enumerate(detections):
      crop = self.get_padded_crop(image, det, pad_ratio=0.1)
      path = os.path.join(output_dir, f"{prefix}_{i}.png")
      cv2.imwrite(path, crop)
      paths.append(path)
    return paths
