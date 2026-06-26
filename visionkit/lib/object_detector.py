from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL = str(_MODEL_DIR / "efficientdet_lite.tflite")


class ObjectDetector:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_results=5,
        running_mode="IMAGE",
        display_names_locale=b"en",
        category_allowlist=None,
        category_denylist=None,
    ):
        self.running_mode = getattr(vision.RunningMode, running_mode)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            max_results=max_results,
            running_mode=self.running_mode,
            display_names_locale=display_names_locale,
            category_allowlist=category_allowlist,
            category_denylist=category_denylist,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.MARGIN = 15
        self.FONT_THICKNESS = 2
        self.ROW_SIZE = 10
        self.TEXT_COLOR = (0, 255, 0)
        self.FONT_SIZE = 1

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
        Detect objects in the given image using the MediaPipe Object Detector.
        Args:
          image: The input image in BGR format (as used by OpenCV).
          timestamp_ms: Optional timestamp in milliseconds for video processing (ignored in IMAGE mode).
        Returns:
          A list of detected objects with their bounding boxes and labels.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._to_mp_image(rgb)
        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.detector.detect(mp_image)
        else:
            result = self.detector.detect_for_video(mp_image, timestamp_ms or 0)
        return result, mp_image

    def visualize_detections(self, image, detection_result):
        """
        Visualize detected objects on the image using MediaPipe's visualization utilities.
        Args:
          image: The input image in BGR format (as used by OpenCV).
          detection_result: The result from the detect method containing detected objects.
        Returns:
          An annotated image with detected objects visualized.
        """
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0] if detection.categories else None
            label = category.category_name if category else "Unknown"
            score = category.score if category else 0.0

            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (
                int(bbox.origin_x + bbox.width),
                int(bbox.origin_y + bbox.height),
            )

            cv2.rectangle(
                image, start_point, end_point, self.TEXT_COLOR, self.FONT_THICKNESS
            )
            text = f"{label}: {score:.2f}"
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SIZE, self.FONT_THICKNESS
            )
            text_origin = (start_point[0], start_point[1] + self.MARGIN)
            cv2.putText(
                image,
                text,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.FONT_SIZE,
                self.TEXT_COLOR,
                self.FONT_THICKNESS,
            )
        return image

    def detect_objects(self, image, timestamp_ms=None):
        detection_result, mp_image = self.detect(image, timestamp_ms=timestamp_ms)
        image_copy = np.copy(mp_image.numpy_view())
        annotated_image = self.visualize_detections(image_copy, detection_result)
        return annotated_image

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def count_objects(self, detection_result, label=None):
        """Return total detections, optionally filtered to one class.

        Args:
          detection_result: Raw result from detect().
          label: If provided, count only detections with this category name (case-insensitive).
        Returns:
          int
        """
        if label is None:
            return len(detection_result.detections)
        label_lower = label.lower()
        return sum(
            1
            for d in detection_result.detections
            if d.categories and d.categories[0].category_name.lower() == label_lower
        )

    def filter_by_class(self, detection_result, allowed_classes):
        """Return only detections whose top category is in allowed_classes.

        Args:
          detection_result: Raw result from detect().
          allowed_classes: List of category name strings (case-insensitive).
        Returns:
          List of Detection objects.
        """
        allowed = {c.lower() for c in allowed_classes}
        return [
            d
            for d in detection_result.detections
            if d.categories and d.categories[0].category_name.lower() in allowed
        ]

    def get_largest_object(self, detection_result):
        """Return the detection with the largest bounding-box area, or None.

        Args:
          detection_result: Raw result from detect().
        Returns:
          Detection object or None.
        """
        if not detection_result.detections:
            return None
        return max(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )

    def is_object_in_zone(self, detection, zone_rect):
        """Check whether the center of a detection falls inside a zone rectangle.
        Useful for counting objects crossing a virtual boundary or entering an area.

        Args:
          detection: A Detection object from detection_result.detections.
          zone_rect: (x, y, w, h) zone in pixel coordinates.
        Returns:
          bool
        """
        b = detection.bounding_box
        cx = int(b.origin_x + b.width / 2)
        cy = int(b.origin_y + b.height / 2)
        zx, zy, zw, zh = zone_rect
        return (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)

    def get_object_centers(self, image, detection_result):
        """Return center coordinates and label for every detection.

        Args:
          image: Source BGR frame (used only for shape — not drawn on).
          detection_result: Raw result from detect().
        Returns:
          List of dicts: [{'label': str, 'score': float, 'center': (cx, cy)}]
        """
        results = []
        for d in detection_result.detections:
            b = d.bounding_box
            cx = int(b.origin_x + b.width / 2)
            cy = int(b.origin_y + b.height / 2)
            label = d.categories[0].category_name if d.categories else "unknown"
            score = d.categories[0].score if d.categories else 0.0
            results.append({"label": label, "score": score, "center": (cx, cy)})
        return results

    def draw_zone(
        self, image, zone_rect, color=(0, 255, 255), label="Zone", thickness=2
    ):
        """Draw a named rectangular detection zone on the image.

        Args:
          image: BGR numpy array.
          zone_rect: (x, y, w, h).
          color: BGR color tuple.
          label: Text displayed above the rectangle.
          thickness: Border thickness in pixels.
        Returns:
          Annotated BGR numpy array.
        """
        out = image.copy()
        x, y, w, h = zone_rect
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            out,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return out

    def get_class_summary(self, detection_result):
        """Return a dict mapping each detected class to its count.

        Args:
          detection_result: Raw result from detect().
        Returns:
          dict: {'person': 3, 'car': 1, ...}
        """
        summary = {}
        for d in detection_result.detections:
            if not d.categories:
                continue
            name = d.categories[0].category_name
            summary[name] = summary.get(name, 0) + 1
        return summary

    def filter_by_confidence(self, detection_result, threshold=0.5):
        """Return only detections whose top-category score meets or exceeds the threshold.

        Args:
          detection_result: Raw result from detect().
          threshold: Minimum confidence score (inclusive). Default 0.5.
        Returns:
          List of Detection objects.
        """
        return [
            d
            for d in detection_result.detections
            if d.categories and d.categories[0].score >= threshold
        ]

    def get_bounding_boxes(self, detection_result):
        """Return bounding box info for every detection as a list of dicts.

        Args:
          detection_result: Raw result from detect().
        Returns:
          List of dicts: [{'label': str, 'score': float, 'bbox': (x, y, w, h)}]
        """
        boxes = []
        for detection in detection_result.detections:
            bb = detection.bounding_box
            label = (
                detection.categories[0].category_name
                if detection.categories
                else "unknown"
            )
            score = detection.categories[0].score if detection.categories else 0.0
            boxes.append(
                {
                    "label": label,
                    "score": score,
                    "bbox": (bb.origin_x, bb.origin_y, bb.width, bb.height),
                }
            )
        return boxes

    def is_crowded(self, detection_result, threshold=5):
        """Return True if the number of detections meets or exceeds the threshold.

        Args:
          detection_result: Raw result from detect().
          threshold: Minimum count to consider crowded. Default 5.
        Returns:
          bool
        """
        return len(detection_result.detections) >= threshold

    def get_objects_by_size(self, detection_result):
        """Return detections sorted by bounding-box area, largest first.

        Args:
          detection_result: Raw result from detect().
        Returns:
          List of Detection objects sorted descending by area.
        """
        return sorted(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
            reverse=True,
        )

    def get_proximity(self, det_a, det_b):
        """Return the Euclidean distance between the centers of two detections.

        Args:
          det_a: A Detection object.
          det_b: A Detection object.
        Returns:
          float — pixel distance between bounding-box centers.
        """
        bb_a, bb_b = det_a.bounding_box, det_b.bounding_box
        cx_a = bb_a.origin_x + bb_a.width / 2
        cy_a = bb_a.origin_y + bb_a.height / 2
        cx_b = bb_b.origin_x + bb_b.width / 2
        cy_b = bb_b.origin_y + bb_b.height / 2
        return float(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5)

    def detect_line_crossing(
        self, detection_result, line_start, line_end, line_threshold=10
    ):
        """Return detections whose center is within line_threshold pixels of the line segment.

        Useful for virtual tripwire / counting lines.

        Args:
          detection_result: Raw result from detect().
          line_start: (x, y) tuple — start of the line segment.
          line_end: (x, y) tuple — end of the line segment.
          line_threshold: Maximum perpendicular distance in pixels. Default 10.
        Returns:
          List of dicts: [{'label': str, 'center': (cx, cy), 'distance': float}]
        """
        p1 = np.array(line_start, dtype=float)
        p2 = np.array(line_end, dtype=float)
        line_len = np.linalg.norm(p2 - p1)
        crossing = []
        for detection in detection_result.detections:
            bb = detection.bounding_box
            cx = bb.origin_x + bb.width / 2
            cy = bb.origin_y + bb.height / 2
            p = np.array([cx, cy])
            if line_len == 0:
                dist = float(np.linalg.norm(p - p1))
            else:
                t = float(np.clip(np.dot(p - p1, p2 - p1) / (line_len**2), 0, 1))
                proj = p1 + t * (p2 - p1)
                dist = float(np.linalg.norm(p - proj))
            if dist <= line_threshold:
                label = (
                    detection.categories[0].category_name
                    if detection.categories
                    else "unknown"
                )
                crossing.append({"label": label, "center": (cx, cy), "distance": dist})
        return crossing

    def export_to_json(self, detection_result):
        """Serialise detection results to a JSON-compatible dict.

        Args:
          detection_result: Raw result from detect().
        Returns:
          dict with key 'detections', each entry containing label, score, and bbox.
        """
        return {
            "detections": [
                {
                    "label": (
                        d.categories[0].category_name if d.categories else "unknown"
                    ),
                    "score": d.categories[0].score if d.categories else 0.0,
                    "bbox": {
                        "x": d.bounding_box.origin_x,
                        "y": d.bounding_box.origin_y,
                        "width": d.bounding_box.width,
                        "height": d.bounding_box.height,
                    },
                }
                for d in detection_result.detections
            ]
        }

    def batch_detect(self, images):
        """Run detect() on a list of images and return one result per frame.

        Args:
          images: List of BGR numpy arrays.
        Returns:
          List of DetectionResult objects (one per image).
        """
        return [self.detect(img)[0] for img in images]
