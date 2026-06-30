# OpenVisionKit Utility Methods — Design Spec
**Date:** 2026-06-09
**Approach:** In-place additions to existing class files (Approach A)
**Scope:** 11 classes across `openvisionkit/lib/`

---

## Goal

Add comprehensive utility/wrapper methods to all detector and segmentation classes so developers can cover real-world use cases (real-time video, document processing, fitness/health, security/surveillance) without writing boilerplate on top of the library.

All new methods are added directly to existing class files. No new files, no mixins, no architectural change.

---

## Conventions

- All methods follow existing naming/style: `snake_case`, BGR input, BGR annotated output where applicable.
- Methods that take `image` always accept a BGR `np.ndarray`.
- Methods that produce an annotated image always return a copy (`image.copy()`), never mutate in place.
- Methods that may fail gracefully (e.g. `langdetect` absent) log a warning and return a safe default.
- Return types are documented inline with each method.

---

## FaceDetector (`openvisionkit/lib/face_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `pixelate_faces` | `(image, detections, block_size=10)` | `np.ndarray` | Replace each face bbox with pixelated blocks for privacy |
| `is_frontal` | `(detection, threshold=0.8)` | `bool` | True if detection score ≥ threshold (proxy for frontal confidence) |
| `get_padded_crop` | `(image, detection, pad_ratio=0.2)` | `np.ndarray` | Crop face region with proportional padding, clipped to image bounds |
| `draw_face_ids` | `(image, tracked_faces)` | `np.ndarray` | Draw numeric ID from `track_faces` output above each bbox |
| `get_attention_score` | `(detections, img_w, img_h)` | `float` | 0–1 score: face area proportion × centrality in frame |
| `batch_detect` | `(images)` | `list[list[dict]]` | Run `detect_faces` on each frame, return list of detection lists |
| `save_crops` | `(image, detections, output_dir, prefix="face")` | `list[str]` | Save each face crop as `{prefix}_{i}.png`, return saved paths |

---

## FaceMeshDetector (`openvisionkit/lib/face_mesh_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `is_smiling` | `(blend, threshold=0.4)` | `bool` | `mouthSmileLeft` + `mouthSmileRight` avg > threshold |
| `is_yawning` | `(face, ratio_threshold=0.5)` | `bool` | Mouth openness ratio > threshold |
| `is_surprised` | `(blend, face, brow_threshold=0.3, mouth_threshold=0.3)` | `bool` | Both `browInnerUp` > brow_threshold AND mouth open > mouth_threshold |
| `get_eyebrow_raise` | `(blend)` | `float` | `browInnerUp` blendshape coefficient (0–1) |
| `is_eyes_closed` | `(face, ear_threshold=0.22)` | `bool` | Both left and right EAR below threshold |
| `get_face_bounding_box` | `(face)` | `tuple[int,int,int,int]` | `(x, y, w, h)` pixel bbox from landmark extents |
| `get_face_symmetry_score` | `(face)` | `float` | 0–1 mirror score: mean absolute difference of mirrored landmark pairs, normalized |
| `draw_face_oval` | `(image, face)` | `np.ndarray` | Draw ellipse fitted to face oval landmark indices |
| `get_attention_level` | `(face, blend)` | `float` | Composite 0–1: penalise large yaw/pitch and off-center gaze |
| `get_lip_separation` | `(face)` | `float` | Pixel distance between upper and lower lip center landmarks |
| `is_drowsy` | `(face, ear_threshold=0.22, frames_threshold=1)` | `bool` | Both EARs below threshold — same as `is_eyes_closed` but named for driver-monitoring context |

---

## HandDetector (`openvisionkit/lib/hand_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `is_ok_sign` | `(hand_landmarks)` | `bool` | Thumb tip near index tip and remaining fingers extended |
| `is_call_me` | `(hand_landmarks)` | `bool` | Thumb + pinky extended, index/middle/ring folded |
| `is_rock_sign` | `(hand_landmarks)` | `bool` | Index + pinky extended, middle/ring folded, thumb folded |
| `get_hand_orientation` | `(hand_landmarks)` | `str` | `"palm_up"/"palm_down"/"palm_left"/"palm_right"` from wrist→middle-MCP vector |
| `get_swipe_direction` | `(prev_wrist, curr_wrist, threshold=20)` | `str` | `"left"/"right"/"up"/"down"/"none"` from pixel delta |
| `get_all_finger_angles` | `(hand_landmarks)` | `dict[str, float]` | `{"thumb":…, "index":…, …}` bend angles (MCP–PIP–DIP) |
| `draw_gesture_label` | `(image, hand_data, label)` | `np.ndarray` | Render label string above hand bounding box |
| `to_json` | `(hand_data)` | `dict` | Serialize hand dict (landmarks as plain lists, bounding box, hand_type) |
| `recognize_number` | `(hand_landmarks)` | `int` | 0–5 from finger count via `get_finger_count` |

---

## PoseDetector (`openvisionkit/lib/pose_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `get_spine_angle` | `(detection_result)` | `float` | Angle (degrees) of shoulder-midpoint → hip-midpoint vector from vertical |
| `is_arms_raised` | `(detection_result, threshold=0.2)` | `bool` | Both wrists' normalized y < shoulders' normalized y − threshold |
| `get_torso_tilt` | `(detection_result)` | `float` | Signed degrees: shoulder line angle from horizontal (positive = right tilt) |
| `detect_fall` | `(detection_result)` | `bool` | Nose/head y > hip y in normalized coords → person is horizontal/fallen |
| `get_body_bounding_box` | `(detection_result, image)` | `tuple[int,int,int,int]` | `(x,y,w,h)` pixel bbox enclosing all visible landmarks |
| `get_symmetry_score` | `(detection_result)` | `float` | 0–1 left/right landmark mirror score (shoulder, elbow, wrist, hip, knee, ankle) |
| `get_knee_angle` | `(detection_result, side="left")` | `float` | Knee flexion angle via `calculate_angle` on hip–knee–ankle |
| `get_hip_angle` | `(detection_result, side="left")` | `float` | Hip hinge angle via shoulder–hip–knee |
| `is_hunching` | `(detection_result, threshold=20)` | `bool` | Shoulder slope > threshold degrees from horizontal |
| `is_arms_crossed` | `(detection_result)` | `bool` | Left wrist x > right shoulder x AND right wrist x < left shoulder x |

---

## ObjectDetector (`openvisionkit/lib/object_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `filter_by_confidence` | `(detection_result, threshold=0.5)` | `list[Detection]` | Return list of Detection objects with score ≥ threshold (MediaPipe DetectionResult is immutable) |
| `get_bounding_boxes` | `(detection_result)` | `list[dict]` | `[{"label", "score", "bbox": (x,y,w,h)}, …]` |
| `is_crowded` | `(detection_result, threshold=5)` | `bool` | Total detection count ≥ threshold |
| `get_objects_by_size` | `(detection_result)` | `list` | Detections sorted by bbox area descending |
| `get_proximity` | `(det_a, det_b)` | `float` | Euclidean pixel distance between center points of two detections |
| `detect_line_crossing` | `(detection_result, line_start, line_end, line_threshold=10)` | `list[dict]` | Detections whose center lies within `line_threshold` pixels of the line segment |
| `export_to_json` | `(detection_result)` | `dict` | JSON-safe dict: `{"detections": [{label, score, bbox}, …]}` |
| `batch_detect` | `(images)` | `list` | Run `detect` on each frame, return list of DetectionResults |

---

## SelfieSegmentation (`openvisionkit/lib/selfie_segmentation.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `is_person_present` | `(image, min_area_ratio=0.01)` | `bool` | Foreground mask area ÷ total pixels > min_area_ratio |
| `get_person_center` | `(image)` | `tuple[int,int]` | `(cx, cy)` centroid of foreground mask |
| `get_foreground_bounds` | `(image)` | `tuple[int,int,int,int]` | `(x,y,w,h)` bbox from `cv2.boundingRect` on foreground mask |
| `create_green_screen` | `(image)` | `np.ndarray` | Replace background with pure green `(0,255,0)` |
| `extract_foreground_on_white` | `(image)` | `np.ndarray` | Replace background with white `(255,255,255)` |
| `apply_bokeh_effect` | `(image, blur_radius=25)` | `np.ndarray` | Gaussian blur background, sharp foreground composite |
| `apply_edge_glow` | `(image, color=(0,255,0), thickness=3)` | `np.ndarray` | Draw dilated silhouette contour as glow outline |
| `measure_foreground_height` | `(image)` | `int` | Pixel height of foreground bounding box |

---

## HairSegmentation (`openvisionkit/lib/hair_segmentation.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `get_hair_bounding_box` | `(bgr_image)` | `tuple[int,int,int,int]` | `(x,y,w,h)` bbox of hair mask region |
| `get_hair_top_position` | `(bgr_image)` | `int` | Topmost y-coordinate of hair mask pixels |
| `apply_gradient_color` | `(bgr_image, color1, color2)` | `np.ndarray` | Vertical gradient dye: color1 at top → color2 at bottom of hair bbox |
| `apply_highlights` | `(bgr_image, highlight_color=(255,255,200), intensity=0.4)` | `np.ndarray` | Blend lighter color patches using a sparse random mask within hair region |
| `detect_hair_length_estimate` | `(bgr_image)` | `str` | `"short"/"medium"/"long"` from hair-height ÷ image-height ratio |
| `get_hair_density_map` | `(bgr_image)` | `np.ndarray` | Grayscale heatmap (Gaussian-blurred hair mask) as density visualization |

---

## ImageDetector (`openvisionkit/lib/image_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `resize_to_fit` | `(max_width, max_height)` | `np.ndarray` | Resize preserving aspect ratio to fit within bounds |
| `pad_to_square` | `(fill=0)` | `np.ndarray` | Pad shorter axis symmetrically with fill value |
| `normalize` | `(mean=(0,0,0), std=(1,1,1))` | `np.ndarray` | Per-channel float32 normalize: `(img/255 - mean) / std` |
| `get_dominant_colors` | `(k=5)` | `list[tuple]` | K-means on pixels → list of k BGR color tuples |
| `overlay_image` | `(overlay, x, y, alpha=1.0)` | `np.ndarray` | Alpha-composite overlay BGR image at `(x, y)` |
| `create_thumbnail` | `(size=(128,128))` | `np.ndarray` | Resize to fixed size using `cv2.INTER_AREA` |
| `compare_histograms` | `(other_image)` | `float` | 0–1 similarity via `cv2.compareHist` with `HISTCMP_CORREL` |
| `to_base64` | `()` | `str` | Encode image as base64 PNG string for web/API use |
| `batch_crop` | `(boxes)` | `list[np.ndarray]` | Crop list of `(x,y,w,h)` regions from image |

---

## TextDetector (`openvisionkit/lib/text_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `is_text_present` | `(min_confidence=60.0)` | `bool` | True if any word detected above confidence |
| `extract_dates` | `(text=None)` | `list[str]` | Regex patterns: DD/MM/YYYY, YYYY-MM-DD, Month DD YYYY |
| `extract_phone_numbers` | `(text=None)` | `list[str]` | Regex: `+country`, `(area)`, plain 7–15 digit patterns |
| `extract_emails` | `(text=None)` | `list[str]` | Standard RFC-5322 simplified regex |
| `redact_sensitive` | `(patterns=None)` | `np.ndarray` | Black box over bounding regions of matched words on annotated image |
| `get_reading_order` | `(words)` | `list[dict]` | Sort word dicts by `top` then `left` (natural reading order) |
| `get_text_density` | `()` | `float` | Character count ÷ image area (chars per pixel²) |
| `detect_language` | `(text=None)` | `str` | Language code via `langdetect`; returns `"unknown"` if package absent |

---

## FormROIDetector (`openvisionkit/lib/form_roi_detector.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `get_empty_fields` | `(regions)` | `list[ROIRegion]` | Filter regions where `fill_state == "empty"` |
| `validate_required_fields` | `(regions, required_labels)` | `dict` | `{"missing": [...], "filled": [...]}` by label match |
| `get_field_by_label` | `(regions, label)` | `ROIRegion \| None` | Case-insensitive label match, returns first hit |
| `get_form_completion_score` | `(regions)` | `float` | Filled count ÷ total count (0–1) |
| `highlight_empty_fields` | `(image, regions, color=(0,0,255), thickness=2)` | `np.ndarray` | Draw colored border on each empty field |
| `extract_all_text` | `(image, regions)` | `dict[str, str]` | `{label: ocr_text}` for all regions via `_ocr_text` |

---

## FormROIAnnotator (`openvisionkit/lib/form_roi_annotator.py`)

### New Methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
Annotation format: each annotation is `[(x1,y1), (x2,y2), "type", "label", "category"]` (matches `self.rois` list-of-lists).

| `get_annotations_by_type` | `(annotations, field_type)` | `list` | Filter annotation list where `ann[2] == field_type` |
| `export_to_json` | `(annotations, path)` | `None` | Serialize annotation list to JSON file (same format as existing auto-save) |
| `get_annotation_count` | `(annotations)` | `dict[str, int]` | Count per `ann[2]` type string |
| `merge_annotations` | `(ann_list_a, ann_list_b)` | `list` | Combine two lists, deduplicate by bbox IoU > 0.5 |
| `draw_annotation_summary` | `(image, annotations)` | `np.ndarray` | Overlay legend box with type counts in top-left corner |

---

## Summary

| Class | New methods |
|---|---|
| FaceDetector | 7 |
| FaceMeshDetector | 11 |
| HandDetector | 9 |
| PoseDetector | 10 |
| ObjectDetector | 8 |
| SelfieSegmentation | 8 |
| HairSegmentation | 6 |
| ImageDetector | 9 |
| TextDetector | 8 |
| FormROIDetector | 6 |
| FormROIAnnotator | 5 |
| **Total** | **87** |
