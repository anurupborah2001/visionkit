import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

_MODEL_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL = str(_MODEL_DIR / "pose_landmarker.task")


class PoseDetector:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        running_mode: VisionTaskRunningMode = vision.RunningMode.VIDEO,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = False,
    ):
        """
        Args:
            model_path: Path to the .task model file
            running_mode: IMAGE (static) or VIDEO (real-time stream - recommended)
            num_poses: Maximum number of people to detect (1 is fastest)
            min_pose_detection_confidence: Detection threshold
            min_pose_presence_confidence: Pose presence threshold
            min_tracking_confidence: Tracking threshold (only used in VIDEO mode)
            output_segmentation_masks: Enable body segmentation (person vs background)

        Note: For real-time applications, use VIDEO mode with num_poses=1 for best performance.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        self.running_mode = running_mode
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        self.mp_drawing_utils = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.frame_count = 0  # Used for automatic timestamp in VIDEO mode

        # WORKOUT COUNTER STATE
        self.rep_count = 0
        self.stage = "up"  # "up" = arm extended (0%), "down" = curled (100%)
        self.min_angle = 160.0  # Start assuming arm is mostly straight
        self.max_angle = 30.0  # Expected minimum when fully curled
        self.angle = 0.0  # current angle at the joint being monitored

        self.rep_min_threshold = 40  # must go below this angle
        self.rep_max_threshold = 140  # must extend above this
        self.rep_start_time = None
        self.min_rep_time = 0.5  # seconds (too fast = cheat)
        self.session_start = time.time()
        self.rep_times = []

    def detect(
        self,
        img: cv2.typing.MatLike,
        draw_landmarks: bool = True,
        timestamp_ms: int | None = None,
        to_draw_landmarks: bool = True,
    ) -> tuple[cv2.typing.MatLike, PoseLandmarkerResult]:
        """
        Main method - detects pose and returns:
            (annotated_image, full_detection_result)

        Use this directly in your video_capture_template custom_logic.

        For VIDEO mode: timestamp is auto-managed if not provided.

        Args:
            img: The input image (BGR format as read by OpenCV) to process for pose detection.
            draw_landmarks: Whether to draw the detected pose landmarks on the image.
            timestamp_ms: Optional timestamp in milliseconds for VIDEO mode (if not provided, it will be auto-calculated to ensure smooth tracking).
            to_draw_landmarks: Whether to draw circles at the landmark positions on the image (for visualization).

          Returns:
            A tuple containing the annotated image (with landmarks drawn if enabled) and the full detection result from the pose detector, which includes landmark positions, visibility, and other relevant information.
        """
        # BGR → RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Choose correct detection method based on running mode
        if self.running_mode == vision.RunningMode.IMAGE:
            detection_result = self.pose_detector.detect(mp_image)
        else:  # VIDEO mode
            if timestamp_ms is None:
                self.frame_count += 1
                timestamp_ms = self.frame_count * 33  # ~30 FPS smooth tracking
            detection_result = self.pose_detector.detect_for_video(
                mp_image, timestamp_ms
            )

        annotated_image = img.copy()
        # Draw landmarks with beautiful styles (color + thickness + visibility)
        if draw_landmarks and detection_result.pose_landmarks and to_draw_landmarks:
            for landmarks in detection_result.pose_landmarks:
                # Landmark: x=0.7140882015228271, y=0.25891464948654175, z=-0.19056275486946106, visibility=0.999813973903656
                # print("Landmark:", landmarks[0])  # Print the first landmark of each detected pose for debugging
                self.mp_drawing_utils.draw_landmarks(
                    annotated_image,
                    landmarks,
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                )

        return annotated_image, detection_result

    def get_all_postion(
        self,
        img: cv2.typing.MatLike,
        detection_result: PoseLandmarkerResult,
        to_draw_landmarks: bool = True,
    ) -> list | None:
        """
        Get list of all landmark positions for a specific pose (default is first detected pose).

        Args:
            img: The original image (used for dimensions)
            detection_result: The full detection result from the process() method
            to_draw_landmarks: Whether to draw circles at the landmark positions on the image (for visualization)

        Returns:
            A list of landmarks with their pixel coordinates and visibility, or None if no landmarks are detected.
        """
        self.list_of_landmarks = []
        if detection_result:
            h, w, _ = img.shape
            for _id, landmarks in enumerate(detection_result.pose_landmarks):
                for idx, landmark in enumerate(landmarks):
                    cx, cy, cz = int(landmark.x * w), int(landmark.y * h), landmark.z
                    self.list_of_landmarks.append(
                        {
                            "id": idx,
                            "x": cx,
                            "y": cy,
                            "z": cz,
                            "center": (cx, cy),
                            "visibility": landmark.visibility,
                            "presence": landmark.presence,
                            "name": landmark.name,
                        }
                    )
            if to_draw_landmarks:
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.list_of_landmarks

    def get_landmark(
        self,
        detection_result: PoseLandmarkerResult,
        pose_index: int = 0,
        landmark_id: int = 0,
    ) -> dict | None:
        """
        Get specific landmark position for a specific pose (default is first detected pose).

        Args:
            detection_result: The full detection result from the process() method
            pose_index: Index of the detected pose (default is 0 for first detected pose)
            landmark_id: ID of the landmark to retrieve (e.g., 0 for nose, 11 for left shoulder, etc.)

        Returns:
            A dictionary with the landmark's pixel coordinates, visibility, presence, and name, or None if the landmark is not detected.
        """
        if (
            detection_result.pose_landmarks
            and len(detection_result.pose_landmarks) > pose_index
        ):
            landmark = detection_result.pose_landmarks[pose_index][landmark_id]
            return {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
                "presence": landmark.presence,
                "name": landmark.name,
            }
        return None

    def get_world_landmark(
        self,
        detection_result: PoseLandmarkerResult,
        pose_index: int = 0,
        landmark_id: int = 0,
    ) -> dict | None:
        """
        Get 3D world coordinates (meters) - very useful for real 3D pose estimation and applications like AR/VR. Note that the world landmark coordinates are in meters with the origin at the center of the hips, and the y-axis pointing upwards.

        Args:
            detection_result: The full detection result from the process() method
            pose_index: Index of the detected pose (default is 0 for first detected pose)
            landmark_id: ID of the landmark to retrieve (e.g., 0 for nose, 11 for left shoulder, etc.)

        Returns:
            A dictionary with the landmark's 3D world coordinates (x, y, z in meters) and name, or None if the landmark is not detected.
        """
        if (
            detection_result.pose_world_landmarks
            and len(detection_result.pose_world_landmarks) > pose_index
        ):
            landmark = detection_result.pose_world_landmarks[pose_index][landmark_id]
            return {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "name": landmark.name,
            }
        return None

    def calculate_angle(
        self,
        image: cv2.typing.MatLike,
        detection_result: PoseLandmarkerResult,
        p1: int,
        p2: int,
        p3: int,
        pose_index: int = 0,
        to_draw_landmarks: bool = True,
    ) -> tuple[cv2.typing.MatLike, float]:
        """
        Calculate angle (in degrees) at joint p2 formed by points p1-p2-p3.
        Example: Left elbow = calculate_angle(result, 11, 13, 15)

        Args:
            image: The original image (used for dimensions and optional drawing)
            detection_result: The full detection result from the process() method
            p1, p2, p3: Landmark IDs for the three points to calculate the angle (e.g., for left elbow, p1=11 (left shoulder), p2=13 (left elbow), p3=15 (left wrist))
            pose_index: Index of the detected pose to use (default is 0 for first detected pose)
            to_draw_landmarks: Whether to draw circles at the landmark positions and lines between them on the image (for visualization). If enabled, it will draw the angle being calculated for better understanding of the joint movement.

        Returns:
            A tuple containing the annotated image (with landmarks and angle drawn if enabled) and the calculated
        """
        if (
            not detection_result.pose_landmarks
            or len(detection_result.pose_landmarks) <= pose_index
        ):
            return image, 0.0

        lm = detection_result.pose_landmarks[pose_index]
        # Get normalized points → convert to pixel
        h, w = image.shape[:2]
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in (p1, p2, p3)], dtype=int)
        (x1, y1), (x2, y2), (x3, y3) = pts

        # Angle calculation
        a, b, c = pts.astype(float)
        ba, bc = a - b, c - b
        self.angle = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                    -1,
                    1,
                )
            )
        )

        if to_draw_landmarks:
            # Lines between the points to visualize the angle being calculated
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(image, (x3, y3), (x2, y2), (255, 255, 255), 3)

            # Draw circles at the landmark positions
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(image, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(image, (x, y), 15, (0, 0, 255), 2)

            # Put the angle text near the joint (p2)
            cv2.putText(
                image,
                str(int(self.angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return image, self.angle

    def detect_exercise(self, img, detection_result, to_draw_exercise=True) -> str:
        if not detection_result.pose_landmarks:
            return "No Person"

        lm = detection_result.pose_landmarks[0]

        def pt(i):
            return np.array([lm[i].x, lm[i].y])

        def angle(a, b, c):
            ba, bc = a - b, c - b
            return np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(ba, bc)
                        / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                        -1,
                        1,
                    )
                )
            )

        # -------------------------
        # Angles
        # -------------------------
        left_elbow = angle(pt(11), pt(13), pt(15))
        right_elbow = angle(pt(12), pt(14), pt(16))
        elbow = min(left_elbow, right_elbow)

        left_knee = angle(pt(23), pt(25), pt(27))
        right_knee = angle(pt(24), pt(26), pt(28))
        knee = min(left_knee, right_knee)

        shoulder_y = (lm[11].y + lm[12].y) / 2
        wrist_y = min(lm[15].y, lm[16].y)
        hip_y = (lm[23].y + lm[24].y) / 2
        knee_y = (lm[25].y + lm[26].y) / 2

        # -------------------------
        # SCORING SYSTEM
        # -------------------------
        scores = {
            "Bicep Curl": 0,
            "Shoulder Press": 0,
            "Squat": 0,
            "Push-Up": 0,
            "Lunge": 0,
            "Standing": 0,
        }

        # 🏋️ BICEP CURL
        if 30 < elbow < 140:
            scores["Bicep Curl"] += 2
        if wrist_y > shoulder_y:
            scores["Bicep Curl"] += 1

        # 🏋️ SHOULDER PRESS
        if wrist_y < shoulder_y:
            scores["Shoulder Press"] += 2
        if elbow > 120:
            scores["Shoulder Press"] += 1

        # 🏋️ SQUAT
        if knee < 130:
            scores["Squat"] += 2
        if hip_y > knee_y:
            scores["Squat"] += 1

        # 🏋️ PUSH-UP
        if elbow < 110:
            scores["Push-Up"] += 1
        if abs(shoulder_y - hip_y) < 0.08:
            scores["Push-Up"] += 2

        # 🏋️ LUNGE
        if abs(lm[25].y - lm[26].y) > 0.1:
            scores["Lunge"] += 2
        if knee < 120:
            scores["Lunge"] += 1

        # 🧍 STANDING
        if elbow > 150 and knee > 160:
            scores["Standing"] += 3

        # -------------------------
        # PICK BEST MATCH
        # -------------------------
        exercise = max(scores, key=scores.get)

        # -------------------------
        # CONFIDENCE FILTER
        # -------------------------
        if scores[exercise] < 2:
            return "Straight Pose"

        if to_draw_exercise:
            cv2.putText(
                img, exercise, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
            )

        return exercise

    def select_active_arm(self, detection_result: PoseLandmarkerResult):
        # p1, p2, p3 = pose_detector.select_active_arm(result)
        lm = detection_result.pose_landmarks[0]
        left_vis = lm[11].visibility + lm[13].visibility + lm[15].visibility
        right_vis = lm[12].visibility + lm[14].visibility + lm[16].visibility
        if right_vis > left_vis:
            return (12, 14, 16)  # right arm
        else:
            return (11, 13, 15)  # left arm

    def get_workout_stats(self, frame, to_draw_landmarks=True):
        duration = time.time() - self.session_start
        avg_time = sum(self.rep_times) / len(self.rep_times) if self.rep_times else 0
        calories = self.rep_count * 0.5
        if to_draw_landmarks:
            cv2.putText(
                frame, f"Reps: {self.rep_count}", (30, 130), 0, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Avg: {avg_time:.2f}s", (30, 160), 0, 1, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Time: {int(duration)}s", (30, 190), 0, 1, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Cal: {calories:.1f}", (30, 220), 0, 1, (0, 165, 255), 2
            )
        return {
            "reps": self.rep_count,
            "avg_time": avg_time,
            "duration": duration,
            "calories": calories,
        }

    def calculate_workout_percentage(self) -> tuple[float, float, int]:
        angle = self.angle

        if not hasattr(self, "rep_min"):
            self.rep_min = angle
            self.rep_max = angle

        # Dynamically update min & max angles for the current rep to adapt to user's range of motion. This allows for more accurate percentage calculation and rep counting based on the user's actual movement, rather than relying on fixed angle thresholds which may not fit everyone.
        self.rep_min = min(self.rep_min, angle)
        self.rep_max = max(self.rep_max, angle)

        # Avoid division by zero and bad percentage in the first few frames when the angles are still stabilizing
        if abs(self.rep_max - self.rep_min) < 20:
            return angle, 0.0, self.rep_count

        # Calculate percentage based on current rep's min and max angles. High angle (arm straight) = 0%, Low angle (arm curled) = 100%
        percent = np.interp(angle, [self.rep_min, self.rep_max], [0, 100])  # high → low
        percent = np.clip(percent, 0, 100)

        # When the use arm is curled down and reaches near the max curl position (percent > 80%), we mark the stage as "down" and start the rep timer. When the arm is extended back up and reaches near the starting position (percent < 20%), we check if the rep was valid based on the range of motion and time taken, print the rep status, and reset for the next rep.
        if percent >= 80 and self.stage == "up":
            self.stage = "down"
            self.rep_start_time = time.time()
            self.rep_count += 1

        # When the arm is extended back up and reaches near the starting position (percent < 20%), we check if the rep was valid based on the range of motion and time taken, print the rep status, and reset for the next rep.
        if percent <= 20 and self.stage == "down":
            rep_time = time.time() - self.rep_start_time if self.rep_start_time else 1
            self.stage = "up"
            self.rep_min, self.rep_max = angle, angle
            self.rep_times.append(rep_time)

        # print(f"Angle: {angle:.1f} | RepMin: {self.rep_min:.1f} | RepMax: {self.rep_max:.1f} | %: {percent:.1f} | Reps: {self.rep_count}")

        return angle, percent, self.rep_count

    def draw_landmarks_on_image(
        self,
        image: cv2.typing.MatLike,
        detection_result: PoseLandmarkerResult,
        list_of_landmarks: list[int] | None = None,
    ) -> cv2.typing.MatLike:
        """
        Draw pose landmarks on the image with beautiful styles (color + thickness + visibility).

        Args:
            image: The original image to draw the landmarks on.
            detection_result: The full detection result from the process() method, which should include pose landmarks.
            list_of_landmarks: Optional list of landmark IDs to draw (if None, draws all landmarks). This allows you to selectively draw only certain landmarks if desired.

        Returns:
            The image with pose landmarks drawn if detected and enabled, otherwise returns the original image.
        """
        annotated_image = image.copy()
        if (
            detection_result.pose_landmarks
            and list_of_landmarks
            and len(list_of_landmarks) > 0
        ):
            for landmark_id in list_of_landmarks:
                landmark = detection_result.pose_landmarks[0][landmark_id]
                print(f"Drawing landmark {landmark_id}: {landmark}")
                cv2.circle(
                    annotated_image,
                    (
                        int(landmark.x * image.shape[1]),
                        int(landmark.y * image.shape[0]),
                    ),
                    5,
                    (0, 255, 0),
                    cv2.FILLED,
                )
        return annotated_image

    def draw_segmentation_mask(
        self,
        image: cv2.typing.MatLike,
        detection_result: PoseLandmarkerResult,
        alpha: float = 0.6,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> cv2.typing.MatLike:
        """
        Overlay segmentation mask if output_segmentation_masks=True.

        Args:
            image: The original image to draw the segmentation mask on.
            detection_result: The full detection result from the process() method, which should include segmentation masks if output_segmentation_masks was enabled during initialization.
            alpha: The transparency level of the segmentation mask overlay (0.0 to 1.0, where 0.0 is fully transparent and 1.0 is fully opaque).
            color: The color to use for the segmentation mask overlay in BGR format (default is green). This color will be applied to the areas of the image where the segmentation mask indicates the presence of a person. You can change this color to visualize the segmentation
            differently (e.g., (255, 0, 0) for red, (0, 0, 255) for blue, etc.).

        Returns:
            The image with the segmentation mask overlay if available, otherwise returns the original image.
        """
        if not detection_result.segmentation_masks:
            return image

        mask = detection_result.segmentation_masks[0].numpy_view() > 0.5

        # Fix shape → (H, W, 1)
        if mask.ndim == 2:
            mask = mask[..., None]

        # Ensure correct dtype
        mask = mask.astype(bool)

        color_arr = np.array(color, dtype=np.uint8)

        # Apply overlay
        overlay = np.where(mask, color_arr, image)

        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def reset_workout(self):
        """Reset all rep-counter state so the same instance can be reused for a new session.
        Call between sets, people, or exercise changes.
        """
        self.rep_count = 0
        self.stage = "up"
        self.min_angle = 160.0
        self.max_angle = 30.0
        self.angle = 0.0
        self.rep_start_time = None
        self.session_start = time.time()
        self.rep_times = []
        if hasattr(self, "rep_min"):
            del self.rep_min
        if hasattr(self, "rep_max"):
            del self.rep_max

    def get_body_center(self, image, detection_result):
        """Return the pixel midpoint of the hips — a stable body anchor.
        Falls back to shoulder midpoint if hips are not visible.

        Args:
          image: BGR numpy array (used for shape).
          detection_result: PoseLandmarkerResult from detect().
        Returns:
          tuple(int, int) or None if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return None
        h, w = image.shape[:2]
        lm = detection_result.pose_landmarks[0]
        # Hips: left=23, right=24; Shoulders: left=11, right=12
        left_id, right_id = (23, 24) if lm[23].visibility > 0.5 else (11, 12)
        cx = int((lm[left_id].x + lm[right_id].x) / 2 * w)
        cy = int((lm[left_id].y + lm[right_id].y) / 2 * h)
        return (cx, cy)

    def is_standing(self, detection_result):
        """Return True when both knees are nearly straight (angle > 160°).

        Args:
          detection_result: PoseLandmarkerResult from detect().
        Returns:
          bool
        """
        if not detection_result.pose_landmarks:
            return False
        lm = detection_result.pose_landmarks[0]

        def angle(a, b, c):
            ba = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
            bc = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
            cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_val, -1, 1)))

        left_knee = angle(23, 25, 27)
        right_knee = angle(24, 26, 28)
        return left_knee > 160 and right_knee > 160

    def is_sitting(self, detection_result):
        """Return True when at least one knee angle is between 70° and 130° (seated position).

        Args:
          detection_result: PoseLandmarkerResult from detect().
        Returns:
          bool
        """
        if not detection_result.pose_landmarks:
            return False
        lm = detection_result.pose_landmarks[0]

        def angle(a, b, c):
            ba = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
            bc = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
            cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_val, -1, 1)))

        left_knee = angle(23, 25, 27)
        right_knee = angle(24, 26, 28)
        return (70 < left_knee < 130) or (70 < right_knee < 130)

    def get_body_orientation(self, detection_result):
        """Estimate whether the person is facing the camera, showing their side, or back.
        Uses shoulder-width-to-hip-width ratio as the discriminator.

        Args:
          detection_result: PoseLandmarkerResult from detect().
        Returns:
          str: 'front' | 'side' | 'back' | 'unknown'
        """
        if not detection_result.pose_landmarks:
            return "unknown"
        lm = detection_result.pose_landmarks[0]
        shoulder_width = abs(lm[11].x - lm[12].x)
        hip_width = abs(lm[23].x - lm[24].x)
        ratio = shoulder_width / (hip_width + 1e-6)
        nose_vis = lm[0].visibility
        if nose_vis > 0.7 and ratio > 0.6:
            return "front"
        elif ratio < 0.3:
            return "side"
        elif nose_vis < 0.3:
            return "back"
        return "front"

    def count_visible_keypoints(self, detection_result, visibility_threshold=0.5):
        """Return how many landmarks are visible above the threshold.

        Args:
          detection_result: PoseLandmarkerResult from detect().
          visibility_threshold: Minimum visibility score to count a landmark.
        Returns:
          int
        """
        if not detection_result.pose_landmarks:
            return 0
        return sum(
            1
            for lm in detection_result.pose_landmarks[0]
            if lm.visibility >= visibility_threshold
        )

    def get_shoulder_angle(self, detection_result):
        """Return the tilt angle (degrees) of the shoulder line relative to horizontal.
        Positive = right shoulder higher, negative = left shoulder higher.
        Useful for posture analysis.

        Args:
          detection_result: PoseLandmarkerResult from detect().
        Returns:
          float: angle in degrees, or 0.0 if not detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lm = detection_result.pose_landmarks[0]
        dx = lm[12].x - lm[11].x
        dy = lm[12].y - lm[11].y
        return float(np.degrees(np.arctan2(dy, dx)))

    def get_all_visible_landmarks(
        self, detection_result: PoseLandmarkerResult, visibility_threshold: float = 0.5
    ) -> list[dict]:
        """
        Return list of all landmarks that are clearly visible (visibility above threshold) across all detected poses. This can be useful for filtering out unreliable landmarks in downstream applications.
        A landmark is considered visible if its visibility score is above the specified threshold. The returned list includes the pose index, landmark ID, pixel coordinates, visibility score, and landmark name for each visible landmark.

        Args:
            detection_result: The full detection result from the process() method
            visibility_threshold: The minimum visibility score for a landmark to be considered visible (default is 0.5)

        Returns:
            A list of dictionaries, each containing information about a visible landmark (pose index, landmark ID
        """
        visible = []
        if not detection_result.pose_landmarks:
            return visible

        for pose_idx, landmarks in enumerate(detection_result.pose_landmarks):
            for i, lm in enumerate(landmarks):
                if lm.visibility >= visibility_threshold:
                    visible.append(
                        {
                            "pose": pose_idx,
                            "id": i,
                            "x": lm.x,
                            "y": lm.y,
                            "visibility": lm.visibility,
                            "name": lm.name,
                        }
                    )
        return visible

    # ─────────────────────── POSTURE ANALYSIS METHODS ───────────────────────

    def get_spine_angle(self, detection_result) -> float:
        """Return the lateral lean of the spine in degrees (angle of shoulder-mid → hip-mid
        vector away from vertical). Near 0° = upright; larger value = leaning.

        Args:
            detection_result: PoseLandmarkerResult from detect().
        Returns:
            float: angle in degrees, or 0.0 if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        shoulder_mid_x = (lms[11].x + lms[12].x) / 2
        shoulder_mid_y = (lms[11].y + lms[12].y) / 2
        hip_mid_x = (lms[23].x + lms[24].x) / 2
        hip_mid_y = (lms[23].y + lms[24].y) / 2
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        return float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    def get_torso_tilt(self, detection_result) -> float:
        """Return the tilt of the shoulder line relative to horizontal (degrees).
        Positive = right shoulder lower; negative = left shoulder lower.

        Args:
            detection_result: PoseLandmarkerResult from detect().
        Returns:
            float: angle in degrees, or 0.0 if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        dx = lms[12].x - lms[11].x
        dy = lms[12].y - lms[11].y
        return float(math.degrees(math.atan2(dy, dx + 1e-6)))

    def is_hunching(self, detection_result, threshold: float = 20) -> bool:
        """Return True when the absolute torso tilt exceeds *threshold* degrees,
        which is a simple proxy for asymmetric shoulder hunch.

        Args:
            detection_result: PoseLandmarkerResult from detect().
            threshold: Tilt angle (degrees) above which hunching is flagged.
        Returns:
            bool
        """
        return abs(self.get_torso_tilt(detection_result)) > threshold

    def get_symmetry_score(self, detection_result) -> float:
        """Return a body-symmetry score in [0, 1] where 1.0 = perfect bilateral
        symmetry and 0.0 = highly asymmetric. Compares mirrored left landmarks
        against their right counterparts using visible pairs only.

        Args:
            detection_result: PoseLandmarkerResult from detect().
        Returns:
            float: symmetry score, or 0.0 if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
        mid_x = (lms[11].x + lms[12].x) / 2
        diffs = []
        for l_idx, r_idx in pairs:
            lm_l, r = lms[l_idx], lms[r_idx]
            if lm_l.visibility < 0.5 or r.visibility < 0.5:
                continue
            mirrored_lx = 2 * mid_x - lm_l.x
            dx = abs(mirrored_lx - r.x)
            dy = abs(lm_l.y - r.y)
            diffs.append((dx + dy) / 2)
        if not diffs:
            return 0.0
        return float(max(0.0, 1.0 - sum(diffs) / len(diffs) * 10))

    # ──────────────────── ACTION AND SPATIAL DETECTION ──────────────────────

    def is_arms_raised(self, detection_result, threshold: float = 0.2) -> bool:
        """Return True when both wrists are at least *threshold* above their
        respective shoulders (in normalised y, where smaller y = higher in frame).

        Args:
            detection_result: PoseLandmarkerResult from detect().
            threshold: Minimum y-distance (normalised) wrists must be above shoulders.
        Returns:
            bool
        """
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        return (
            lms[11].y - lms[15].y >= threshold - 1e-9
            and lms[12].y - lms[16].y >= threshold - 1e-9
        )

    def detect_fall(self, detection_result) -> bool:
        """Return True when the nose (head) y-coordinate is below the hip midpoint
        y-coordinate, indicating the person may have fallen.

        Args:
            detection_result: PoseLandmarkerResult from detect().
        Returns:
            bool
        """
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        hip_y = (lms[23].y + lms[24].y) / 2
        return lms[0].y > hip_y

    def is_arms_crossed(self, detection_result) -> bool:
        """Return True when the left wrist is on the right side of the body
        midline and the right wrist is on the left side (arms crossed).

        Args:
            detection_result: PoseLandmarkerResult from detect().
        Returns:
            bool
        """
        if not detection_result.pose_landmarks:
            return False
        lms = detection_result.pose_landmarks[0]
        mid_x = (lms[11].x + lms[12].x) / 2
        return lms[15].x > mid_x and lms[16].x < mid_x

    def get_knee_angle(self, detection_result, side: str = "left") -> float:
        """Return the knee flexion angle (degrees) for the given side.
        Uses hip → knee → ankle landmarks. ~180° = straight leg; ~90° = seated.

        Args:
            detection_result: PoseLandmarkerResult from detect().
            side: "left" or "right".
        Returns:
            float: angle in degrees, or 0.0 if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        if side == "left":
            hip, knee, ankle = lms[23], lms[25], lms[27]
        else:
            hip, knee, ankle = lms[24], lms[26], lms[28]
        a = np.array([hip.x, hip.y])
        b = np.array([knee.x, knee.y])
        c = np.array([ankle.x, ankle.y])
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

    def get_hip_angle(self, detection_result, side: str = "left") -> float:
        """Return the hip flexion angle (degrees) for the given side.
        Uses shoulder → hip → knee landmarks. ~180° = standing upright; smaller = bent.

        Args:
            detection_result: PoseLandmarkerResult from detect().
            side: "left" or "right".
        Returns:
            float: angle in degrees, or 0.0 if no pose detected.
        """
        if not detection_result.pose_landmarks:
            return 0.0
        lms = detection_result.pose_landmarks[0]
        if side == "left":
            shoulder, hip, knee = lms[11], lms[23], lms[25]
        else:
            shoulder, hip, knee = lms[12], lms[24], lms[26]
        a = np.array([shoulder.x, shoulder.y])
        b = np.array([hip.x, hip.y])
        c = np.array([knee.x, knee.y])
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))

    def get_body_bounding_box(self, detection_result, image) -> tuple:
        """Return a bounding box (x, y, width, height) in pixel coordinates that
        encloses all visible pose landmarks.

        Args:
            detection_result: PoseLandmarkerResult from detect().
            image: BGR numpy array (used for pixel-space dimensions).
        Returns:
            tuple(int, int, int, int): (x, y, w, h), or (0, 0, 0, 0) if no pose.
        """
        if not detection_result.pose_landmarks:
            return (0, 0, 0, 0)
        lms = detection_result.pose_landmarks[0]
        h, w = image.shape[:2]
        visible = [(lm.x * w, lm.y * h) for lm in lms if lm.visibility > 0.5]
        if not visible:
            return (0, 0, 0, 0)
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        return (x1, y1, x2 - x1, y2 - y1)
