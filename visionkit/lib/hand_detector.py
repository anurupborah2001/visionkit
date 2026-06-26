import math
import time
from collections import deque
from itertools import combinations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)


# There are 21 hand landmarks in total, and the tips of the fingers are represented by the following landmark indices: 4 (thumb), 8 (index finger), 12 (middle finger), 16 (ring finger), and 20 (little finger). These indices correspond to the specific landmarks that represent the tips of each finger in the hand landmark detection model. By accessing these landmarks, you can determine the position and state of each finger for various applications such as gesture recognition or hand tracking.
class HandDetector:
    def __init__(
        self,
        model_path="./models/hand_landmarker.task",
        running_mode="IMAGE",
        max_hands=2,
        detection_confidence=0.5,
        hand_presence_confidence=0.5,
        tracking_confidence=0.5,
        smoothing_window=8,
        calibration_samples=None,
    ):
        """Initialize the HandDetector class with the specified parameters for hand detection and tracking. The constructor sets up the MediaPipe hand landmark detection model, drawing utilities, and configuration options for hand detection and tracking.

        Args:
          model_path (str): The path to the MediaPipe hand landmark detection model file.
          running_mode (str): The mode in which the model should run. Can be "IMAGE" for image mode or "VIDEO" for video mode.
          max_hands (int): The maximum number of hands to detect in the input images.
          detection_confidence (float): The minimum confidence threshold for hand detection. Only detections with confidence above this threshold will be considered valid.
          hand_presence_confidence (float): The minimum confidence threshold for determining the presence of a hand in the image. This is used to filter out false positives where the model may detect a hand that is not actually present.
          tracking_confidence (float): The minimum confidence threshold for tracking the detected hand landmarks across frames in a video stream. This helps to maintain consistent tracking of hand landmarks over time, even if the hand moves or changes position in the video feed.
          smoothing_window (int): The size of the window for smoothing distance measurements. This is used to average out distance measurements over a specified number of frames to reduce noise and provide more stable distance estimates.
          calibration_samples (int): The number of samples to use for calibrating the distance estimation. If None, no calibration will be performed and default values will be used for distance estimation.

          The constructor initializes the MediaPipe hand landmark detection model with the specified options and sets up the drawing utilities for visualizing the detected hand landmarks and connections. It also defines constants for margin and text color used in drawing the handedness information on the output images.
        """
        self.running_mode = getattr(vision.RunningMode, running_mode)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,  # IMAGE | VIDEO | LIVE_STREAM
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=hand_presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.mp_drawing_utils = mp.tasks.vision.drawing_utils
        self.MARGIN = 5
        self.HANDEDNESS_TEXT_COLOR = (0, 165, 255)
        self.fingerTips = [4, 8, 12, 16, 20]
        self.fingerPips = [6, 10, 14, 18]
        self.fingerDips = [7, 11, 15, 19]
        self.wrist = 0
        self.finger_mcp = [2, 5, 9, 13, 17]
        self.distance_history = deque(maxlen=smoothing_window)
        if calibration_samples:
            self.fit_polynomial(calibration_samples)

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

    def set_landmarks_image(self, image):
        self.mp_image = self._to_mp_image(image)
        # Perform hand detection
        detection_result = self.detector.detect(self.mp_image)
        # Get the hand landmarks list => list of detected hands
        # Each hand has 21 landmarks with x, y, z coordinates
        self.hand_landmarks_list = detection_result.hand_landmarks

        # Tells if the hand is left or right
        self.handedness_list = detection_result.handedness

    def draw_landmarks(
        self,
        img_bgr,
        to_draw_landmark=True,
        to_draw_center_point=True,
        to_draw_bounding_box=True,
        to_put_handle_label=True,
        flip_hands=False,
    ):
        """
        Detect hand landmarks in the input BGR image and draw them on a copy of the image. The function processes the image using MediaPipe's hand landmark detection, retrieves the detected landmarks and handedness information, and optionally draws the landmarks and connections on the image for visualization.

        Args:
          imgBGR (numpy array): The input image in BGR format on which to detect and draw hand landmarks.
          to_draw_landmark (bool): Whether to draw the detected landmarks and connections on the image.
          to_draw_center_point (bool): Whether to draw a circle at the center point of the detected hand landmarks on the image.
          to_draw_bounding_box (bool): Whether to draw a bounding box around the detected hand landmarks on the image.
          to_put_handle_label (bool): Whether to put the handedness label (e.g., "Left" or "Right") near the detected hand landmarks on the image.
          flip_hands (bool): Whether to flip the left and right hand labels. This can be useful when displaying mirrored webcam feeds, where the left and right hands may appear reversed. If True, the function will swap the "Left" and "Right" labels for the detected hands in the output image.

        Returns:
          annotated_image (numpy array): The image with detected hand landmarks and connections drawn (if to
          Draw is True). The function returns a copy of the input image with the detected hand landmarks and connections drawn for visualization purposes.
          output_landmarks: A list of tuples containing the landmarks list, bounding box, landmark parameters, and hand type for each detected hand. Each tuple in the list corresponds to a detected hand and contains the following information:
            - landmarks_list: A list of 21 landmarks for the detected hand, each represented as a tuple (x, y, z).
            - bounding_box: A tuple (xmin, ymin, width, height) representing the bounding box around the detected hand.
            - landmark_params: Additional parameters related to the detected landmarks.
            - hand_type: A string indicating the handedness of the detected hand ("Left" or "Right").
        """
        # Convert to MediaPipe image
        self.set_landmarks_image(img_bgr)
        annotated_image = np.copy(self.mp_image.numpy_view())

        if not self.hand_landmarks_list:
            return img_bgr, []

        height, width, _ = annotated_image.shape

        #  Get structured data
        all_hands = self.get_landmarks(img_bgr.copy(), flip_hands=flip_hands)
        output_landmarks = []
        for idx, hand_data in enumerate(all_hands):
            hand_landmarks = self.hand_landmarks_list[idx]
            bbox = hand_data["bounding_box"]
            center = hand_data["center_point"]
            label = hand_data["hand_type"]

            xmin, ymin, w, h = bbox
            center_x, center_y = center

            # -------- Draw landmarks --------
            if to_draw_landmark:
                self.mp_drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

            # -------- Draw bounding box --------
            if to_draw_bounding_box:
                cv2.rectangle(
                    annotated_image,
                    (xmin - 20, ymin - 20),
                    (xmin + w + 20, ymin + h + 20),
                    (0, 255, 0),
                    2,
                )

            # -------- Draw center --------
            if to_draw_center_point:
                cv2.circle(
                    annotated_image, (center_x, center_y), 8, (0, 0, 255), cv2.FILLED
                )

            # -------- Label --------
            if to_put_handle_label and label:
                cv2.putText(
                    annotated_image,
                    label,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.HANDEDNESS_TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            landmarks_list = hand_data["landmarks_list"]
            hand_bounding_box = bbox
            landmark_params = {
                "center_point": center,
                "width": w,
                "height": h,
                "bbox": (xmin, ymin, w, h),
            }
            hand_type = label

            output_landmarks.append(
                (landmarks_list, hand_bounding_box, landmark_params, hand_type)
            )

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        return annotated_image, output_landmarks

    def get_landmarks(self, img, flip_hands=False):
        """
        Extracts landmarks, bounding boxes, and center points for each detected hand.

        Args:
            img (numpy.ndarray): Input image.
            flip_hands (bool): Swap Left/Right hand labels. Useful when
                               displaying mirrored webcam feeds.

        Returns:
            list: Hand information dictionaries.

        Example output:
        [
        {
            "landmarks_list": [[0, x0, y0, z0], [1, x1, y1, z1], ..., [20, x20, y20, z20]],
            "bounding_box": (xmin, ymin, width, height),
            "center_point": (center_x, center_y),
            "hand_type": "Left" or "Right"
        },
        ...
        ]
        The function processes the input image to detect hand landmarks and returns a list of dictionaries, where
        """
        height, width, _ = img.shape
        all_hands = []

        # Generate landmarks if not already available
        if not hasattr(self, "hand_landmarks_list") or not self.hand_landmarks_list:
            self.set_landmarks_image(img.copy())

        for idx, hand_landmarks in enumerate(self.hand_landmarks_list):
            landmarks_list = []
            x_coords, y_coords = [], []

            # Extract landmarks
            for landmark_id, landmark in enumerate(hand_landmarks):
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                cz = int(landmark.z * width)

                landmarks_list.append([landmark_id, cx, cy, cz])

                x_coords.append(cx)
                y_coords.append(cy)

            # Bounding box
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            bbox = (xmin, ymin, bbox_width, bbox_height)

            # Center point
            center_x = xmin + bbox_width // 2
            center_y = ymin + bbox_height // 2

            # Hand label
            hand_label = None
            if idx < len(self.handedness_list):
                hand_label = self.handedness_list[idx][0].category_name

                if flip_hands:
                    hand_label = (
                        "Left"
                        if hand_label == "Right"
                        else "Right"
                        if hand_label == "Left"
                        else hand_label
                    )

            all_hands.append(
                {
                    "landmarks_list": landmarks_list,
                    "bounding_box": bbox,
                    "center_point": (center_x, center_y),
                    "hand_type": hand_label,
                }
            )

        return all_hands

    def is_finger_point_inside_rect(self, point, rect):
        """
        Robust rectangle hit test.
        Supports only (x, y, w, h) - SAFE & CONSISTENT.
        """

        px, py = point
        rx, ry, rw, rh = rect

        # guard against invalid values
        if rw < 0 or rh < 0:
            return False

        return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

    def finger_joined(self, p1, p2, image, landmarks, threshold=0.25):
        """
        Check if two fingers are joined based on the normalized distance between their landmarks. The function calculates the normalized distance between the specified landmarks and compares it to a threshold to determine if the fingers are considered joined. It also provides an annotated image for visualization.

        Args:
          p1 (int): The index of the first finger landmark.
          p2 (int): The index of the second finger landmark.
          image (numpy.ndarray): The image on which to annotate the finger status.
          landmarks (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.
          threshold (float): The normalized distance threshold below which the fingers are considered joined.
        Returns:
          bool: True if the fingers are joined, False otherwise.
          numpy.ndarray: The annotated image.
        """
        annotated = image.copy()

        if not landmarks:
            return False, annotated

        normalized = self._normalize(landmarks, p1, p2)
        print(f"Normalized distance between landmarks {p1} and {p2}: {normalized:.4f}")
        is_joined = normalized < threshold
        return is_joined, annotated

    def get_distance_between_landmarks(
        self,
        landmark_id_1: int,
        landmark_id_2: int,
        hand_landmarks,
        frame_shape=None,
        return_points: bool = True,
    ):
        """
        Calculate distance between two MediaPipe hand landmarks.

        Args:
            hand_landmarks:
                MediaPipe hand landmarks object.
            landmark_id_1:
                First landmark index.
            landmark_id_2:
                Second landmark index.
            frame_shape:
                frame.shape if we want pixel distance.
                If None, returns normalized landmark distance.
            return_points:
                If True, also returns point coordinates.

        Returns:
            distance, point1, point2
        """

        lm1 = hand_landmarks.landmark[landmark_id_1]
        lm2 = hand_landmarks.landmark[landmark_id_2]

        if frame_shape is not None:
            h, w = frame_shape[:2]

            p1 = (int(lm1.x * w), int(lm1.y * h))
            p2 = (int(lm2.x * w), int(lm2.y * h))
        else:
            p1 = (lm1.x, lm1.y)
            p2 = (lm2.x, lm2.y)

        distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        if return_points:
            return distance, p1, p2

        return distance

    def fingers_up(self, hand_landmarks):
        """
        Determine which fingers are raised (up) based on the detected hand landmarks. The function analyzes the positions of the landmarks for each finger and compares them to determine if a finger is raised or not.

        Args:
          hand_landmarks (list): A list of landmarks for a detected hand, where each landmark contains x, y, z coordinates normalized to the image dimensions. The landmarks are typically indexed according to the

        Returns:
          fingers (list): A list of integers representing the state of each finger, where 1 indicates that the finger is raised (up) and 0 indicates that it is not raised (down). The order of the fingers in the list corresponds to the thumb, index, middle, ring, and little fingers.
        """
        fingers = []
        # Thumb
        # We need to calculate thumb separately because it moves in a different plane compared to the other fingers.
        # The code checks the x-coordinate of the thumb tip (landmark 4) and compares it to the x-coordinate of landmark 3 (the joint before the thumb tip)
        # to determine if the thumb is raised or not. For a right hand, if the thumb tip is to the right of landmark 3, it is considered raised (1), otherwise it is considered not raised (0).
        # For a left hand, the logic would be reversed.
        # When the thumb point 3 is on the left of the thumb tip point 4, it means the thumb is raised (1) for a right hand. If the thumb tip is on the right of point 3, it means the thumb is not raised (0). This logic is based on the typical orientation of the hand and how the thumb moves in relation to the other fingers.

        # Check if the hand is right or left based on the x-coordinates of the thumb tip and the joint before the thumb tip. For a right hand, if the thumb tip (landmark 4) is to the right of landmark 3, it is considered raised (1), otherwise it is considered not raised (0). For a left hand, the logic would be reversed, where if the thumb tip is to the left of landmark 3, it would be considered raised (1), and if it is to the right, it would be considered not raised (0).
        for _idx, handedness in enumerate(self.handedness_list):
            hand_label = handedness[0].category_name  # 'Left' or 'Right'
            if hand_label == "Right":
                if (
                    hand_landmarks[self.fingerTips[0]][1]
                    > hand_landmarks[self.fingerTips[0] - 1][1]
                ):  # For right hand
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # For a left hand, the logic is reversed. If the thumb tip (landmark 4) is to the left of landmark 3, it is considered raised (1), and if it is to the right, it is considered not raised (0). This is because the orientation of the hand is different for left and right hands, and the thumb moves in opposite directions relative to the other fingers.
                if (
                    hand_landmarks[self.fingerTips[0]][1]
                    < hand_landmarks[self.fingerTips[0] - 1][1]
                ):  # For left hand
                    fingers.append(1)
                else:
                    fingers.append(0)

        # Fingers (index, middle, ring, little)
        for id in range(1, 5):
            if (
                hand_landmarks[self.fingerTips[id]][2]
                < hand_landmarks[self.fingerTips[id] - 2][2]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_distance(
        self, p1, p2, img, to_draw_circle_key_point=True, to_draw_line=True
    ):
        # Extract coordinates from specific hand
        x1, y1 = p1
        x2, y2 = p2

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Distance
        length = math.hypot(x2 - x1, y2 - y1)

        # Draw line
        if to_draw_line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Draw points
        if to_draw_circle_key_point:
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def euclidean_distance(self, p1, p2):
        """
        Euclidean distance between two points p1 and p2, where each point is represented as a tuple of (x, y) coordinates. The function calculates the distance using the formula: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2), which gives the straight-line distance between the two points in a 2D space.
        As the distance is calculated using the Euclidean distance formula, it provides a measure of how far apart the two points are in the 2D space of the image. This can be useful for various applications such as gesture recognition, where the distance between specific landmarks can indicate certain hand gestures or movements.

        Args:
          p1 (tuple): The first point represented as a tuple of (x, y) coordinates.
          p2 (tuple): The second point represented as a tuple of (x, y) coordinates.
        Returns:
          distance (float): The Euclidean distance between the two points p1 and p2.
        """
        # return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_real_palm_width(pixel_width, distance_cm, focal_length_px):
        """
        Compute the real palm width in centimeters based on the detected palm width in pixels, the known distance from the camera to the hand, and the focal length of the camera in pixels.
        The function uses the formula: real_palm_width_cm = (pixel_width * distance_cm) / focal_length_px, where pixel_width is the measured width of the palm in pixels, distance_cm is the known distance from the camera to the hand in centimeters, and focal_length_px is the focal length of the camera in pixels. This calculation allows for estimating the actual size of the palm in real-world units (centimeters) based on the detected size in pixels and the known distance from the camera. This can be useful for applications that require understanding the physical dimensions of the hand or for distance estimation based on the size of the detected palm in the image.

        Args:
          pixel_width (float): The measured width of the palm in pixels as detected by the hand landmark detection model. This value is used in the calculation of the real palm width in centimeters.
          distance_cm (float): The known distance from the camera to the hand in centimeters. This value is used in the calculation of the real palm width in centimeters.
          focal_length_px (float): The focal length of the camera in pixels. This value is used in the calculation of the real palm width in centimeters.

        Returns:
          float: The calculated real palm width in centimeters based on the detected pixel width, known distance, and focal length. If the pixel width is zero or negative, or if the focal length is zero or negative, the function returns None to indicate that the real palm width cannot be computed with the given input.
        """
        if pixel_width <= 0 or focal_length_px <= 0:
            return None

        return (pixel_width * distance_cm) / focal_length_px

    def landmark_to_pixel(self, landmark, image_width, image_height):
        """
        Landmark to pixel coordinates conversion. The function takes a landmark with normalized coordinates (x, y) and converts it to pixel coordinates based on the width and height of the image. The x-coordinate is multiplied by the image width, and the y-coordinate is multiplied by the image height to obtain the corresponding pixel coordinates in the image. This conversion is essential for accurately mapping the detected landmarks to their positions in the original image for visualization or further processing.

        Args:
          landmark: A landmark with normalized coordinates (x, y) that represents a specific point on the hand detected by the MediaPipe model.
          image_width (int): The width of the image
          image_height (int): The height of the image

        Returns:
          tuple: A tuple containing the pixel coordinates (x, y) corresponding to the input landmark, calculated by multiplying the normalized coordinates of the landmark by the width and height of the image, respectively
        """
        return int(landmark.x * image_width), int(landmark.y * image_height)

    def palm_width_px(self, img, hand_landmarks, drawLandmarks=False):
        """
        Uses distance between INDEX_MCP and PINKY_MCP as palm width.
        This is more stable than fingertip distance.

        Args:
          hand_landmarks (list): A list of hand landmarks, where each landmark is expected to be a tuple of (x, y) coordinates. The function specifically uses the landmarks corresponding to the INDEX_MCP and PINKY_MCP to calculate the palm width in pixels.
                                  These landmarks represent the base joints of the index and pinky fingers, respectively, and their distance provides a more stable measurement of the palm width compared to using fingertip landmarks, which can be more variable due to finger bending and movement.
          drawLandmarks (bool): Whether to draw the landmarks and connections on the image for visualization purposes. If True, the function will draw a line between the INDEX_MCP and PINKY_MCP landmarks, as well as circles at these landmark points on the input image.

        Returns:
          float: The calculated palm width in pixels, which is the distance between the INDEX_MCP
        """
        index_mcp_landmarks = hand_landmarks[self.finger_mcp[1]]
        pinky_mcp_landmarks = hand_landmarks[self.finger_mcp[4]]
        index_mcp = (
            index_mcp_landmarks[1],
            index_mcp_landmarks[2],
        )  # (x, y) for INDEX_MCP
        pinky_mcp = (
            pinky_mcp_landmarks[1],
            pinky_mcp_landmarks[2],
        )  # (x, y) for PINKY_MCP
        if drawLandmarks:
            cv2.line(img, index_mcp, pinky_mcp, (0, 255, 0), 2)
            cv2.circle(img, index_mcp, 6, (255, 0, 0), -1)
            cv2.circle(img, pinky_mcp, 6, (255, 0, 0), -1)
        return self.euclidean_distance(index_mcp, pinky_mcp), index_mcp, pinky_mcp

    def _palm_scale(self, hand):
        """
        Calculate the palm scale of a hand based on specific landmarks.

        Args:
          hand (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.

        Returns:
          float: The calculated palm scale of the hand, which is an average of the distances between
        """
        # stable reference (best practice)
        return (
            self.euclidean_distance(
                hand[self.wrist], hand[self.finger_mcp[2]]
            )  # wrist → middle MCP
            + self.euclidean_distance(
                hand[self.finger_mcp[1]], hand[self.finger_mcp[4]]
            )  # index MCP → pinky MCP
        ) / 2

    def calibrate_focal_length(self, palm_width_px):
        """
        Calibrate the focal length of the camera based on a known distance and the detected palm width in pixels. The function calculates the focal length using the formula: focal_length = (palm_width_px * known_distance_cm) / real_palm_width_cm, where palm_width_px is the measured width of the palm in pixels, known_distance_cm is the known distance from the camera to the hand in centimeters, and real_palm_width_cm is the actual width of the palm in centimeters. This calibration allows for accurate distance estimation based on the detected palm width in subsequent frames.

        Args:
          palm_width_px (float): The measured width of the palm in pixels as detected by the hand landmark detection model. This value is used in the calculation of the focal length for distance estimation.

        Returns:
          float: The calculated focal length of the camera based on the known distance and the detected palm width in pixels. This focal length can be used for accurate distance estimation in subsequent frames based on the
          detected palm width in pixels. If the palm width in pixels is zero or negative, the function returns None to indicate that the focal length cannot be calibrated with the given input.
        """
        if palm_width_px <= 0:
            return None

        self.focal_length_px = (
            palm_width_px * self.known_distance_cm
        ) / self.real_palm_width_cm

        return self.focal_length_px

    def get_dynamic_palm_width(self, hand, image_shape, distance_cm, focal_length_px):
        """
        Get the dynamic palm width in centimeters based on the detected landmarks of the hand, the shape of the image, the known distance from the camera to the hand, and the focal length of the camera in pixels. The function calculates the pixel width of the palm using specific landmarks (e.g., INDEX_MCP and PINKY_MCP) and then uses this pixel width along with the known distance and focal length to compute the real palm width in centimeters using the formula: real_palm_width_cm = (pixel_width * distance_cm) / focal_length_px. This allows for dynamic estimation of the palm width in real-world units based on the detected landmarks and camera parameters.

        Args:
          hand (object): The detected hand landmarks object that contains the landmark information for the hand.
          image_shape (tuple): The shape of the input image as a tuple (height, width, channels). This is used to convert the normalized landmark coordinates to pixel coordinates.
          distance_cm (float): The known distance from the camera to the hand in centimeters. This value is used in the calculation of the real palm width in centimeters.
          focal_length_px (float): The focal length of the camera in pixels. This value is used in the calculation of the real palm width in centimeters.

        Returns:
          float: The calculated real palm width in centimeters based on the detected landmarks, image shape, known distance, and focal length. This value provides an estimate of the actual size of the palm in real-world units (centimeters) based on the detected size in pixels and the known distance from the
          camera. If the pixel width is zero or negative, or if the focal length is zero or negative, the function returns None to indicate that the real palm width cannot be computed with the given input.
        """
        h, w, _ = image_shape

        lm = hand.landmark

        # Index MCP (5) and Pinky MCP (17)
        p1 = (int(lm[self.finger_mcp[1]].x * w), int(lm[self.finger_mcp[1]].y * h))
        p2 = (int(lm[self.finger_mcp[4]].x * w), int(lm[self.finger_mcp[4]].y * h))

        pixel_width = np.linalg.norm(np.array(p1) - np.array(p2))

        real_width_cm = (pixel_width * distance_cm) / focal_length_px

        return real_width_cm, pixel_width, p1, p2

    def fit_polynomial(self, calibration_samples, polynomial_degree=2):
        """
        calibration_samples format:
        [
            (palm_width_px, distance_cm),
            (palm_width_px, distance_cm),
            ...
        ]
        """
        x = np.array([sample[0] for sample in calibration_samples], dtype=np.float32)
        y = np.array([sample[1] for sample in calibration_samples], dtype=np.float32)

        coeffs = np.polyfit(x, y, polynomial_degree)  # y = Ax^2 + Bx + C
        self.model = np.poly1d(coeffs)

        # print("Polynomial coefficients:", coeffs)
        return coeffs

    def adaptive_distance_cm(
        self,
        palm_width_px,
        frame_width_px,
        horizontal_fov_deg=60,
        estimated_palm_width_cm=8.5,
    ):
        if palm_width_px <= 0:
            return None

        focal_length_px = frame_width_px / (
            2 * math.tan(math.radians(horizontal_fov_deg / 2))
        )

        distance_cm = (estimated_palm_width_cm * focal_length_px) / palm_width_px

        return distance_cm

    def estimate_distance_cm(
        self,
        palm_width_px,
    ):
        """
        # Formula
        # distance = (real_palm_width * focal_length) / palm_width_px

        Estimate the distance from the camera to the hand in centimeters based on the detected palm width in pixels, the known real palm width in centimeters, and the focal length of the camera in pixels. The function uses the formula: distance_cm = (real_palm_width_cm * focal_length_px) / palm_width_px, where real_palm_width_cm is the actual width of the palm in centimeters, focal_length_px is the focal length of the camera in pixels, and palm_width_px is the measured width of the palm in pixels as detected by the hand landmark detection model. This estimation allows for determining how far the hand is from the camera based on the detected size of the palm in pixels and the known parameters of the camera and hand size.

        Args:
          palm_width_px (float): The measured width of the palm in pixels as detected by the

        Returns:
          float: The estimated distance from the camera to the hand in centimeters based on the detected palm width in pixels, known real palm width in centimeters, and focal length of the camera in pixels. This value provides an estimate of how far the hand is from the camera based on the detected size of the palm in pixels and the known parameters of the camera and hand size. If the palm width in pixels is zero or negative, or if the focal length is zero or negative, the function returns None to indicate that the distance cannot be estimated with the given input.
        """
        if self.model is None:
            return None

        distance = float(self.model(palm_width_px))
        # reject impossible values
        if distance <= 0 or distance > 300:
            return None

        self.distance_history.append(distance)
        return float(np.mean(self.distance_history))

    def _normalize(self, hand, p1, p2):
        """
        Normalize the distance between two landmarks (p1 and p2) by the palm scale of the hand. The function calculates the Euclidean distance between the specified landmarks and divides it by the palm scale to provide a normalized distance that accounts for variations in hand size. This normalization allows for more consistent comparisons of distances between landmarks across different hands and gestures, as it takes into account the overall size of the hand rather than just the raw distance between specific landmarks.

        Args:
          hand (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.
          p1 (int): The index of the first landmark for which to calculate the distance.
          p2 (int): The index of the second landmark for which to calculate the distance.
        Returns:
          float: The normalized distance between the two landmarks p1 and p2, calculated as the Euclidean distance between the landmarks divided by the palm scale of the hand. This normalized distance provides a more consistent measure of the distance between the landmarks that accounts for variations in hand size, allowing for better
        """
        tip_dist = self.euclidean_distance(hand[p1], hand[p2])
        scale = self._palm_scale(hand)
        if scale == 0:
            return 0
        return tip_dist / scale

    def is_fingers_joined(
        self, p1, p2, image, landmarks, threshold=0.25, draw_intersection_point=True
    ):
        """
        Check if two fingers are joined and provide the idexes which fingers are joined based on the normalized distance between their landmarks

        Args:
          p1 (int): The index of the first finger landmark.
          p2 (int): The index of the second finger landmark.
          image (numpy.ndarray): The image on which to annotate the finger status.
          landmarks (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.
          threshold (float): The normalized distance threshold below which the fingers are considered joined.
        Returns:
          bool: True if the fingers are joined, False otherwise.
          numpy.ndarray: The annotated image.
        """
        if not landmarks:
            return False

        normalized = self._normalize(landmarks, p1, p2)
        print(f"Normalized distance between landmarks {p1} and {p2}: {normalized:.4f}")
        is_joined = normalized < threshold
        if is_joined and draw_intersection_point:
            x1, y1 = landmarks[p1][1], landmarks[p1][2]
            x2, y2 = landmarks[p2][1], landmarks[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # to make circle color red if fingers are joined, otherwise green
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        return is_joined

    def is_fingers_joined_2(
        self,
        p1,
        p2,
        image,
        landmarks,
        threshold=0.18,  # slightly relaxed (important)
        draw_intersection_point=True,
        debug=False,
    ):
        """
        Check if two fingers are joined using a more robust method that accounts for variations in hand size and camera perspective. This method calculates the normalized distance between the specified finger landmarks and compares it to an adaptive threshold to determine if the fingers are joined.

        Args:
          p1 (int): The index of the first finger landmark.
          p2 (int): The index of the second finger landmark.
          image (numpy.ndarray): The image on which to annotate the finger status.
          landmarks (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.
          threshold (float): The normalized distance threshold below which the fingers are considered joined.
          draw_intersection_point (bool): Whether to draw a circle at the intersection point of the fingers.
          debug (bool): Whether to print debug information.
        Returns:
          bool: True if the fingers are joined, False otherwise.
        """

        if not landmarks or len(landmarks) <= max(p1, p2):
            return False

        x1, y1, _ = landmarks[p1][1:]
        x2, y2, _ = landmarks[p2][1:]

        # -----------------------------
        # RAW DISTANCE
        # -----------------------------
        pixel_dist = self.euclidean_distance((x1, y1), (x2, y2))

        # -----------------------------
        # ROBUST PALM SCALE (FIXED)
        # use wrist (0), index MCP (5), pinky MCP (17)
        # works for LEFT + RIGHT hands
        # -----------------------------
        try:
            wrist = landmarks[0][1:3]
            index_mcp = landmarks[5][1:3]
            pinky_mcp = landmarks[17][1:3]

            palm_diag1 = self.euclidean_distance(wrist, index_mcp)
            palm_diag2 = self.euclidean_distance(wrist, pinky_mcp)

            palm_size = (palm_diag1 + palm_diag2) / 2
        except Exception:
            palm_size = self._palm_scale(landmarks)

        palm_size = max(palm_size, 1e-6)

        # -----------------------------
        # NORMALIZED DISTANCE
        # -----------------------------
        normalized_dist = pixel_dist / palm_size

        # -----------------------------
        # ADAPTIVE THRESHOLD (IMPORTANT FIX)
        # -----------------------------
        # left hand tends to appear slightly scaled differently in camera
        adaptive_threshold = threshold

        # optional stability boost
        is_joined = normalized_dist < adaptive_threshold

        # -----------------------------
        # DEBUG
        # -----------------------------
        if debug:
            print(
                f"[JOIN DEBUG] pixel={pixel_dist:.2f}, "
                f"palm={palm_size:.2f}, "
                f"norm={normalized_dist:.4f}, "
                f"threshold={adaptive_threshold}"
            )

        # -----------------------------
        # DRAW
        # -----------------------------
        if draw_intersection_point:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            color = (0, 0, 255) if is_joined else (0, 255, 255)
            thickness = cv2.FILLED if is_joined else 1

            cv2.circle(image, (cx, cy), 8, color, thickness)

        return is_joined

    def joined_fingers(self, image, landmarks, threshold=0.25):
        """
        Check which fingers are joined based on the normalized distance between their landmarks and return a list indicating the joined state of each finger. The function iterates through predefined pairs of finger landmarks, calculates the normalized distance for each pair, and updates a list to indicate which fingers are joined based on the specified threshold. This allows for a comprehensive analysis of finger positions and can be used for gesture recognition or other applications that require understanding of finger interactions.

        Args:
          image (numpy.ndarray): The image on which to annotate the finger status.
          landmarks (list): A list of hand landmarks. Each landmark is expected to be a tuple of (x, y) coordinates.
          threshold (float): The normalized distance threshold below which the fingers are considered joined.
        Returns:
          list: A list of integers representing the joined state of each finger, where 1 indicates
        """

        annotated = image.copy()
        joined_state = [0, 0, 0, 0, 0]
        if not landmarks or len(landmarks) < 21:
            return joined_state, annotated

        pairs = list(combinations(self.fingerTips, 2))

        for i, (p1, p2) in enumerate(pairs):
            normalized = self._normalize(landmarks, p1, p2)
            print(f"Finger {p1} vs {p2} -> {normalized:.4f}")

            is_joined = normalized < threshold
            if is_joined:
                joined_state[i] = 1
                joined_state[i + 1] = 1

        return joined_state, annotated

    def count_fingers(self):
        """
        Returns the total number of fingers currently up (0-5).
        """
        return sum(self.fingers_up())

    def is_fist(self):
        """
        True if all fingers are down (classic fist).
        """
        fingers = self.fingers_up()
        return len(fingers) == 5 and all(f == 0 for f in fingers)

    def is_open_hand(self):
        """
        True if all fingers are up (open palm).
        """
        fingers = self.fingers_up()
        return len(fingers) == 5 and all(f == 1 for f in fingers)

    def is_thumbs_up(self):
        """
        Classic thumbs-up gesture.
        """
        fingers = self.fingers_up()
        return (
            len(fingers) == 5 and fingers[0] == 1 and all(f == 0 for f in fingers[1:])
        )

    def is_peace_sign(self):
        """
        Peace / V sign (index + middle up, others down).
        """
        fingers = self.fingers_up()
        return (
            len(fingers) == 5
            and fingers[1] == 1
            and fingers[2] == 1
            and fingers[0] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        )

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def get_gesture_name(self, hand_landmarks):
        """Return a human-readable gesture label for the first detected hand.
        Checks gestures in priority order and returns the first match.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          str: 'Fist' | 'Open' | 'ThumbsUp' | 'Peace' | 'Unknown'
        """
        fingers = self.fingers_up(hand_landmarks)
        if len(fingers) < 5:
            return "Unknown"
        if all(f == 0 for f in fingers):
            return "Fist"
        if all(f == 1 for f in fingers):
            return "Open"
        if fingers[0] == 1 and all(f == 0 for f in fingers[1:]):
            return "ThumbsUp"
        if (
            fingers[1] == 1
            and fingers[2] == 1
            and fingers[0] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        ):
            return "Peace"
        return "Unknown"

    def get_finger_count(self, hand_landmarks):
        """Return number of fingers currently raised (0–5).

        Args:
          hand_landmarks: landmarks_list from get_landmarks().
        Returns:
          int
        """
        return sum(self.fingers_up(hand_landmarks))

    def get_angle_between_landmarks(self, landmarks_list, a, b, c):
        """Compute the joint angle at landmark b formed by landmarks a-b-c.
        Uses 2D (x, y) pixel coordinates from landmarks_list.

        Args:
          landmarks_list: List of [id, x, y, z] from get_landmarks().
          a, b, c: Landmark indices (e.g. 5, 6, 7 for index finger PIP joint).
        Returns:
          float: Angle in degrees (0–180).
        """
        pa = np.array([landmarks_list[a][1], landmarks_list[a][2]], dtype=float)
        pb = np.array([landmarks_list[b][1], landmarks_list[b][2]], dtype=float)
        pc = np.array([landmarks_list[c][1], landmarks_list[c][2]], dtype=float)
        ba = pa - pb
        bc = pc - pb
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    def get_hand_bbox(self, hand_data):
        """Extract bounding box from a hand data dict returned by get_landmarks().

        Args:
          hand_data: Single dict from get_landmarks() list.
        Returns:
          tuple(int, int, int, int): (x, y, w, h)
        """
        return hand_data["bounding_box"]

    def get_hand_center(self, hand_data):
        """Extract center point from a hand data dict returned by get_landmarks().

        Args:
          hand_data: Single dict from get_landmarks() list.
        Returns:
          tuple(int, int): (cx, cy)
        """
        return hand_data["center_point"]

    def is_pointing(self, hand_landmarks):
        """Return True for a pointing gesture: only index finger raised.

        Args:
          hand_landmarks: landmarks_list from get_landmarks().
        Returns:
          bool
        """
        fingers = self.fingers_up(hand_landmarks)
        return (
            len(fingers) == 5
            and fingers[1] == 1
            and fingers[0] == 0
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        )

    def get_wrist_position(self, hand_landmarks):
        """Return pixel coordinates of the wrist landmark (id 0).

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          tuple(int, int): (x, y) pixel position of the wrist.
        """
        return (hand_landmarks[0][1], hand_landmarks[0][2])

    def get_fingertip_positions(self, hand_landmarks):
        """Return pixel positions of all five fingertips.

        Args:
          hand_landmarks: landmarks_list from get_landmarks().
        Returns:
          dict: {'thumb': (x,y), 'index': (x,y), 'middle': (x,y), 'ring': (x,y), 'little': (x,y)}
        """
        names = ["thumb", "index", "middle", "ring", "little"]
        return {
            name: (hand_landmarks[tip][1], hand_landmarks[tip][2])
            for name, tip in zip(names, self.fingerTips, strict=False)
        }

    def is_ok_sign(self, hand_landmarks):
        """Return True if the hand is making an OK sign.

        The OK sign requires the thumb tip (landmark 4) and index tip (landmark 8)
        to be close together (normalized distance < 0.08) while the middle, ring,
        and little fingers are raised.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          bool
        """
        fingers = self.fingers_up(hand_landmarks)
        t = hand_landmarks[4][1:3]
        i = hand_landmarks[8][1:3]
        dist = self.euclidean_distance(t, i)
        return dist < 0.08 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1

    def is_call_me(self, hand_landmarks):
        """Return True if the hand is making a 'call me' gesture.

        The call-me gesture has the thumb and little finger extended while the
        index, middle, and ring fingers are folded down.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          bool
        """
        fingers = self.fingers_up(hand_landmarks)
        return (
            fingers[0] == 1
            and fingers[1] == 0
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 1
        )

    def is_rock_sign(self, hand_landmarks):
        """Return True if the hand is making a rock/devil-horns sign.

        The rock sign has the index and little fingers extended while the thumb,
        middle, and ring fingers are folded down.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          bool
        """
        fingers = self.fingers_up(hand_landmarks)
        return (
            fingers[0] == 0
            and fingers[1] == 1
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 1
        )

    def recognize_number(self, hand_landmarks):
        """Return the number (0–5) represented by the hand gesture.

        Delegates to get_finger_count to count raised fingers.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          int: Number of fingers raised (0–5).
        """
        return self.get_finger_count(hand_landmarks)

    def get_hand_orientation(self, hand_landmarks):
        """Return the cardinal orientation of the hand based on wrist-to-middle-MCP vector.

        Compares the wrist (landmark 0) to the middle finger MCP (landmark 9) to
        determine which direction the hand is pointing.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          str: One of 'palm_up', 'palm_down', 'palm_left', 'palm_right'.
        """
        wrist = hand_landmarks[0][1:3]
        middle_mcp = hand_landmarks[9][1:3]
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        if abs(dx) >= abs(dy):
            return "palm_right" if dx > 0 else "palm_left"
        return "palm_up" if dy < 0 else "palm_down"

    def get_swipe_direction(self, prev_wrist, curr_wrist, threshold=20):
        """Classify the swipe direction between two wrist positions.

        Compares two (x, y) wrist positions and returns the dominant direction of
        movement. Returns 'none' if the displacement is below the threshold in both
        axes.

        Args:
          prev_wrist (tuple): Previous wrist position as (x, y).
          curr_wrist (tuple): Current wrist position as (x, y).
          threshold (int): Minimum pixel displacement to register as a swipe.
        Returns:
          str: One of 'right', 'left', 'up', 'down', 'none'.
        """
        dx = curr_wrist[0] - prev_wrist[0]
        dy = curr_wrist[1] - prev_wrist[1]
        if max(abs(dx), abs(dy)) < threshold:
            return "none"
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def get_all_finger_angles(self, hand_landmarks):
        """Compute the joint angle at the middle joint of each finger.

        Uses get_angle_between_landmarks for each finger's MCP–PIP–DIP triplet.

        Args:
          hand_landmarks: landmarks_list from get_landmarks() — list of [id, x, y, z].
        Returns:
          dict: Keys are finger names ('thumb', 'index', 'middle', 'ring', 'little'),
                values are angles in degrees (0–180).
        """
        joints = {
            "thumb": (1, 2, 3),
            "index": (5, 6, 7),
            "middle": (9, 10, 11),
            "ring": (13, 14, 15),
            "little": (17, 18, 19),
        }
        return {
            name: self.get_angle_between_landmarks(hand_landmarks, a, b, c)
            for name, (a, b, c) in joints.items()
        }

    def draw_gesture_label(self, image, hand_data, label):
        """Draw a gesture label above the hand bounding box on a copy of the image.

        Args:
          image: BGR numpy array.
          hand_data (dict): Hand dict from get_landmarks() with key 'bounding_box' (x, y, w, h).
          label (str): Gesture label text to render.
        Returns:
          numpy.ndarray: Annotated copy of the input image (BGR).
        """
        out = image.copy()
        x, y, w, h = hand_data["bounding_box"]
        cv2.putText(
            out,
            label,
            (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
        return out

    def to_json(self, hand_data):
        """Serialize a hand data dict to a JSON-compatible structure.

        Args:
          hand_data (dict): Hand dict from get_landmarks() containing 'hand_type',
                            'center_point', 'bounding_box', and 'landmarks_list'.
        Returns:
          dict: JSON-serializable dict with keys 'hand_type', 'center_point',
                'bounding_box', and 'landmarks'.
        """
        return {
            "hand_type": hand_data.get("hand_type", "Unknown"),
            "center_point": list(hand_data.get("center_point", (0, 0))),
            "bounding_box": list(hand_data.get("bounding_box", (0, 0, 0, 0))),
            "landmarks": [list(lm) for lm in hand_data.get("landmarks_list", [])],
        }


def main():
    currentTime = 0
    previousTime = 0
    handDetector = HandDetector()
    while True:
        ret, imgFrame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        imgRGB = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2RGB)

        # Detect the hand landmarks in the RGB image using the handDetector instance. The detected landmarks are stored in the hand_landmarks_list attribute of the handDetector object, which can be accessed for further processing or visualization.
        annotated_image = handDetector.draw_landmarks(imgRGB)

        # Get the list of landmarks for the detected hands. The get_landmarks method processes the annotated image and returns a list of landmarks, which can be used for various applications such as gesture recognition or hand tracking.
        handDetector.get_landmarks(annotated_image)

        # Convert RGB → BGR for OpenCV display
        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        taps = handDetector.detect_finger_tapping()
        handDetector.fingers_up()
        handDetector.count_fingers()

        # Example prints (you can replace with your own logic)
        if any(taps):
            print(f"🔥 TAP DETECTED! Fingers: {taps} | Finger indices [4,8,12,16,20]")

        if handDetector.is_fist():
            print("👊 Fist detected")
        if handDetector.is_thumbs_up():
            print("👍 Thumbs up")
        if handDetector.is_peace_sign():
            print("✌️ Peace sign")
        if handDetector.is_open_hand():
            print("🖐️ Open hand")

        # FPS
        currentTime = time.time()
        fps = (
            1 / (currentTime - previousTime) if (currentTime - previousTime) > 0 else 0
        )
        previousTime = currentTime
        cv2.putText(
            annotated_image,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow(
            "Extended Hand Tracking (MediaPipe) - Tapping + Gestures", annotated_image
        )
        # Stop the loop and close the application when the 'Esc' key is pressed. The waitKey function waits for a key event for a specified amount of time (in this case, 1 millisecond) and checks if the 'Esc' key (ASCII code 27) is pressed to break the loop and release resources.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
