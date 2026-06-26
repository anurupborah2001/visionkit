import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


class SelfieSegmentation:
    def __init__(
        self,
        model_path: str = "./models/deeplab_v3.tflite",
        output_category_mask: bool = True,
        output_confidence_masks: bool = False,
        running_mode: VisionTaskRunningMode = VisionTaskRunningMode.IMAGE,
    ):
        base_options = BaseOptions(model_asset_path=model_path)

        self.options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=output_category_mask,
            output_confidence_masks=output_confidence_masks,
            running_mode=running_mode,
        )

        self.segmentor = vision.ImageSegmenter.create_from_options(self.options)

    #  Correct process method
    def process(self, image: np.ndarray):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = self.segmentor.segment(mp_image)
        return result

    def _get_mask(self, result, smooth=True):
        mask = result.category_mask.numpy_view()
        mask = np.squeeze(mask).astype(np.float32)
        if smooth:
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def _expand_mask(self, mask):
        return mask[..., None]

    def remove_background2(self, image: np.ndarray) -> np.ndarray:
        result = self.process(image)
        mask = self._get_mask(result)
        condition = self._expand_mask(mask > 0.5)
        return np.where(condition, image, 0)

    # Remove background
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        result = self.process(image)
        category_mask = result.category_mask.numpy_view()
        category_mask = np.squeeze(category_mask)
        condition = category_mask > 0.5
        condition = condition[..., None]
        output = np.where(condition, image, 0)
        return output

    # Blur background
    def blur_background(self, image: np.ndarray, blur_strength=(55, 55)) -> np.ndarray:
        result = self.process(image)
        category_mask = result.category_mask.numpy_view()
        blurred = cv2.GaussianBlur(image, blur_strength, 0)
        category_mask = np.squeeze(result.category_mask.numpy_view())
        condition = (category_mask > 0.5)[..., None]
        output = np.where(condition, image, blurred)
        return output

    # Replace background
    def replace_background(self, image: np.ndarray, background_path: str) -> np.ndarray:
        result = self.process(image)
        category_mask = result.category_mask.numpy_view()
        bg = cv2.imread(background_path)
        bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
        category_mask = np.squeeze(result.category_mask.numpy_view())
        condition = (category_mask > 0.5)[..., None]
        output = np.where(condition, image, bg)
        return output

    # Color background with a specified color
    def color_background(self, image: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
        result = self.process(image)
        mask = self._get_mask(result)
        bg = np.full_like(image, color, dtype=np.uint8)
        # category_mask = np.squeeze(result.category_mask.numpy_view())
        condition = (mask > 0.5)[..., None]
        output = np.where(condition, image, bg)
        return output

    def extract_foreground(self, image: np.ndarray) -> np.ndarray:
        result = self.process(image)
        mask = self._get_mask(result)
        alpha = self._expand_mask(mask)
        return (image * alpha).astype(np.uint8)

    def alpha_blend(self, image: np.ndarray, bg: np.ndarray) -> np.ndarray:
        result = self.process(image)
        mask = self._get_mask(result)
        bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
        alpha = self._expand_mask(mask)
        return (image * alpha + bg * (1 - alpha)).astype(np.uint8)

    def overlay_mask(self, image: np.ndarray) -> np.ndarray:
        """Debug visualization"""
        result = self.process(image)
        mask = self._get_mask(result, smooth=False)

        heatmap = (mask * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    def threshold_mask(self, image: np.ndarray, threshold=0.5) -> np.ndarray:
        result = self.process(image)
        mask = self._get_mask(result, smooth=False)

        binary = (mask > threshold).astype(np.uint8) * 255
        return binary

    def fast_remove_background(self, image: np.ndarray) -> np.ndarray:
        """No smoothing → faster"""
        result = self.process(image)
        mask = result.category_mask.numpy_view()
        mask = np.squeeze(mask)
        condition = (mask > 0.5)[..., None]
        return np.where(condition, image, 0)

    def fast_process(self, frame, scale=0.5):
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        result = self.process(small)
        mask = result.category_mask.numpy_view()
        mask = np.squeeze(mask)
        # Upscale mask back
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        return mask

    def optimize_virtual_background(self, frame, bg: np.ndarray):
        # Initialize persistent mask (store in class instead ideally)
        if not hasattr(self, "prev_mask"):
            self.prev_mask = None

        # 1. Downscale (use 0.7 instead of 0.5 for better clarity)
        small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

        # 2. Segment
        result = self.process(small)
        mask = result.category_mask.numpy_view()
        mask = np.squeeze(mask).astype(np.float32)

        # 3. Upscale
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # 4. Normalize mask (IMPORTANT)
        mask = np.clip(mask, 0, 1)

        # 5. Temporal smoothing (fixed)
        if self.prev_mask is not None:
            mask = 0.8 * self.prev_mask + 0.2 * mask
        self.prev_mask = mask

        # 6. Edge refinement (LESS blur, more clarity)
        mask = cv2.bilateralFilter(mask, 9, 50, 50)

        # 7. Sharpen mask slightly (optional but improves clarity)
        mask = np.clip(mask * 1.2, 0, 1)

        alpha = mask[..., None]

        # 8. Resize background
        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        # 9. Convert to float (CRITICAL)
        frame_f = frame.astype(np.float32)
        bg_f = bg.astype(np.float32)

        # 10. Blend
        output = frame_f * alpha + bg_f * (1 - alpha)

        return output.astype(np.uint8)

    def optimize_virtual_background_improved(self, frame, bg: np.ndarray):
        """
        Optimized virtual background that keeps ONLY ONE person (the largest foreground blob).
        Other people in the background are removed and replaced with the background image.
        Additionally, it includes temporal smoothing and edge refinement for better visual quality.

        Args:
          frame: Input video frame (BGR image).
          bg: Background image to replace the removed background (should be same size as frame).

        Returns:
          output: Frame with virtual background applied, keeping only the main person.
        """
        # Initialize persistent mask for temporal smoothing
        if not hasattr(self, "prev_mask"):
            self.prev_mask = None

        # 1. Downscale for faster processing
        small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

        # 2. Run MediaPipe Selfie Segmentation
        result = self.process(small)
        mask = result.category_mask.numpy_view()
        mask = np.squeeze(mask).astype(np.float32)

        # 3. Upscale mask to original frame size
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # 4. Normalize and threshold to binary (person vs background)
        mask = np.clip(mask, 0, 1)
        _, binary_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)

        # 5. NEW: Keep ONLY the largest contour (main person)
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Select the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a clean single-person mask
            single_person_mask = np.zeros_like(binary_uint8)
            cv2.drawContours(
                single_person_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
            )

            # Convert back to float [0,1]
            single_person_mask = single_person_mask.astype(np.float32) / 255.0
        else:
            single_person_mask = binary_mask  # fallback if no contours found

        # 6. Temporal smoothing (using the single-person mask)
        if self.prev_mask is not None:
            single_person_mask = 0.8 * self.prev_mask + 0.2 * single_person_mask
        self.prev_mask = single_person_mask.copy()

        # 7. Edge refinement with bilateral filter (keeps edges sharp)
        refined_mask = cv2.bilateralFilter(single_person_mask, 9, 50, 50)

        # 8. Slight sharpening for cleaner edges
        refined_mask = np.clip(refined_mask * 1.15, 0, 1)

        # 9. Prepare alpha channel
        alpha = refined_mask[..., None]

        # 10. Resize background to match frame
        bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        # 11. Blend: foreground (person) + background
        frame_f = frame.astype(np.float32)
        bg_f = bg_resized.astype(np.float32)

        output = frame_f * alpha + bg_f * (1 - alpha)

        return output.astype(np.uint8)

    def confidence_alpha_blend(self, image: np.ndarray, bg: np.ndarray) -> np.ndarray:
        result = self.process(image)

        # Use confidence mask (foreground probability)
        confidence_masks = result.confidence_masks

        if not confidence_masks:
            raise ValueError("Enable output_confidence_masks=True")

        fg_mask = confidence_masks[0].numpy_view()  # foreground prob
        fg_mask = np.squeeze(fg_mask).astype(np.float32)

        # Smooth + normalize
        fg_mask = cv2.GaussianBlur(fg_mask, (11, 11), 0)
        fg_mask = np.clip(fg_mask, 0, 1)

        alpha = fg_mask[..., None]

        bg = cv2.resize(bg, (image.shape[1], image.shape[0]))

        return (
            image.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)

    def morphological_segmentation(self, image: np.ndarray) -> np.ndarray:
        result = self.process(image)
        mask = np.squeeze(result.category_mask.numpy_view()).astype(np.float32)
        # Convert to binary
        binary = (mask > 0.5).astype(np.uint8)
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = binary[..., None]
        return np.where(binary, image, 0)

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def get_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask where 255 = foreground (person) pixels.
        Raw access without any compositing — useful when you want to apply your own logic.

        Args:
          image: BGR numpy array.
        Returns:
          Binary mask numpy array, shape (H, W), dtype uint8.
        """
        result = self.process(image)
        mask = np.squeeze(result.category_mask.numpy_view())
        return (mask > 0.5).astype(np.uint8) * 255

    def count_people(self, image: np.ndarray) -> int:
        """Estimate the number of distinct people by counting separate foreground blobs.
        Uses connected-component analysis on the segmentation mask.

        Args:
          image: BGR numpy array.
        Returns:
          int: estimated person count (0 or more).
        """
        mask = self.get_foreground_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_blob_area = image.shape[0] * image.shape[1] * 0.01  # 1 % of frame
        return sum(1 for c in contours if cv2.contourArea(c) > min_blob_area)

    def measure_foreground_ratio(self, image: np.ndarray) -> float:
        """Return the fraction of the frame occupied by the foreground (person).
        Useful for presence detection or exposure compensation.

        Args:
          image: BGR numpy array.
        Returns:
          float: 0.0–1.0
        """
        mask = self.get_foreground_mask(image)
        return float(np.sum(mask > 0)) / float(mask.size)

    def draw_foreground_contour(
        self, image: np.ndarray, color=(0, 255, 0), thickness=2
    ) -> np.ndarray:
        """Draw the outline of the detected person silhouette.

        Args:
          image: BGR numpy array.
          color: BGR contour color.
          thickness: Contour line thickness in pixels.
        Returns:
          Annotated BGR numpy array.
        """
        mask = self.get_foreground_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = image.copy()
        cv2.drawContours(out, contours, -1, color, thickness)
        return out

    def layered_background(self, image: np.ndarray, bg1, bg2) -> np.ndarray:
        result = self.process(image)

        mask = np.squeeze(result.category_mask.numpy_view()).astype(np.float32)

        bg1 = cv2.resize(bg1, (image.shape[1], image.shape[0]))
        bg2 = cv2.resize(bg2, (image.shape[1], image.shape[0]))

        # Split mask into layers
        near = np.clip(mask * 1.5, 0, 1)
        far = 1 - near

        near = near[..., None]
        far = far[..., None]
        return (image * near + bg1 * (far * 0.5) + bg2 * (far * 0.5)).astype(np.uint8)

    def blur_background2(self, image: np.ndarray) -> np.ndarray:
        # Convert BGR → RGB
        results = self.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Blur background
        blurred_image = cv2.GaussianBlur(image, (55, 55), 0)
        mask = results.category_mask.numpy_view()  # usually (H, W)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        condition = np.expand_dims(mask, axis=-1) > 0.1
        output_image = np.where(condition, image, blurred_image)
        return output_image

    # ─────────────────────── UTILITY METHODS ───────────────────────

    def is_person_present(
        self, image: np.ndarray, min_area_ratio: float = 0.01
    ) -> bool:
        """Return True if a person occupies at least min_area_ratio of the frame.

        Args:
            image: BGR numpy array.
            min_area_ratio: Minimum fraction of total pixels that must be foreground.
        Returns:
            bool: True if person is detected above the area threshold.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        fg_area = np.count_nonzero(mask > 128)
        total = mask.shape[0] * mask.shape[1]
        return bool((fg_area / total) > min_area_ratio)  # noqa: SIM901

    def get_person_center(self, image: np.ndarray) -> tuple:
        """Return the centroid (cx, cy) of the foreground person mask.

        Falls back to the image center when no foreground is detected.

        Args:
            image: BGR numpy array.
        Returns:
            tuple(int, int): Pixel coordinates (cx, cy).
        """
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        M = cv2.moments(binary)
        if M["m00"] == 0:
            return (image.shape[1] // 2, image.shape[0] // 2)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def get_foreground_bounds(self, image: np.ndarray) -> tuple:
        """Return the bounding rectangle of the foreground region.

        Args:
            image: BGR numpy array.
        Returns:
            tuple(int, int, int, int): (x, y, w, h) bounding box, or (0,0,0,0)
            when no foreground is detected.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        pts = cv2.findNonZero(binary)
        if pts is None:
            return (0, 0, 0, 0)
        return cv2.boundingRect(pts)

    def measure_foreground_height(self, image: np.ndarray) -> int:
        """Return the pixel height of the foreground bounding box.

        Args:
            image: BGR numpy array.
        Returns:
            int: Height in pixels (0 when no foreground detected).
        """
        return self.get_foreground_bounds(image)[3]

    def create_green_screen(self, image: np.ndarray) -> np.ndarray:
        """Replace the background with solid green, keeping the foreground person.

        Useful as input to chroma-key compositing pipelines.

        Args:
            image: BGR numpy array.
        Returns:
            BGR numpy array with green background.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        fg = (mask > 128)[..., np.newaxis]
        bg = np.zeros_like(image)
        bg[:] = (0, 255, 0)
        return np.where(fg, image, bg)

    def extract_foreground_on_white(self, image: np.ndarray) -> np.ndarray:
        """Place the foreground person on a pure white background.

        Args:
            image: BGR numpy array.
        Returns:
            BGR numpy array with white background.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        fg = (mask > 128)[..., np.newaxis]
        bg = np.full_like(image, 255)
        return np.where(fg, image, bg)

    def apply_bokeh_effect(
        self, image: np.ndarray, blur_radius: int = 25
    ) -> np.ndarray:
        """Apply a lens-blur (bokeh) effect to the background while keeping the
        foreground person sharp.

        Args:
            image: BGR numpy array.
            blur_radius: Gaussian kernel size (will be made odd if even).
        Returns:
            BGR numpy array with blurred background.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        r = blur_radius | 1  # ensure odd kernel size
        blurred = cv2.GaussianBlur(image, (r, r), 0)
        fg = (mask > 128)[..., np.newaxis]
        return np.where(fg, image, blurred)

    def apply_edge_glow(
        self, image: np.ndarray, color=(0, 255, 0), thickness: int = 3
    ) -> np.ndarray:
        """Draw a colored glow outline around the detected person silhouette.

        Args:
            image: BGR numpy array.
            color: BGR color tuple for the glow contour.
            thickness: Contour line thickness in pixels.
        Returns:
            Annotated BGR numpy array.
        """
        result = self.process(image)
        mask = self._get_mask(result)
        binary = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out = image.copy()
        cv2.drawContours(out, contours, -1, color, thickness)
        return out
