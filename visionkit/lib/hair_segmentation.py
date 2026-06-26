from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

_MODEL_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL = str(_MODEL_DIR / "hair_segmenter.tflite")


class HairSegmentation:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        output_category_mask: bool = True,
        output_confidence_masks: bool = False,
        running_mode: VisionTaskRunningMode = VisionTaskRunningMode.IMAGE,
    ):
        base_options = BaseOptions(model_asset_path=model_path)

        self.options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=output_category_mask,
            output_confidence_masks=output_confidence_masks,
            running_mode=running_mode,
        )

        self.segmentor = vision.ImageSegmenter.create_from_options(self.options)

    def process(self, image: np.ndarray):
        """Segment hair in an RGB image and return the raw MediaPipe result.

        Args:
          image: RGB numpy array (NOT BGR — convert with cv2.cvtColor first).
        Returns:
          MediaPipe ImageSegmenterResult with category_mask.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        return self.segmentor.segment(mp_image)

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def get_hair_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask where 255 = hair pixels.
        Accepts BGR input (standard OpenCV format) and converts internally.

        Args:
          bgr_image: BGR numpy array.
        Returns:
          Binary mask numpy array, shape (H, W), dtype uint8.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self.process(rgb)
        mask = np.squeeze(result.category_mask.numpy_view())
        return (mask > 0.5).astype(np.uint8) * 255

    def recolor_hair(self, bgr_image: np.ndarray, color=(180, 0, 255)) -> np.ndarray:
        """Replace hair pixels with a solid color.

        Args:
          bgr_image: BGR numpy array.
          color: BGR color tuple to paint over hair.
        Returns:
          BGR numpy array with hair region recolored.
        """
        mask = self.get_hair_mask(bgr_image)
        out = bgr_image.copy()
        out[mask > 0] = color
        return out

    def blend_hair_color(
        self, bgr_image: np.ndarray, color=(180, 0, 255), alpha=0.5
    ) -> np.ndarray:
        """Alpha-blend a color over the hair region — preserves texture unlike recolor_hair.

        Args:
          bgr_image: BGR numpy array.
          color: BGR color tuple for the hair overlay.
          alpha: Blend strength (0 = no change, 1 = solid color).
        Returns:
          BGR numpy array with hair color blended.
        """
        mask = self.get_hair_mask(bgr_image)
        overlay = bgr_image.copy()
        overlay[mask > 0] = color
        hair_region = mask > 0
        out = bgr_image.copy().astype(np.float32)
        out[hair_region] = (1 - alpha) * bgr_image[hair_region].astype(
            np.float32
        ) + alpha * np.array(color, dtype=np.float32)
        return out.astype(np.uint8)

    def get_hair_ratio(self, bgr_image: np.ndarray) -> float:
        """Return the fraction of image pixels classified as hair.

        Args:
          bgr_image: BGR numpy array.
        Returns:
          float: 0.0–1.0 (e.g. 0.15 means 15 % of the frame is hair).
        """
        mask = self.get_hair_mask(bgr_image)
        return float(np.sum(mask > 0)) / float(mask.size)

    def visualize(
        self, bgr_image: np.ndarray, color=(180, 0, 255), alpha=0.5
    ) -> np.ndarray:
        """Convenience wrapper: blend hair color and return the annotated frame.
        Equivalent to blend_hair_color() with default settings.

        Args:
          bgr_image: BGR numpy array.
          color: BGR highlight color for hair.
          alpha: Overlay transparency (0–1).
        Returns:
          Annotated BGR numpy array.
        """
        return self.blend_hair_color(bgr_image, color=color, alpha=alpha)

    def draw_hair_contours(
        self, bgr_image: np.ndarray, color=(0, 255, 255), thickness=2
    ) -> np.ndarray:
        """Draw the outline of the detected hair region on the image.
        Useful for debugging segmentation quality.

        Args:
          bgr_image: BGR numpy array.
          color: BGR contour color.
          thickness: Contour line thickness in pixels.
        Returns:
          Annotated BGR numpy array.
        """
        mask = self.get_hair_mask(bgr_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = bgr_image.copy()
        cv2.drawContours(out, contours, -1, color, thickness)
        return out

    def get_hair_bounding_box(self, bgr_image: np.ndarray):
        """Return the bounding rect of the hair region as (x, y, w, h).

        Args:
          bgr_image: BGR numpy array.
        Returns:
          Tuple (x, y, w, h) in pixels; (0, 0, 0, 0) if no hair detected.
        """
        mask = self.get_hair_mask(bgr_image)
        pts = cv2.findNonZero(mask)
        if pts is None:
            return (0, 0, 0, 0)
        return cv2.boundingRect(pts)

    def get_hair_top_position(self, bgr_image: np.ndarray) -> int:
        """Return the y-coordinate of the topmost hair pixel.

        Args:
          bgr_image: BGR numpy array.
        Returns:
          int: Row index of the first hair pixel from the top; 0 if no hair.
        """
        mask = self.get_hair_mask(bgr_image)
        rows = np.any(mask > 0, axis=1)
        if not rows.any():
            return 0
        return int(np.argmax(rows))

    def detect_hair_length_estimate(self, bgr_image: np.ndarray) -> str:
        """Estimate hair length category based on vertical extent of hair mask.

        Thresholds (fraction of image height):
          < 15% → "short"
          < 35% → "medium"
          >= 35% → "long"
          no hair → "none"

        Args:
          bgr_image: BGR numpy array.
        Returns:
          str: One of "none", "short", "medium", "long".
        """
        x, y, w, h = self.get_hair_bounding_box(bgr_image)
        if h == 0:
            return "none"
        ratio = h / bgr_image.shape[0]
        if ratio < 0.15:
            return "short"
        if ratio < 0.35:
            return "medium"
        return "long"

    def get_hair_density_map(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return a density heatmap of hair coverage as a uint8 image.

        Applies a Gaussian blur to the binary hair mask to produce a smooth
        density map where brighter pixels indicate denser hair regions.

        Args:
          bgr_image: BGR numpy array.
        Returns:
          numpy array of shape (H, W), dtype uint8, values 0–255.
        """
        mask = self.get_hair_mask(bgr_image)
        density = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
        max_val = density.max()
        if max_val > 0:
            density = density / max_val * 255
        return density.astype(np.uint8)

    def apply_gradient_color(self, bgr_image: np.ndarray, color1, color2) -> np.ndarray:
        """Paint a vertical gradient over the hair region.

        color1 is applied at the top of the bounding box and color2 at the
        bottom; pixels are linearly interpolated between the two colors.

        Args:
          bgr_image: BGR numpy array.
          color1: BGR tuple for the top of the gradient.
          color2: BGR tuple for the bottom of the gradient.
        Returns:
          BGR numpy array with gradient applied to hair pixels.
        """
        mask = self.get_hair_mask(bgr_image)
        x, y, w, h = self.get_hair_bounding_box(bgr_image)
        if h == 0:
            return bgr_image.copy()
        out = bgr_image.copy()
        for row in range(y, min(y + h, bgr_image.shape[0])):
            t = (row - y) / h
            c = tuple(int(color1[i] * (1 - t) + color2[i] * t) for i in range(3))
            row_mask = mask[row] > 0
            out[row, row_mask] = c
        return out

    def apply_highlights(
        self, bgr_image: np.ndarray, highlight_color=(255, 255, 200), intensity=0.4
    ) -> np.ndarray:
        """Simulate hair highlights by brightening a random subset of hair pixels.

        Selects ~20% of hair pixels, dilates them into streak shapes, then
        alpha-blends the highlight color over those pixels.

        Args:
          bgr_image: BGR numpy array.
          highlight_color: BGR color tuple for the highlight tint.
          intensity: Blend strength (0 = no effect, 1 = full highlight color).
        Returns:
          BGR numpy array with simulated highlights applied.
        """
        mask = self.get_hair_mask(bgr_image)
        out = bgr_image.copy()
        hair_pixels = np.argwhere(mask > 0)
        if len(hair_pixels) == 0:
            return out
        rng = np.random.default_rng(0)
        n = max(1, len(hair_pixels) // 5)
        indices = rng.choice(len(hair_pixels), size=n, replace=False)
        sparse = np.zeros_like(mask)
        for idx in indices:
            r, c = hair_pixels[idx]
            sparse[r, c] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sparse = cv2.dilate(sparse, kernel, iterations=2)
        hl = np.array(highlight_color, dtype=np.uint8)
        blend_mask = (sparse > 0) & (mask > 0)
        out[blend_mask] = (out[blend_mask] * (1 - intensity) + hl * intensity).astype(
            np.uint8
        )
        return out

    def _get_mask(self, result, smooth=True):
        mask = result.category_mask.numpy_view()
        mask = (mask * 255).astype(np.uint8)
        return mask

    def detect(self, image: np.ndarray, smooth=True, mask_color=(255, 0, 255)):
        result = self.process(image)
        mask = self._get_mask(result, smooth)
        if smooth:
            _, mask = cv2.threshold(
                mask,
                1,
                255,
                cv2.THRESH_BINARY,
            )
        overlay = image.copy()
        overlay[mask > 0] = mask_color
        result_frame = cv2.addWeighted(
            overlay,
            0.4,
            image,
            0.6,
            0,
        )
        return result_frame
