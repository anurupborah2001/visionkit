"""
FormROIDetector - Enhanced Form Field Detection Library
=======================================================
Detects: text fields, checkboxes, radio buttons, date-range boxes,
         tables, dropdowns, signature areas.

Output ROI format:
    roi = [
        [(x1, y1), (x2, y2), "field_type", "label"],
        ...
    ]

Field types
-----------
"text"       – single-line text input
"textarea"   – multi-line text area
"checkbox"   – square tick box
"radio"      – circular option button
"date"       – date or date-range field
"table"      – data table region
"dropdown"   – select / combo box
"signature"  – signature / initials box
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class ROIRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    field_type: str  # "text" | "textarea" | "checkbox" | "radio" |
    # "date" | "table" | "dropdown" | "signature"
    label: str = ""
    checked: bool | None = None  # checkbox / radio only
    confidence: float = 1.0

    # ------------------------------------------------------------------ #
    def to_tuple(self) -> list:
        """Return the canonical output format requested by the user."""
        return [(self.x1, self.y1), (self.x2, self.y2), self.field_type, self.label]

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        return self.width / max(self.height, 1)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


# ---------------------------------------------------------------------------
# Helper – date-like pattern matcher
# ---------------------------------------------------------------------------
_DATE_PATTERNS = re.compile(
    r"(date|dob|d\.o\.b|birth|expir|valid|from|to|period|dd[/\-_]mm|"
    r"mm[/\-_]yy|yyyy|day|month|year)",
    re.IGNORECASE,
)

_DROPDOWN_PATTERNS = re.compile(
    r"(select|choose|pick|▼|v\b|\bv\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


class FormROIDetector:
    """
    Detect form fields in document / form images and return ROIs in the
    standardised list-of-tuples format.

    Parameters
    ----------
    min_area        : minimum contour area to consider (pixels²)
    enable_ocr      : whether to use pytesseract for label extraction
    morph_kernel    : morphological kernel size used for contour cleanup
    row_tolerance   : pixel tolerance for grouping ROIs into the same row
    circle_dp       : HoughCircles dp parameter (radio-button detection)
    debug           : draw intermediate steps (returned in result dict)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        min_area: int = 400,
        enable_ocr: bool = True,
        morph_kernel: tuple[int, int] = (3, 3),
        row_tolerance: int = 18,
        circle_dp: float = 1.2,
        debug: bool = False,
    ):
        self.min_area = min_area
        self.enable_ocr = enable_ocr and TESSERACT_AVAILABLE
        self.morph_kernel = morph_kernel
        self.row_tolerance = row_tolerance
        self.circle_dp = circle_dp
        self.debug = debug

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def process(self, image: np.ndarray) -> dict:
        """
        Full detection pipeline.

        Returns
        -------
        {
            "roi"        : [[(x1,y1),(x2,y2), type, label], ...],  ← canonical
            "regions"    : [ROIRegion, ...],                         ← rich objects
            "rows"       : [[ROIRegion, ...], ...],
            "key_values" : [{"key": str, "value_bbox": tuple, "type": str}, ...],
            "debug_image": np.ndarray | None,
        }
        """
        regions: list[ROIRegion] = []

        # 1. Table detection (before general contour search)
        table_regions = self._detect_tables(image)
        table_masks = self._build_table_mask(image, table_regions)
        regions.extend(table_regions)

        # 2. Checkbox detection (Hough squares / contour aspect)
        cb_regions = self._detect_checkboxes(image, table_masks)
        regions.extend(cb_regions)

        # 3. Radio-button detection
        radio_regions = self._detect_radio_buttons(image, table_masks)
        regions.extend(radio_regions)

        # 4. General text / textarea / date / dropdown / signature fields
        general_regions = self._detect_general_fields(image, table_masks, regions)
        regions.extend(general_regions)

        # 5. De-duplicate / merge overlapping regions
        regions = self._deduplicate(regions)

        # 6. OCR labels
        if self.enable_ocr:
            regions = self._assign_labels(image, regions)

        # 7. Checkbox / radio fill state
        regions = self._detect_fill_state(image, regions)

        # 8. Row grouping & key-value pairs
        rows = self._group_rows(regions)
        key_values = self._extract_key_values(rows)

        # 9. Build canonical output
        roi_list = [r.to_tuple() for r in regions]

        debug_img = None
        if self.debug:
            debug_img = self.visualize(image, regions)

        return {
            "roi": roi_list,
            "regions": regions,
            "rows": rows,
            "key_values": key_values,
            "debug_image": debug_img,
        }

    # ------------------------------------------------------------------
    # Convenience wrapper – returns only the ROI list
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> list:
        """
        Shorthand that returns only the canonical ROI list.

        roi = detector.detect(img)
        # → [[(x1,y1),(x2,y2), type, label], ...]
        """
        return self.process(image)["roi"]

    # ==================================================================
    # DETECTION MODULES
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. Table detection
    # ------------------------------------------------------------------
    def _detect_tables(self, image: np.ndarray) -> list[ROIRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )

        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        table_mask = cv2.add(horizontal, vertical)

        # Dilate to merge nearby lines into table blocks
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(table_mask, dilate_k, iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area * 4:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Must have multiple lines to be a table
            h_lines = cv2.countNonZero(horizontal[y : y + h, x : x + w])
            v_lines = cv2.countNonZero(vertical[y : y + h, x : x + w])
            if h_lines > 0 and v_lines > 0:
                regions.append(ROIRegion(x, y, x + w, y + h, "table"))

        return regions

    def _build_table_mask(
        self, image: np.ndarray, table_regions: list[ROIRegion]
    ) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for r in table_regions:
            cv2.rectangle(mask, (r.x1, r.y1), (r.x2, r.y2), 255, -1)
        return mask

    # ------------------------------------------------------------------
    # 2. Checkbox detection
    # ------------------------------------------------------------------
    def _detect_checkboxes(
        self, image: np.ndarray, table_mask: np.ndarray
    ) -> list[ROIRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 8000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / max(h, 1)

            # Must be roughly square and small
            if not (0.6 <= ar <= 1.6 and 8 <= w <= 80 and 8 <= h <= 80):
                continue

            # Solidity check – filled squares have high solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            # Must NOT be inside a table region already handled
            cx, cy = x + w // 2, y + h // 2
            if table_mask[cy, cx] > 0:
                continue

            # Approx polygon – checkbox ≈ 4 vertices
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if not (3 <= len(approx) <= 8):
                continue

            regions.append(ROIRegion(x, y, x + w, y + h, "checkbox"))

        return regions

    # ------------------------------------------------------------------
    # 3. Radio button detection
    # ------------------------------------------------------------------
    def _detect_radio_buttons(
        self, image: np.ndarray, table_mask: np.ndarray
    ) -> list[ROIRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=15,
            param1=60,
            param2=25,
            minRadius=5,
            maxRadius=25,
        )

        regions = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for cx, cy, r in circles:
                if table_mask[cy, cx] > 0:
                    continue
                x1, y1 = int(cx - r), int(cy - r)
                x2, y2 = int(cx + r), int(cy + r)
                regions.append(ROIRegion(max(0, x1), max(0, y1), x2, y2, "radio"))

        return regions

    # ------------------------------------------------------------------
    # 4. General field detection (text / textarea / date / dropdown / sig)
    # ------------------------------------------------------------------
    def _detect_general_fields(
        self,
        image: np.ndarray,
        table_mask: np.ndarray,
        existing: list[ROIRegion],
    ) -> list[ROIRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        existing_bboxes = [(r.x1, r.y1, r.x2, r.y2) for r in existing]
        regions = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            x2, y2 = x + w, y + h

            # Skip if centre is inside a table
            if table_mask[cy, cx] > 0:
                continue

            # Skip if heavily overlapping an already-detected region
            if self._overlaps_any(x, y, x2, y2, existing_bboxes, thresh=0.5):
                continue

            ar = w / max(h, 1)

            # --- Classify field type by geometry ---
            field_type = self._classify_general(image, x, y, x2, y2, ar, w, h)

            regions.append(ROIRegion(x, y, x2, y2, field_type))

        return regions

    def _classify_general(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        ar: float,
        w: int,
        h: int,
    ) -> str:
        """Classify a rectangular region into a field type."""
        label_text = ""
        if self.enable_ocr:
            # Peek at OCR content inside the box
            crop = image[y1:y2, x1:x2]
            label_text = self._ocr_text(crop)

        # Dropdown: wide, short, with a dropdown arrow character
        if ar > 3 and h < 60 and _DROPDOWN_PATTERNS.search(label_text):
            return "dropdown"

        # Date field: label contains date keywords or has slashes drawn inside
        if _DATE_PATTERNS.search(label_text):
            return "date"
        # Date: look for separator lines inside (dd/mm/yyyy boxes)
        if ar > 1.5 and h < 70 and self._has_internal_dividers(image, x1, y1, x2, y2):
            return "date"

        # Signature / large blank area: very wide, taller than a text line
        if ar > 4 and h > 60:
            return "signature"

        # Textarea: roughly square or portrait, large area
        if 0.3 <= ar <= 2.5 and h > 60 and w > 80:
            return "textarea"

        # Single-line text input: wide and short
        if ar >= 2.5 and h < 70:
            return "text"

        # Fallback
        return "text"

    def _has_internal_dividers(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        """Check whether a box contains internal vertical dividers (date parts)."""
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, crop.shape[0] // 2))
        )
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        return cv2.countNonZero(vertical) > 10

    # ==================================================================
    # OCR
    # ==================================================================

    def _ocr_text(self, crop: np.ndarray) -> str:
        if not self.enable_ocr or crop.size == 0:
            return ""
        try:
            text = pytesseract.image_to_string(crop, config="--psm 6 --oem 3")
            return text.strip()
        except Exception:
            return ""

    # ==================================================================
    # LABEL ASSIGNMENT
    # ==================================================================

    def _assign_labels(
        self, image: np.ndarray, regions: list[ROIRegion]
    ) -> list[ROIRegion]:
        """
        For each region, look for OCR text immediately to the LEFT or ABOVE
        the bounding box and assign it as the label.
        """
        h_img, w_img = image.shape[:2]

        for region in regions:
            if region.label:
                continue

            # Search window: same height as the field, to its left
            search_x1 = max(0, region.x1 - 300)
            search_x2 = region.x1
            search_y1 = max(0, region.y1 - 5)
            search_y2 = min(h_img, region.y2 + 5)

            left_crop = image[search_y1:search_y2, search_x1:search_x2]
            label = self._ocr_text(left_crop)

            if not label:
                # Try above
                search_y1b = max(0, region.y1 - 40)
                search_y2b = region.y1
                above_crop = image[search_y1b:search_y2b, region.x1 : region.x2]
                label = self._ocr_text(above_crop)

            region.label = label.replace("\n", " ").strip()[:80]

        return regions

    # ==================================================================
    # FILL STATE (checkbox / radio)
    # ==================================================================

    def _detect_fill_state(
        self, image: np.ndarray, regions: list[ROIRegion]
    ) -> list[ROIRegion]:
        for region in regions:
            if region.field_type not in ("checkbox", "radio"):
                continue
            x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            filled_ratio = np.sum(thresh == 255) / thresh.size
            region.checked = filled_ratio > 0.18

        return regions

    # ==================================================================
    # DEDUPLICATION
    # ==================================================================

    def _deduplicate(self, regions: list[ROIRegion]) -> list[ROIRegion]:
        """Remove regions that are nearly identical or heavily overlapping."""
        if not regions:
            return regions

        # Sort by area descending (keep larger / more specific detections)
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        kept: list[ROIRegion] = []

        for candidate in regions:
            dominated = False
            for existing in kept:
                iou = self._iou(candidate, existing)
                if iou > 0.45:
                    # Prefer more specific type
                    dominated = True
                    break
            if not dominated:
                kept.append(candidate)

        return kept

    @staticmethod
    def _iou(a: ROIRegion, b: ROIRegion) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = a.area + b.area - inter
        return inter / max(union, 1)

    @staticmethod
    def _overlaps_any(x1, y1, x2, y2, bboxes, thresh=0.5) -> bool:
        area = max(1, (x2 - x1) * (y2 - y1))
        for bx1, by1, bx2, by2 in bboxes:
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / area >= thresh:
                return True
        return False

    # ==================================================================
    # ROW GROUPING
    # ==================================================================

    def _group_rows(self, regions: list[ROIRegion]) -> list[list[ROIRegion]]:
        sorted_regions = sorted(regions, key=lambda r: (r.y1, r.x1))
        rows: list[list[ROIRegion]] = []
        current_row: list[ROIRegion] = []

        for region in sorted_regions:
            if not current_row:
                current_row.append(region)
                continue
            prev_y = current_row[-1].y1
            if abs(region.y1 - prev_y) < self.row_tolerance:
                current_row.append(region)
            else:
                rows.append(sorted(current_row, key=lambda r: r.x1))
                current_row = [region]

        if current_row:
            rows.append(sorted(current_row, key=lambda r: r.x1))

        return rows

    # ==================================================================
    # KEY-VALUE EXTRACTION
    # ==================================================================

    def _extract_key_values(self, rows: list[list[ROIRegion]]) -> list[dict]:
        key_values = []
        for row in rows:
            text_fields = [
                r
                for r in row
                if r.field_type in ("text", "textarea", "date", "dropdown", "signature")
            ]
            input_fields = [r for r in row if r.field_type in ("checkbox", "radio")]

            for tf in text_fields:
                # Nearest input to the right
                candidates = [b for b in input_fields if b.x1 > tf.x2]
                if candidates:
                    nearest = min(candidates, key=lambda b: b.x1 - tf.x2)
                    key_values.append(
                        {
                            "key": tf.label or "?",
                            "value_bbox": nearest.bbox,
                            "type": nearest.field_type,
                        }
                    )

        return key_values

    # ==================================================================
    # VISUALIZATION
    # ==================================================================

    # Color palette per field type
    _TYPE_COLORS = {
        "text": (34, 197, 94),  # green
        "textarea": (16, 185, 129),  # teal
        "checkbox": (59, 130, 246),  # blue  (unchecked)
        "radio": (168, 85, 247),  # purple
        "date": (249, 115, 22),  # orange
        "table": (234, 179, 8),  # yellow
        "dropdown": (236, 72, 153),  # pink
        "signature": (239, 68, 68),  # red
    }
    _CHECKED_COLOR = (22, 163, 74)  # dark green when checked
    _UNCHECKED_COLOR = (59, 130, 246)  # blue when unchecked

    def visualize(
        self,
        image: np.ndarray,
        regions: list[ROIRegion] | None = None,
        result: dict | None = None,
        show_labels: bool = True,
        show_type_legend: bool = True,
    ) -> np.ndarray:
        """
        Draw all detected regions on a copy of *image* and return it.

        Pass either *regions* directly or the full *result* dict from process().
        """
        if regions is None and result is not None:
            regions = result.get("regions", [])
        if regions is None:
            regions = []

        vis = image.copy()

        for region in regions:
            color = self._TYPE_COLORS.get(region.field_type, (200, 200, 200))

            # Override checkbox / radio color by state
            if (
                region.field_type in ("checkbox", "radio")
                and region.checked is not None
            ):
                color = self._CHECKED_COLOR if region.checked else self._UNCHECKED_COLOR

            # Draw bounding rect
            cv2.rectangle(vis, (region.x1, region.y1), (region.x2, region.y2), color, 2)

            if show_labels:
                tag = region.field_type.upper()
                if region.label:
                    tag += f": {region.label[:25]}"
                if region.checked is not None:
                    tag += " ✓" if region.checked else " ✗"

                # Background pill for readability
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                ty = max(region.y1 - 4, th + 4)
                cv2.rectangle(
                    vis,
                    (region.x1, ty - th - 4),
                    (region.x1 + tw + 6, ty + 2),
                    color,
                    -1,
                )
                cv2.putText(
                    vis,
                    tag,
                    (region.x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Legend
        if show_type_legend:
            lx, ly = 10, 10
            for ft, color in self._TYPE_COLORS.items():
                cv2.rectangle(vis, (lx, ly), (lx + 16, ly + 16), color, -1)
                cv2.putText(
                    vis,
                    ft,
                    (lx + 22, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                ly += 22

        return vis

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def crop_roi(self, image: np.ndarray, region: "ROIRegion") -> np.ndarray:
        """Crop a single ROI region out of the source image.
        Useful for feeding individual fields into an OCR or classifier.

        Args:
          image: BGR numpy array (the original form image).
          region: ROIRegion object from process()["regions"].
        Returns:
          BGR numpy array crop, or empty array if out of bounds.
        """
        h, w = image.shape[:2]
        x1 = max(0, region.x1)
        y1 = max(0, region.y1)
        x2 = min(w, region.x2)
        y2 = min(h, region.y2)
        return image[y1:y2, x1:x2].copy()

    def extract_field_values(self, image: np.ndarray, regions) -> dict:
        """OCR every detected field and return {label: text} mapping.
        Skips checkbox/radio (use region.checked instead) and empty labels.

        Args:
          image: BGR numpy array.
          regions: List of ROIRegion from process()["regions"].
        Returns:
          dict: {field_label: ocr_text}
        """
        if not TESSERACT_AVAILABLE:
            return {}
        values = {}
        for region in regions:
            if region.field_type in ("checkbox", "radio"):
                key = region.label or f"{region.field_type}_{region.x1}_{region.y1}"
                values[key] = region.checked
                continue
            crop = self.crop_roi(image, region)
            if crop.size == 0:
                continue
            text = self._ocr_text(crop)
            key = region.label or f"{region.field_type}_{region.x1}_{region.y1}"
            values[key] = text
        return values

    def filter_by_type(self, regions, field_type: str):
        """Return only regions matching the given field_type.

        Args:
          regions: List of ROIRegion from process()["regions"].
          field_type: One of 'text', 'textarea', 'checkbox', 'radio', 'date',
                      'table', 'dropdown', 'signature'.
        Returns:
          List[ROIRegion]
        """
        return [r for r in regions if r.field_type == field_type]

    def get_checked_fields(self, regions):
        """Return only checkbox and radio regions that are checked.

        Args:
          regions: List of ROIRegion from process()["regions"].
        Returns:
          List[ROIRegion]
        """
        return [
            r
            for r in regions
            if r.field_type in ("checkbox", "radio") and r.checked is True
        ]

    def export_to_json(self, result: dict, path: str = "form_rois.json"):
        """Save the canonical ROI list from process() to a JSON file.

        Args:
          result: Dict returned by process().
          path: Output file path.
        Returns:
          str: Absolute path of the written file.
        """
        import json
        import os

        roi_serialisable = []
        for entry in result.get("roi", []):
            (x1, y1), (x2, y2), ftype, label = entry
            roi_serialisable.append(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "type": ftype, "label": label}
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(roi_serialisable, f, indent=2)
        return os.path.abspath(path)

    def export_to_csv(self, result: dict, path: str = "form_rois.csv"):
        """Save the ROI list from process() to a CSV file.
        Columns: x1, y1, x2, y2, type, label, checked.

        Args:
          result: Dict returned by process().
          path: Output file path.
        Returns:
          str: Absolute path of the written file.
        """
        import csv
        import os

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["x1", "y1", "x2", "y2", "type", "label", "checked"]
            )
            writer.writeheader()
            for region in result.get("regions", []):
                writer.writerow(
                    {
                        "x1": region.x1,
                        "y1": region.y1,
                        "x2": region.x2,
                        "y2": region.y2,
                        "type": region.field_type,
                        "label": region.label,
                        "checked": region.checked,
                    }
                )
        return os.path.abspath(path)

    def get_field_count(self, regions) -> dict:
        """Return count of each field type detected.

        Args:
          regions: List of ROIRegion from process()["regions"].
        Returns:
          dict: {'text': 4, 'checkbox': 6, ...}
        """
        counts: dict = {}
        for r in regions:
            counts[r.field_type] = counts.get(r.field_type, 0) + 1
        return counts

    def get_empty_fields(self, regions):
        """Return checkbox/radio regions that are not checked.

        Args:
            regions: List of ROIRegion objects.
        Returns:
            List of ROIRegion where field_type is 'checkbox' or 'radio'
            and checked is not True.
        """
        return [
            r
            for r in regions
            if r.field_type in ("checkbox", "radio") and not r.checked
        ]

    def validate_required_fields(self, regions, required_labels) -> dict:
        """Check which required labels have been filled (checked).

        Args:
            regions: List of ROIRegion objects.
            required_labels: List of label strings that must be checked.
        Returns:
            dict with keys 'missing' and 'filled', each a list of labels.
        """
        checked_labels = {
            r.label.lower()
            for r in regions
            if r.field_type in ("checkbox", "radio") and r.checked
        }
        missing = [lbl for lbl in required_labels if lbl.lower() not in checked_labels]
        filled = [lbl for lbl in required_labels if lbl.lower() in checked_labels]
        return {"missing": missing, "filled": filled}

    def get_field_by_label(self, regions, label):
        """Find the first region whose label matches (case-insensitive).

        Args:
            regions: List of ROIRegion objects.
            label: Label string to search for.
        Returns:
            ROIRegion if found, None otherwise.
        """
        label_lower = label.lower()
        for r in regions:
            if r.label.lower() == label_lower:
                return r
        return None

    def get_form_completion_score(self, regions) -> float:
        """Return fraction of checkboxes/radios that are checked.

        Args:
            regions: List of ROIRegion objects.
        Returns:
            Float in [0.0, 1.0]. Returns 0.0 if no checkable fields exist.
        """
        checkable = [r for r in regions if r.field_type in ("checkbox", "radio")]
        if not checkable:
            return 0.0
        filled = sum(1 for r in checkable if r.checked)
        return filled / len(checkable)

    def highlight_empty_fields(self, image, regions, color=(0, 0, 255), thickness=2):
        """Draw rectangles around empty (unchecked) checkbox/radio fields.

        Args:
            image: BGR numpy array.
            regions: List of ROIRegion objects.
            color: BGR rectangle color. Defaults to red (0, 0, 255).
            thickness: Rectangle border thickness in pixels.
        Returns:
            Annotated BGR numpy array (copy of input).
        """
        out = image.copy()
        for r in self.get_empty_fields(regions):
            cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, thickness)
        return out

    def extract_all_text(self, image, regions) -> dict:
        """OCR each region and return a mapping of label → text.

        Args:
            image: BGR numpy array.
            regions: List of ROIRegion objects.
        Returns:
            dict mapping each region's label to its OCR text string.
        """
        result = {}
        for r in regions:
            crop = image[r.y1 : r.y2, r.x1 : r.x2]
            result[r.label] = self._ocr_text(crop)
        return result


# Usages:

# import cv2
# from form_roi_detector import FormROIDetector

# # Load your form image
# image = cv2.imread("my_form.png")

# # Create detector (OCR optional — needs pytesseract)
# detector = FormROIDetector(enable_ocr=True)

# # detect() → canonical ROI list only
# roi = detector.detect(image)

# # Each entry: [(x1,y1), (x2,y2), field_type, label]
# for entry in roi:
#     (x1, y1), (x2, y2), ftype, label = entry
#     print(f"[{ftype}] '{label}' → ({x1},{y1})→({x2},{y2})")

# result = detector.process(image)

# roi        = result["roi"]         # canonical list  ← same as detect()
# regions    = result["regions"]     # list[ROIRegion]  ← rich objects
# rows       = result["rows"]        # grouped by Y position
# key_values = result["key_values"]  # [{key, value_bbox, type}, ...]


# Advanced Usage:
# image = cv2.imread("my_form.png")
# detector = FormROIDetector(
#     min_area     = 400,    # px² — ignore tiny noise contours
#     enable_ocr   = True,   # False if pytesseract not installed
#     morph_kernel = (3, 3),  # larger → merges nearby strokes
#     row_tolerance= 18,    # px — Y-delta for same-row grouping
#     circle_dp    = 1.2,   # HoughCircles dp (radio detection)
#     debug        = False,  # True → result["debug_image"] set
# )
# result = detector.detect(image)

# regions = result["regions"]

# for r in regions:
#     print(r.field_type)  # "text"|"checkbox"|"radio"|"date"…
#     print(r.label)       # OCR text to the left / above
#     print(r.checked)     # True/False/None (checkbox+radio only)
#     print(r.bbox)        # (x1, y1, x2, y2)
#     print(r.to_tuple())  # canonical [(x1,y1),(x2,y2),type,label]
# kv = result["key_values"]
# # Links a text label field to the nearest checkbox/radio
# # [{"key": "Allergic", "value_bbox": (740,980,1320,1078), "type": "checkbox"}, …]

# for pair in kv:
#     print(f"{pair['key']} → {pair['type']} @ {pair['value_bbox']}")

# detector = FormROIDetector(debug=True)
# result = detector.process(image)

# debug_img = result["debug_image"]   # annotated np.ndarray
# cv2.imwrite("debug.png", debug_img)


# ROI Output Format:
# roi = [
#     [(90,  980),  (650,  1120), "text",      "Name"       ],
#     [(740, 980),  (1320, 1078), "checkbox",  "Allergic"   ],
#     [(90,  1140), (650,  1200), "date",      "Date of Birth"],
#     [(740, 1140), (900,  1200), "radio",     "Male"       ],
#     [(920, 1140), (1080, 1200), "radio",     "Female"     ],
#     [(90,  1220), (1320, 1460), "textarea",  "Comments"   ],
#     [(90,  1480), (1320, 1760), "table",     ""           ],
#     [(90,  1780), (600,  1840), "dropdown",  "Country"    ],
#     [(90,  1860), (500,  1940), "signature", "Signature"  ],
# ]


# Visualization
# Option 1 — via process() result
# image = cv2.imread("my_form.png")
# detector = FormROIDetector(enable_ocr=True)
# result = detector.process(image)
# vis = detector.visualize(image, result=result)
# cv2.imwrite("annotated.png", vis)

# # Option 2 — pass regions directly
# vis = detector.visualize(image, regions=result["regions"])

# # Option 3 — show in a window (while developing)
# cv2.imshow("Form Fields", vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Option 4 — Jupyter / Colab inline display
# from IPython.display import display
# import PIL.Image, io, numpy as np

# rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
# display(PIL.Image.fromarray(rgb))
