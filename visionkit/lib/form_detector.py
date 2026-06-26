import cv2
import numpy as np

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class FormROIDetector:
    """
    Form Region of Interest (ROI) Detector that identifies various field types in forms, including text fields, checkboxes, radio buttons, date ranges, and table cells. It uses contour analysis for detection and can optionally perform OCR to extract text labels. The class also includes functionality to group ROIs into rows and link keys to their corresponding values.

    image = cv2.imread(main_form_path)

    detector = FormROIDetector(
        min_area=1000,
        enable_ocr=True,
        debug=True
    )

    result = detector.process(image)
    print(result["roi"])


    # Option 2 — pass regions directly
    vis = detector.visualize(image, regions=result)

    # Option 3 — show in a window (while developing)
    cv2.imshow("Form Fields", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    def __init__(
        self,
        min_area=500,
        enable_ocr=True,
        morph_kernel_size=(5, 5),
        row_tolerance=20,
        debug=False,
    ):
        self.min_area = min_area
        self.enable_ocr = enable_ocr and TESSERACT_AVAILABLE
        self.kernel_size = morph_kernel_size
        self.row_tolerance = row_tolerance
        self.debug = debug

    def _overlap_ratio(self, bbox1, bbox2):
        """
        Compute overlap ratio between two bounding boxes (used to filter duplicates).
        A simple IoU-like metric that considers the area of overlap relative to the smaller box.

        Args:
            bbox1: (x1, y1, x2, y2) for the first box
            bbox2: (x3, y3, x4, y4) for the second box
        Returns:
            A float representing the overlap ratio (0.0 to 1.0)
        """
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        overlap_x1 = max(x1, x3)
        overlap_y1 = max(y1, y3)
        overlap_x2 = min(x2, x4)
        overlap_y2 = min(y2, y4)
        if overlap_x2 < overlap_x1 or overlap_y2 < overlap_y1:
            return 0.0
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        return overlap_area / min(area1, area2) if min(area1, area2) > 0 else 0.0

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def _preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        edges = cv2.Canny(gray, 50, 150)
        return cv2.bitwise_or(thresh, edges)

    # ---------------------------
    # Table Cell Detection (new - extracts individual cells from grid lines)
    # ---------------------------
    def _detect_table_cells(self, image):
        """Detect table grid and return cell ROIs as 'table_cell' type."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        # Detect horizontal and vertical lines (tuned for cell extraction)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Get line positions
        h_contours, _ = cv2.findContours(
            horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        v_contours, _ = cv2.findContours(
            vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        horiz_lines = sorted(
            [cv2.boundingRect(c)[1] for c in h_contours if cv2.contourArea(c) > 50]
        )
        vert_lines = sorted(
            [cv2.boundingRect(c)[0] for c in v_contours if cv2.contourArea(c) > 50]
        )

        # Unique positions
        horiz_lines = sorted(set(horiz_lines))
        vert_lines = sorted(set(vert_lines))

        if len(horiz_lines) < 2 or len(vert_lines) < 2:
            return []

        cells = []
        for i in range(len(horiz_lines) - 1):
            y1 = horiz_lines[i]
            y2 = horiz_lines[i + 1]
            for j in range(len(vert_lines) - 1):
                x1 = vert_lines[j]
                x2 = vert_lines[j + 1]
                w = x2 - x1
                h = y2 - y1
                if w > 20 and h > 20:
                    cells.append(
                        {"bbox": (x1, y1, x2, y2), "type": "table_cell", "text": ""}
                    )
        return cells

    # ---------------------------
    # ROI Detection (enhanced with new field types)
    # ---------------------------
    def detect_rois(self, image):
        thresh = self._preprocess(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Additional shape analysis
            peri = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            is_rect = len(approx) == 4

            # Enhanced type classification
            if (
                is_rect
                and 0.75 <= aspect_ratio <= 1.35
                and max(w, h) < 200
                and min(w, h) > 10
            ):
                # Small square/near-square fields
                roi_type = "radio" if circularity > 0.75 else "checkbox"
            elif is_rect and aspect_ratio >= 2.5:
                # Wide input fields
                roi_type = (
                    "daterange" if aspect_ratio > 5.0 else "textbox"
                )  # Very wide → likely date range box
            else:
                roi_type = "text"  # Labels or irregular text regions

            label = ""
            if self.enable_ocr:
                # OCR on most fields (skip tiny checkboxes/radios - label is usually external)
                if roi_type not in ["checkbox", "radio"]:
                    label = self._ocr(image[y : y + h, x : x + w])
                else:
                    # Light OCR attempt on selection fields (rarely contains text)
                    label = self._ocr(image[y : y + h, x : x + w])

            rois.append(
                {
                    "bbox": (x, y, x + w, y + h),
                    "type": roi_type,
                    "text": label.strip(),
                    "circularity": circularity,
                }
            )

        # Add table cells (new feature)
        table_cells = self._detect_table_cells(image)
        for cell in table_cells:
            if self.enable_ocr:
                x1, y1, x2, y2 = cell["bbox"]
                cell["text"] = self._ocr(image[y1:y2, x1:x2])
            rois.append(cell)

        # Sort and remove near-duplicate ROIs (e.g. line contours vs table cells)
        rois = sorted(rois, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        filtered_rois = []
        for r in rois:
            if not any(
                self._overlap_ratio(r["bbox"], f["bbox"]) > 0.6 for f in filtered_rois
            ):
                filtered_rois.append(r)

        return filtered_rois

    # ---------------------------
    # OCR
    # ---------------------------
    def _ocr(self, roi):
        try:
            text = pytesseract.image_to_string(roi, config="--psm 6")
            return text.strip().split("\n")[0]
        except Exception:
            return ""

    # ---------------------------
    # Row Grouping
    # ---------------------------
    def group_rows(self, rois):
        rows = []
        current_row = []

        for roi in rois:
            if not current_row:
                current_row.append(roi)
                continue

            prev_y = current_row[-1]["bbox"][1]
            curr_y = roi["bbox"][1]

            if abs(curr_y - prev_y) < self.row_tolerance:
                current_row.append(roi)
            else:
                rows.append(sorted(current_row, key=lambda r: r["bbox"][0]))
                current_row = [roi]

        if current_row:
            rows.append(sorted(current_row, key=lambda r: r["bbox"][0]))

        return rows

    # ---------------------------
    # Key-Value Extraction (enhanced for all new field types)
    # ---------------------------
    def extract_key_values(self, rows):
        key_values = []
        field_types = {"textbox", "checkbox", "radio", "daterange"}

        for row in rows:
            texts = [r for r in row if r["type"] == "text"]
            fields = [r for r in row if r["type"] in field_types]

            for t in texts:
                tx1, ty1, tx2, ty2 = t["bbox"]

                # Find nearest field to the right
                candidates = [b for b in fields if b["bbox"][0] > tx2 + 5]

                if not candidates:
                    continue

                nearest = min(candidates, key=lambda b: b["bbox"][0] - tx2)

                key_values.append(
                    {
                        "key": t["text"],
                        "value_bbox": nearest["bbox"],
                        "value_type": nearest["type"],
                        "value_text": nearest.get("text", ""),
                        "checked": (
                            nearest.get("checked", False)
                            if nearest["type"] in ["checkbox", "radio"]
                            else None
                        ),
                    }
                )

        return key_values

    # ---------------------------
    # Selection State Detection (checkbox / radio)
    # ---------------------------
    def detect_selection_state(self, image, rois):
        """Detect checked state for checkboxes and radio buttons."""
        for roi in rois:
            if roi["type"] not in ["checkbox", "radio"]:
                continue

            x1, y1, x2, y2 = roi["bbox"]
            crop = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            filled_ratio = np.sum(thresh == 255) / thresh.size
            roi["checked"] = filled_ratio > 0.2

        return rois

    # ---------------------------
    # Table Mask (kept for backward compatibility + visualization)
    # ---------------------------
    def detect_table(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        table_mask = cv2.add(horizontal, vertical)
        return table_mask

    # ---------------------------
    # Visualization (updated for all new types)
    # ---------------------------
    def visualize(
        self,
        image,
        result,
        show_labels=True,
        show_rows=True,
        show_links=True,
        show_table=False,
    ):
        vis = image.copy()

        rois = result.get("rois", [])
        rows = result.get("rows", [])
        key_values = result.get("key_values", [])

        # ---------------------------
        # Draw ROIs with new type colors
        # ---------------------------
        for roi in rois:
            x1, y1, x2, y2 = roi["bbox"]
            roi_type = roi["type"]

            # Base color by type
            if roi_type == "text":
                color = (0, 255, 0)  # green
            elif roi_type in ["checkbox", "radio"]:
                color = (255, 0, 0)  # blue
            elif roi_type == "textbox":
                color = (0, 165, 255)  # orange
            elif roi_type == "daterange":
                color = (255, 165, 0)  # yellow-orange
            elif roi_type == "table_cell":
                color = (255, 0, 255)  # magenta
            else:
                color = (128, 128, 128)  # gray

            # Override for checked selection fields
            if "checked" in roi:
                color = (0, 0, 255) if roi["checked"] else (255, 0, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                label = roi.get("text", "")
                label_text = f"{roi_type}:{label}" if label else roi_type
                cv2.putText(
                    vis,
                    label_text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # ---------------------------
        # Draw Rows (grouping)
        # ---------------------------
        if show_rows:
            for _i, row in enumerate(rows):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                for roi in row:
                    x1, y1, x2, y2 = roi["bbox"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # ---------------------------
        # Draw Key-Value links
        # ---------------------------
        if show_links:
            for kv in key_values:
                key_text = kv["key"]
                value_bbox = kv["value_bbox"]

                key_roi = next((r for r in rois if r.get("text") == key_text), None)
                if key_roi is None:
                    continue

                kx1, ky1, kx2, ky2 = key_roi["bbox"]
                vx1, vy1, vx2, vy2 = value_bbox

                key_center = ((kx1 + kx2) // 2, (ky1 + ky2) // 2)
                val_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)

                cv2.line(vis, key_center, val_center, (0, 255, 255), 2)

        # ---------------------------
        # Draw Table Structure (optional)
        # ---------------------------
        if show_table:
            table_mask = self.detect_table(image)
            vis = cv2.addWeighted(
                vis, 0.8, cv2.cvtColor(table_mask, cv2.COLOR_GRAY2BGR), 0.5, 0
            )

        return vis

    # ---------------------------
    # Full Pipeline (returns original dict + new "roi" list in requested format)
    # ---------------------------
    def process(self, image):
        rois = self.detect_rois(image)
        rois = self.detect_selection_state(image, rois)

        rows = self.group_rows(rois)
        key_values = self.extract_key_values(rows)

        # NEW: ROI list in the exact format requested by the user
        roi = [
            [
                (r["bbox"][0], r["bbox"][1]),
                (r["bbox"][2], r["bbox"][3]),
                r["type"],
                r.get("text", ""),
            ]
            for r in rois
        ]

        return {
            "rois": rois,
            "rows": rows,
            "key_values": key_values,
            "roi": roi,  # ← Desired output format
            "table_mask": self.detect_table(image),  # kept for convenience
        }
