import json

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None  # type: ignore


class ImageDetector:
    def __init__(self, image, pre_process=False):
        self.image = image
        if pre_process:
            self.image = self._preprocess(image)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for better OCR results.

        Args:
            image (np.ndarray): The input image to preprocess.

        Returns:
            np.ndarray: The preprocessed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        gray = cv2.equalizeHist(gray)

        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def fallback_ssim(self, image1, image2, form_name, draw_frame=False):
        image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)

        diff = (diff * 255).astype("uint8")

        if draw_frame:
            cv2.imshow(f"{form_name} - SSIM Diff (score={score:.3f})", diff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": 0,
            "homography": None,
            "aligned_image": image2_resized,
            "ssim_score": score,
        }

    def compare_matches_knn_matcher(
        self,
        image2,
        form_name,
        no_of_feature=500,
        matched_amount=50,
        percentage_of_matches=20,
        draw_matches=False,
        draw_aligned=False,
    ):
        # Detect keypoints
        image_detector = ImageDetector(image2)

        keypoints1, descriptors1, _ = self.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )
        keypoints2, descriptors2, _ = image_detector.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )

        if descriptors1 is None or descriptors2 is None:
            print("Feature detection failed → using SSIM fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Safety check
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Descriptors could not be computed")

        # Use KNN matcher instead of crossCheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print("Not enough matches → using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Sort matches
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Take top percentage
        keep_n = int(len(good_matches) * (percentage_of_matches / 100))
        good_matches = good_matches[: max(keep_n, 4)]  # ensure at least 4

        # Draw matches
        matchedImage = cv2.drawMatches(
            self.image,
            keypoints1,
            image2,
            keypoints2,
            good_matches[:matched_amount],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Compute homography
        sourcePoints = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        destinationPoints = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(destinationPoints, sourcePoints, cv2.RANSAC, 5.0)

        if M is None:
            print("Homography could not be computed so it will be using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        h, w = self.image.shape[:2]
        imageTransformed = cv2.warpPerspective(image2, M, (w, h))

        imageTransformed_small = cv2.resize(imageTransformed, (w // 3, h // 3))
        matchedImage_small = cv2.resize(matchedImage, (w // 3, h // 3))

        if draw_matches:
            cv2.imshow(f"{form_name} - Matches (Inliers)", matchedImage_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if draw_aligned:
            cv2.imshow(f"{form_name} - Aligned", imageTransformed_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": len(good_matches),
            "homography": M,
            "matched_image": matchedImage,
            "aligned_image": imageTransformed,
        }

    def compare_matches_bf_matcher(
        self,
        image2,
        form_name,
        no_of_feature=500,
        matched_amount=50,
        percentage_of_matches=20,
        draw_matches=False,
        draw_aligned=False,
    ):
        # Detect keypoints
        image_detector = ImageDetector(image2)

        keypoints1, descriptors1, _ = self.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )
        keypoints2, descriptors2, _ = image_detector.detect_keypoints(
            features=no_of_feature, draw_keypoints=False
        )

        if descriptors1 is None or descriptors2 is None:
            print("Feature detection failed → using SSIM fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        # Safety check
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Descriptors could not be computed")

        # Use KNN matcher instead of crossCheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.match(descriptors1, descriptors2)

        # Sort matches
        good_matches = sorted(matches, key=lambda x: x.distance)

        # Take top percentage
        keep_n = int(len(good_matches) * (percentage_of_matches / 100))
        good_matches = good_matches[: max(keep_n, 4)]  # ensure at least 4

        # Draw matches
        matchedImage = cv2.drawMatches(
            self.image,
            keypoints1,
            image2,
            keypoints2,
            good_matches[:matched_amount],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Compute homography
        sourcePoints = np.float32(
            [keypoints1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        destinationPoints = np.float32(
            [keypoints2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(destinationPoints, sourcePoints, cv2.RANSAC, 5.0)

        if M is None:
            print("Homography could not be computed so it will be using fallback")
            return self.fallback_ssim(self.image, image2, form_name)

        h, w = self.image.shape[:2]
        imageTransformed = cv2.warpPerspective(image2, M, (w, h))

        imageTransformed_small = cv2.resize(imageTransformed, (w // 3, h // 3))
        matchedImage_small = cv2.resize(matchedImage, (w // 3, h // 3))

        # it will match the form and the template and show the matched keypoints and the aligned image. The homography matrix can be used to further analyze the geometric transformation between the two images, such as calculating the angle of rotation or the scale difference.
        if draw_matches:
            cv2.imshow(f"{form_name} - Matches (Inliers)", matchedImage_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # it will match the form and the template and show the matched keypoints and the aligned image. The homography matrix can be used to further analyze the geometric transformation between the two images, such as calculating the angle of rotation or the scale difference.
        if draw_aligned:
            cv2.imshow(f"{form_name} - Aligned", imageTransformed_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "matches": len(good_matches),
            "homography": M,
            "matched_image": matchedImage,
            "aligned_image": imageTransformed,
        }

    def compute_tolerance_percentile(self, pixels):
        """
        Adaptively compute dynamic tolerance using local pixel distribution.

        Args:
          pixels: Array of pixel values in HSV color space for a specific region (e.g., highlighted area).

        Returns:
          Tuple of (h_tol, s_tol, v_tol) representing the computed tolerances for hue, saturation, and value channels based on the 10th and 90th percentiles of
        """
        h_vals = pixels[:, 0]
        s_vals = pixels[:, 1]
        v_vals = pixels[:, 2]

        h_tol = int(np.percentile(h_vals, 90) - np.percentile(h_vals, 10))
        s_tol = int(np.percentile(s_vals, 90) - np.percentile(s_vals, 10))
        v_tol = int(np.percentile(v_vals, 90) - np.percentile(v_vals, 10))

        return h_tol, s_tol, v_tol

    def detect_highlighted_text(
        self,
        hsv_colors=None,  # seed HSV values
        h_tol=8,
        s_tol=80,
        v_tol=80,
        show_mask=False,
        show_combined_mask=False,
        show_image_with_mask=False,
    ):
        """
        Detect highlighted text by creating HSV masks around specified colors.
        Returns combined mask and individual masks for each color.

        HSV (Hue, Saturation, Value) image processing is a color representation model, often preferred over RGB in computer vision
        for color-based segmentation and detection. It separates color information (hue) from lighting/brightness (value), allowing
        robust object tracking under varying illumination. Common uses include object tracking, color-based filtering, and thresholding
        in OpenCV.

        Args:
          hsv_colors: List of seed HSV tuples to detect (e.g., yellow, green)
          h_tol, s_tol, v_tol: Tolerances for hue, saturation, and value to create color ranges
          show: Whether to display intermediate masks and results using OpenCV windows
          show_mask: Show individual color masks
          show_combined_mask: Show combined mask of all detected colors
          show_image_with_mask: Show the original image with detected areas masked

        Returns:
          image_with_mask: Original image with detected areas masked
          combined_mask: Binary mask combining all detected colors
          masks: List of individual masks for each specified color

        Usage:
          image = cv2.imread("doc.jpg")

          # Common highlighter HSV seeds (you can refine using click sampling)
          highlight_colors = [
              (30, 200, 250),   # yellow
              (60, 200, 250),   # green
              (150, 200, 250),  # pink
              (15, 200, 250),   # orange
          ]

          mask, masks = detect_highlighted_text(image, highlight_colors)
        """
        if hsv_colors is None:
            hsv_colors = [(27, 167, 251)]
        img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        masks = []

        for i, (h, s, v) in enumerate(hsv_colors):
            # h_tol2, s_tol2, v_tol2 = compute_tolerence = self.compute_dynamic_tolerance2(hsv, h, s, v)
            # print(f"Computed tolerances → H: {compute_tolerence[0]}, S: {compute_tolerence[1]}, V: {compute_tolerence[2]}")
            lower = np.array([max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol)])

            upper = np.array(
                [min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)]
            )

            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)

            # Combine all masks
            combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Remove Noise
            # kernel = np.ones((3,3), np.uint8)
            # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            if show_mask:
                cv2.imshow(f"Mask {i}", mask)

        img_with_mask = cv2.bitwise_and(self.image, self.image, mask=combined_mask)
        if show_combined_mask:
            cv2.imshow("Combined Mask", combined_mask)
        if show_image_with_mask:
            cv2.imshow("Image with Mask", img_with_mask)

        if show_mask or show_combined_mask or show_image_with_mask:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_with_mask, combined_mask, masks

    def get_dominant_hsv_colors(self, k=10):
        img_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        # 🔥 REMOVE WHITE BEFORE CLUSTERING
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = (s > 25) & (v > 60)  # lower threshold → keeps light blue

        pixels = hsv[mask]

        if len(pixels) == 0:
            return []

        pixels = pixels.reshape(-1, 3).astype(np.float32)

        k = min(k, len(pixels))

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        counts = np.bincount(labels.flatten())

        highlight_colors = []

        for i, (h, s, v) in enumerate(centers):
            h, s, v = int(h), int(s), int(v)

            # remove tiny clusters
            if counts[i] < 30:
                continue

            # remove dark
            if v < 60:
                continue

            # remove near-white
            if v > 200 and s < 40:
                continue

            highlight_colors.append((h, s, v))

        # 🔥 softer deduplication
        filtered = []
        for c in highlight_colors:
            if not any(abs(c[0] - fc[0]) < 4 for fc in filtered):
                filtered.append(c)

        predefined = [
            (95, 120, 255),  # light blue
            (30, 200, 250),  # yellow
            (60, 200, 250),  # green
            (100, 200, 250),  # blue
            (150, 200, 250),  # pink
            (20, 200, 250),  # orange
        ]
        return filtered + predefined

    # def get_dominant_hsv_colors(self, k=4):
    #   """
    #   auto-detect highlight colors in the image by clustering pixel colors in HSV space using K-means.
    #   Get dominant HSV colors from the image using K-means clustering.

    #   Args:
    #       k: Number of dominant colors to detect (default is 4)

    #   Returns:
    #       List of dominant HSV color tuples (h, s, v) detected in the image.
    #   Usage:

    #   """
    #   mg_blur = cv2.GaussianBlur(self.image, (5,5), 0)
    #   hsv = cv2.cvtColor(mg_blur, cv2.COLOR_BGR2HSV)
    #   pixels = hsv.reshape(-1, 3).astype(np.float32)

    #   _, labels, centers = cv2.kmeans(
    #       pixels, k, None,
    #       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
    #       10,
    #       cv2.KMEANS_RANDOM_CENTERS
    #   )
    #   # Filter likely highlight colors
    #   highlight_colors = []
    #   for (h, s, v) in centers:
    #       if s > 80 and v > 150:   # high saturation + brightness
    #           highlight_colors.append((int(h), int(s), int(v)))

    #   return highlight_colors

    def detect_single_highlighted_text(self, image, hsv_colors=None):
        """Detect highlighted text based on a single HSV color.
        Args:
            image: Input image in BGR format (as read by OpenCV)
            hsv_colors: List of HSV values to detect (default is a single yellow color)
        Returns:
            image_with_mask: Image with detected highlighted areas masked
            mask: Binary mask of detected highlighted areas
        """
        if hsv_colors is None:
            hsv_colors = [27, 167, 251]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([hsv_colors[0], hsv_colors[1], hsv_colors[2]])
        upper = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        img_with_mask = cv2.bitwise_and(image, image, mask=mask)
        return img_with_mask, mask

    def refine_mask(self, mask, merge=True):
        """
        Refines a binary mask by applying morphological operations to remove noise and merge words in the same line.

        Args:
            mask: The input binary mask to be refined.
            merge: If True, applies dilation to merge words in the same line (default is True).
            Use case	merge
            Detect each highlight separately	: False
            Group text into lines/regions	: True

        Returns:
            The refined binary mask.
        """

        kernel_small = np.ones((3, 3), np.uint8)
        # mask = cv2.erode(mask, kernel_small, iterations=1)
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        if merge:
            # ⚠️ This merges multiple highlights into ONE contour
            kernel_line = np.ones((15, 5), np.uint8)
            mask = cv2.dilate(mask, kernel_line, iterations=1)

        return mask

    def get_cannty_edges(self, low_threshold=50, high_threshold=150):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blur, low_threshold, high_threshold)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def find_contours(
        self,
        mask,
        min_area=200,
        get_canny_edges=False,
        canny_threshold: list[int] | None = None,
        debug=False,
        dilate_merge=True,
        filter_shapes: (
            list | None
        ) = None,  # to detect for triangle or rectangle/square shapes
        sort_contours_smallest_to_largest=False,
        sort_countours_largest_to_smallest=False,
        sort_bbox_smaller_to_largest=False,
        sort_bbox_top_to_bottom=False,
        sort_bbox_left_to_right=False,
        sort_bbox_area_largest_to_smallest=False,
        sort_bbox_grid_wise=False,
        bbox_grid_tolerence=10,
        retrieval_type=cv2.RETR_EXTERNAL,
        approximation_method=cv2.CHAIN_APPROX_SIMPLE,
        draw_contours=False,
        contour_box_color=(0, 255, 0),
        contour_box_thickness=2,
        contour_text_color=(255, 255, 0),
        contour_text_thickness=2,
    ):
        """
        Find contours in a binary mask and filter them based on area and other criteria.
        Contours and bounding boxes are both used for object localization in computer vision, with contours providing precise,
        detailed outlines of shapes, while bounding boxes offer simplified rectangular boxes (min/max coordinates) used primarily
        for detection, tracking, and fast computation. Contours are better for shape analysis, whereas bounding boxes are ideal for
        spatial localization

        Args:
            mask: Binary image (mask) where contours are to be found. This is the image on wich contour detection will be performed, typically a binary mask resulting from color segmentation or thresholding.
            min_area: Minimum area threshold to filter contours (default is 200).
            get_canny_edges: If True, applies Canny edge detection to the mask before finding contours (default is False).
            canny_threshold: List of two integers representing the lower and upper thresholds for Canny edge detection (default is [100, 100]).
            debug: If True, prints debug information about contours found and filtered.
            dilate_merge: If True, applies dilation to merge nearby contours (default is True).
            sort_contours_smallest_to_largest: If True, sorts the contours by area from smallest to largest (default is False).
            filter_shapes: List of vertex counts to filter contours by shape (e.g., [3, 4] for triangles and rectangles). If empty or None, no shape filtering is applied (default is [3, 4]).
            sort_bbox_smaller_to_largest: If True, sorts the bounding boxes of contours by area from smallest to largest (default is False).
            sort_countours_largest_to_smallest: If True, sorts the contours by area from largest to smallest (default is False).
            sort_bbox_top_to_bottom: If True, sorts the bounding boxes of contours from top to bottom (default is False).
            sort_bbox_left_to_right: If True, sorts the bounding boxes of contours from left to right (default is False).
            sort_bbox_area_largest_to_smallest: If True, sorts the bounding boxes of contours by area from largest to smallest (default is False).
            sort_bbox_grid_wise: If True, sorts the bounding boxes in a grid-wise manner (first by rows, then by columns) with a specified tolerence (default is False).
            bbox_grid_tolerence: Tolerence in pixels for grouping bounding boxes into the same row when sort_bbox_grid_wise is True (default is 10).
            retrieval_type: Contour retrieval mode (default is cv2.RETR_EXTERNAL).
            approximation_method: Contour approximation method (default is cv2.CHAIN_APPROX_SIMPLE).
            draw_contours: If True, draws the filtered contours on a copy of the original image for visualization (default is False).

        Returns:
            List of dictionaries containing contour information (contour, area, bounding_box, approx_vertices, center) for each filtered contour.
        """

        if canny_threshold is None:
            canny_threshold = [100, 100]
        if filter_shapes is None:
            filter_shapes = []
        if get_canny_edges:
            cleaned = self.get_cannty_edges(
                low_threshold=canny_threshold[0], high_threshold=canny_threshold[1]
            )
        else:
            # 1. Clean noise (very important)
            cleaned = self.refine_mask(mask, merge=dilate_merge)

        # kernel = np.ones((3, 3), np.uint8)

        # # Remove small noise
        # cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # # Fill gaps inside highlights
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. Find contours
        contours, _ = cv2.findContours(
            cleaned,
            retrieval_type,
            approximation_method,  # only outer regions
        )

        if debug:
            print(f"Found {len(contours)} contours before filtering")

        # filtered_contours = []
        # boxes = []
        contours_results = []
        countour_img = self.image.copy()
        # 3. Filter contours
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            peri = cv2.arcLength(
                cnt, True
            )  # Computes the perimeter of the contour and True → contour is closed
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print("Approx vertices:", len(approx))

            x, y, w, h = cv2.boundingRect(cnt)
            print(f"Contour area: {area}, Bounding box: (x={x}, y={y}, w={w}, h={h})")
            if w < 20 or h < 10:
                continue

            """
          cv2.approxPolyDP(cnt, 0.02 * peri, True)

          It applies the Douglas-Peucker algorithm to simplify the contour.
          What it does:
           1. Reduces a complex contour (many points) → simpler polygon
           2. Keeps the general shape, removes noise/jagged edges
          """
            if draw_contours:
                print(
                    f"Accepted contour with area {area} and bounding box (x={x}, y={y}, w={w}, h={h})"
                )
                cv2.putText(
                    countour_img,
                    str(len(approx)),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    contour_text_color,
                    contour_text_thickness,
                )

            center_x, center_y = x + (w // 2), y + (h // 2)
            cv2.rectangle(
                countour_img,
                (x, y),
                (x + w, y + h),
                contour_box_color,
                contour_box_thickness,
            )
            cv2.circle(
                countour_img, (center_x, center_y), 5, contour_box_color, cv2.FILLED
            )
            # Check if the vertex count matches the filter_shapes criteria (if provided)
            if len(filter_shapes) != 0 and len(approx) not in filter_shapes:
                continue

            contours_results.append(
                {
                    "contour": cnt,
                    "area": area,
                    "bounding_box": (x, y, w, h),
                    "approx_vertices": approx,
                    "center": (center_x, center_y),
                }
            )

            # filtered_contours.append(cnt)
            # boxes.append((x, y, w, h))

            if draw_contours:
                cv2.drawContours(countour_img, [cnt], -1, (0, 255, 0), 2)

        if debug:
            print(f"Total contours (raw): {len(contours)}")
            print(f"Filtered contours: {len(contours_results)}")
            for i, c in enumerate(contours_results):
                x, y, w, h = c["bounding_box"]
                print(
                    f"[{i}] Area={c['area']:.2f}, Box=({x},{y},{w},{h}), Center={c['center']}"
                )

        if sort_bbox_area_largest_to_smallest:
            contours_results = sorted(
                contours_results, key=lambda c: c["area"], reverse=True
            )

        if sort_contours_smallest_to_largest:
            contours_results = sorted(contours_results, key=lambda c: c["area"])

        if sort_countours_largest_to_smallest:
            contours_results = sorted(
                contours_results, key=lambda c: c["area"], reverse=True
            )
            # filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

        if sort_bbox_smaller_to_largest:
            contours_results = sorted(
                contours_results,
                key=lambda c: c["bounding_box"][2] * c["bounding_box"][3],  # w * h
            )

        if sort_bbox_top_to_bottom:
            # Sort top-to-bottom
            contours_results = sorted(
                contours_results,
                key=lambda c: (c["bounding_box"][1], c["bounding_box"][0]),  # y, then x
            )
            # boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        if sort_bbox_left_to_right:
            contours_results = sorted(
                contours_results,
                key=lambda c: c["bounding_box"][0],  # x
            )

        if sort_bbox_grid_wise:
            contours_results = sorted(
                contours_results,
                key=lambda c: (
                    c["bounding_box"][1] // bbox_grid_tolerence,
                    c["bounding_box"][0],  # then sort in row
                ),
            )

            # boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        return contours_results, countour_img

    def detect_contours(self, min_area=500):
        edges = self.get_cannty_edges()
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def export_measurements(data, path="measurements.json"):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def draw_grid(self, pixels_per_cm, color=(200, 200, 200)):
        h, w = self.image.shape[:2]
        step = int(pixels_per_cm)
        for x in range(0, w, step):
            cv2.line(self.image, (x, 0), (x, h), color, 1)

        for y in range(0, h, step):
            cv2.line(self.image, (0, y), (w, y), color, 1)

        return self.image

    def detect_reference(self, contours):
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue

            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]

            if w == 0 or h == 0:
                continue

            aspect = max(w, h) / min(w, h)

            if 1.3 < aspect < 1.5:
                candidates.append(("A4", cnt, 21.0))

            elif 1.5 < aspect < 1.7:
                candidates.append(("CARD", cnt, 8.56))

        if candidates:
            # choose largest
            candidates.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)
            label, cnt, real_size = candidates[0]

            return cnt, real_size, label

        # fallback → AI detection (optional)
        return None, None, None

    # ─────────────────────────── NEW METHODS ───────────────────────────

    def get_image_info(self) -> dict:
        """Return basic metadata about the current image.

        Returns:
          dict: {'height', 'width', 'channels', 'dtype', 'size_bytes'}
        """
        h, w = self.image.shape[:2]
        channels = self.image.shape[2] if self.image.ndim == 3 else 1
        return {
            "height": h,
            "width": w,
            "channels": channels,
            "dtype": str(self.image.dtype),
            "size_bytes": self.image.nbytes,
        }

    def apply_clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE).
        Better than global histogram equalization for documents with uneven lighting.

        Args:
          clip_limit: Threshold for contrast limiting.
          tile_grid_size: Size of the grid for histogram equalization.
        Returns:
          Grayscale numpy array with enhanced contrast.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def detect_blur(self, threshold=100.0) -> bool:
        """Return True if the image is blurry (Laplacian variance below threshold).
        Useful for quality-gating OCR or capture pipelines.

        Args:
          threshold: Variance below this value = blurry. Typical good-image range: 200+.
        Returns:
          bool: True = blurry.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def get_blur_score(self) -> float:
        """Return the Laplacian variance as a focus/sharpness score.
        Higher = sharper. Useful for ranking multiple captures.

        Returns:
          float
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_brightness(self) -> float:
        """Return mean pixel brightness of the image (0–255).
        Useful for auto-exposure feedback or quality checks.

        Returns:
          float
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def crop(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Crop a rectangular region from the image.

        Args:
          x, y: Top-left corner coordinates.
          w, h: Width and height of the crop.
        Returns:
          BGR numpy array crop.
        """
        H, W = self.image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        return self.image[y1:y2, x1:x2].copy()

    def flip(self, direction="horizontal") -> np.ndarray:
        """Flip the image horizontally or vertically.

        Args:
          direction: 'horizontal' | 'vertical' | 'both'
        Returns:
          Flipped BGR numpy array.
        """
        flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(direction, 1)
        self.image = cv2.flip(self.image, flip_code)
        return self.image

    def adjust_brightness_contrast(self, alpha=1.0, beta=0) -> np.ndarray:
        """Apply linear brightness/contrast adjustment: output = alpha * input + beta.

        Args:
          alpha: Contrast multiplier (1.0 = no change, > 1 = more contrast).
          beta: Brightness offset added to all pixels (-255 to 255).
        Returns:
          Adjusted BGR numpy array.
        """
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self.image

    def denoise(self, strength=10) -> np.ndarray:
        """Apply fast non-local means denoising — good for scanned document noise.

        Args:
          strength: Filter strength; higher = more noise removed but more blurry.
        Returns:
          Denoised BGR numpy array.
        """
        self.image = cv2.fastNlMeansDenoisingColored(
            self.image, None, strength, strength, 7, 21
        )
        return self.image

    def extract_text_regions(self, boxes):
        """
        Extract text regions from the input image based on the provided bounding boxes. This method takes an image and a
        list of bounding boxes (each defined by its top-left corner coordinates and dimensions) and extracts the corresponding regions of interest (ROIs) from the image.
        It then applies OCR to each extracted region to obtain the text contained within it. The method returns a list of dictionaries, where each dictionary contains the
        bounding box coordinates and the extracted text for that region. This can be useful for analyzing specific areas of the image or for further processing of the detected text.

        Args:
            image (np.ndarray): The input image from which to extract text regions.
            boxes (List[Tuple[int, int, int, int]]): A list of bounding boxes, where each box is defined by a tuple of (x, y, width, height)

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the bounding box coordinates and the extracted text for that region. For example: [{"bbox": (x1, y1, x2, y2), "text": "Extracted text from the region"}, ...
        """
        results = []

        # Crop the image based on the bounding boxes and apply OCR to each region
        for x, y, w, h in boxes:
            roi = self.image[y : y + h, x : x + w]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            text = pytesseract.image_to_string(thresh, config="--psm 6")

            results.append({"bbox": (x, y, x + w, y + h), "text": text.strip()})

        return results

    # ─────────────────────────── UTILITY METHODS ───────────────────────────

    def resize_to_fit(self, max_width: int, max_height: int) -> np.ndarray:
        """Resize the image to fit within max_width x max_height, preserving aspect ratio.

        Args:
          max_width: Maximum output width in pixels.
          max_height: Maximum output height in pixels.
        Returns:
          Resized BGR numpy array.
        """
        h, w = self.image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def pad_to_square(self, fill: int = 0) -> np.ndarray:
        """Pad the image with a constant border to make it square.

        The shorter dimension is padded symmetrically; the longer dimension is unchanged.

        Args:
          fill: Constant pixel value used for padding (0 = black).
        Returns:
          Square BGR numpy array.
        """
        h, w = self.image.shape[:2]
        size = max(h, w)
        pad_h = size - h
        pad_w = size - w
        top, left = pad_h // 2, pad_w // 2
        return cv2.copyMakeBorder(
            self.image,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=fill,
        )

    def normalize(self, mean=(0, 0, 0), std=(1, 1, 1)) -> np.ndarray:
        """Normalize pixel values to float32 in [0, 1] and optionally subtract mean / divide by std.

        Args:
          mean: Per-channel mean to subtract after scaling to [0, 1].
          std: Per-channel std to divide by after mean subtraction.
        Returns:
          float32 numpy array.
        """
        img = self.image.astype(np.float32) / 255.0
        return ((img - np.array(mean)) / np.array(std)).astype(np.float32)

    def create_thumbnail(self, size=(128, 128)) -> np.ndarray:
        """Resize the image to a fixed thumbnail size (no aspect-ratio preservation).

        Args:
          size: (width, height) tuple for the output thumbnail.
        Returns:
          BGR numpy array of shape (height, width, 3).
        """
        return cv2.resize(self.image, size, interpolation=cv2.INTER_AREA)

    def batch_crop(self, boxes) -> list:
        """Crop multiple regions from the image at once.

        Args:
          boxes: Iterable of (x, y, w, h) tuples in pixel coordinates.
                 Coordinates are clamped to image boundaries.
        Returns:
          List of BGR numpy array crops, one per box.
        """
        h, w = self.image.shape[:2]
        crops = []
        for x, y, bw, bh in boxes:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            crops.append(self.image[y1:y2, x1:x2].copy())
        return crops

    def get_dominant_colors(self, k: int = 5) -> list:
        """Return k dominant BGR colors using K-means clustering on all pixels.

        Args:
          k: Number of clusters / dominant colors to return.
        Returns:
          List of k (B, G, R) tuples as integers.
        """
        pixels = self.image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        return [tuple(int(c) for c in color) for color in centers]

    def overlay_image(
        self, overlay: np.ndarray, x: int, y: int, alpha: float = 1.0
    ) -> np.ndarray:
        """Blend an overlay image onto a copy of self.image at position (x, y).

        Args:
          overlay: BGR numpy array to blend in.
          x: Left edge of the overlay region (pixels).
          y: Top edge of the overlay region (pixels).
          alpha: Opacity of the overlay (0.0 = invisible, 1.0 = fully opaque).
        Returns:
          New BGR numpy array with the overlay blended in.
        """
        out = self.image.copy()
        h, w = overlay.shape[:2]
        y2 = min(y + h, out.shape[0])
        x2 = min(x + w, out.shape[1])
        oh, ow = y2 - y, x2 - x
        if oh > 0 and ow > 0:
            roi = out[y:y2, x:x2]
            out[y:y2, x:x2] = cv2.addWeighted(
                roi, 1 - alpha, overlay[:oh, :ow], alpha, 0
            )
        return out

    def compare_histograms(self, other_image: np.ndarray) -> float:
        """Compare self.image to another image using 3D BGR histogram correlation.

        Both histograms are normalised to [0, 1] before comparison.

        Args:
          other_image: BGR numpy array to compare against.
        Returns:
          Correlation score in [-1, 1]; 1.0 = identical histogram.
        """

        def _hist(img):
            h = cv2.calcHist(
                [img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return h

        return float(
            cv2.compareHist(_hist(self.image), _hist(other_image), cv2.HISTCMP_CORREL)
        )

    def to_base64(self) -> str:
        """Encode the image as a base64 PNG string (UTF-8).

        Useful for embedding images in JSON payloads or HTML data URIs.

        Returns:
          Base64-encoded string of the PNG-encoded image.
        """
        import base64

        _, buf = cv2.imencode(".png", self.image)
        return base64.b64encode(buf).decode("utf-8")
