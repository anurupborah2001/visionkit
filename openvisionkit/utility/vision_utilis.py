import inspect
import math
import os
import textwrap
import time
from glob import glob

import cv2
import numpy as np


def rectangle_corners(
    img, bbox, length=30, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0)
):
    """
    Draws a rectangle with decorative corners on the given image.

    Args:
        img: The image on which to draw.
        bbox: A tuple (x, y, w, h) representing the bounding box.
        l: Length of the corner lines.
        t: Thickness of the corner lines.
        rt: Thickness of the rectangle border. If 0, no border is drawn.
        colorR: Color of the rectangle border.
        colorC: Color of the corner lines.

    Returns:
        The image with the decorative rectangle drawn.
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rt:
        cv2.rectangle(img, bbox, colorR, rt)

    for (cx, cy), dx, dy in [
        ((x, y), length, length),
        ((x1, y), -length, length),
        ((x, y1), length, -length),
        ((x1, y1), -length, -length),
    ]:
        cv2.line(img, (cx, cy), (cx + dx, cy), colorC, t)
        cv2.line(img, (cx, cy), (cx, cy + dy), colorC, t)

    return img


def detect_highlighted_text(
    img: np.ndarray,
    hsv_colors=None,  # seed HSV values
    h_tol=10,
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
      img: Input image in BGR format (as read by OpenCV)
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
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    masks = []

    for i, (h, s, v) in enumerate(hsv_colors):
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

    img_with_mask = cv2.bitwise_and(img, img, mask=combined_mask)
    if show_combined_mask:
        cv2.imshow("Combined Mask", combined_mask)
    if show_image_with_mask:
        img_with_mask = cv2.bitwise_and(img, img, mask=combined_mask)
        cv2.imshow("Image with Mask", img_with_mask)

    if show_mask or show_combined_mask or show_image_with_mask:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_with_mask, combined_mask, masks


def get_dominant_hsv_colors(image, k=4):
    """
    auto-detect highlight colors in the image by clustering pixel colors in HSV space using K-means.
    Get dominant HSV colors from the image using K-means clustering.

    Args:
        image: Input image in BGR format (as read by OpenCV)
        k: Number of dominant colors to detect (default is 4)

    Returns:
        List of dominant HSV color tuples (h, s, v) detected in the image.
    Usage:

    """
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    pixels = np.float32(pixels)

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    return [tuple(map(int, c)) for c in centers]


def refine_mask(mask):
    """
    Refines a binary mask by applying morphological operations to remove noise and merge words in the same line.

    Args:
        mask: The input binary mask to be refined.

    Returns:
        The refined binary mask.
    """

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_line = np.ones((15, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # Merge words in same line
    mask = cv2.dilate(mask, kernel_line, iterations=1)
    return mask


def detect_single_highlighted_text(image, hsv_colors=None):
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

    # Example: detect blue color
    print(hsv[0])
    lower = np.array([hsv_colors[0], hsv_colors[1], hsv_colors[2]])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    img_with_mask = cv2.bitwise_and(image, image, mask=mask)
    return img_with_mask, mask


def find_contours(
    mask,
    min_area=500,
    debug=False,
    sort_countours=False,
    sorted_bounding_box=False,
    retrieval_type=cv2.RETR_EXTERNAL,
    approximation_method=cv2.CHAIN_APPROX_SIMPLE,
):
    """
    Find contours in a binary mask and filter them based on area and other criteria.

    Args:
        mask: Binary image (mask) where contours are to be found.
        min_area: Minimum area threshold to filter contours (default is 500).
        debug: If True, prints debug information about contours found and filtered.
        sort_contours: If True, sorts contours by area in descending order (default is False).
        retrieval_type: Contour retrieval mode (default is cv2.RETR_EXTERNAL).
        approximation_method: Contour approximation method (default is cv2.CHAIN_APPROX_SIMPLE).

    Returns:
        filtered_contours: List of contours that passed the filtering criteria.
        boxes: List of bounding box tuples (x, y, w, h) for the
    """

    # 1. Clean noise (very important)
    cleaned = refine_mask(mask)
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

    filtered_contours = []
    boxes = []

    # 3. Filter contours
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 10:
            continue

        filtered_contours.append(cnt)
        boxes.append((x, y, w, h))

    if debug:
        print(f"Total contours: {len(contours)}")
        print(f"Filtered contours: {len(filtered_contours)}")

    if sort_countours:
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    if sorted_bounding_box:
        # Sort top-to-bottom
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return filtered_contours, boxes


def resize_with_padding(img, target_size, color=(0, 0, 0)):
    """
    Resize an image while maintaining aspect ratio and adding padding to fit the target size.

    Args:
        img: The input image to be resized.
        target_size: A tuple (width, height) representing the desired output size.
        color: The color of the padding (default is black).

    Returns:
        The resized image with padding to fit the target size.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def highlight_image(img, selected=False, color=(0, 255, 0), thickness=3):
    """
    Highlight Selected Image (Border)

    Args:
        img: The input image to be highlighted.
        selected: A boolean indicating whether to apply the highlight (default is False).
        color: The color of the highlight border (default is green).
        thickness: The thickness of the highlight border (default is 3).

    Returns:
        The image with the highlight border applied if selected is True, otherwise the original image.

    Usage:
        img = cv2.imread('input.jpg')
        highlighted_img = highlight_image(img, selected=True, color=(0, 255, 0), thickness=3)
        cv2.imshow('Highlighted Image', highlighted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    if selected:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), color, thickness)
    return img


def zoom_image(img, scale=2.0):
    """
    Zooms into the image by a specified scale factor.

    Args:
        img: The input image to be zoomed.
        scale: The zoom scale factor (default is 2.0, which means 200% zoom).
    Returns:
        The zoomed image.
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def put_text_think_corners(
    img, bounding_box, color=(255, 255, 255), length=20, thickness=2
):
    """
    Puts text on the image with a colored rectangle background for better visibility, using the rectangle_corners function for decorative corners.

    Args:
        img: The input image on which to put the text.
        bounding_box: A tuple (x, y, w, h) representing the bounding box coordinates.
        color: The color of the text and corners (default is white).
        length: The length of the corner lines (default is 20).
        thickness: The thickness of the corner lines (default is 2).

    Returns:
        The image with the text and decorative corners drawn on it.
    """
    x, y, w, h = bounding_box
    # Top Left corner
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Top Right corner
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom Left corner
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom Right corner
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    return img


def overlay_transparent(bg, fg, pos=(0, 0)):
    x, y = pos
    h, w = fg.shape[:2]
    H, W = bg.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, W), min(y + h, H)

    if x1 >= x2 or y1 >= y2:
        return bg

    fx1, fy1 = max(-x, 0), max(-y, 0)
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    fg_crop = fg[fy1:fy2, fx1:fx2]

    # 🔥 Handle alpha safely
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[..., 3:4] / 255.0
        fg_rgb = fg_crop[..., :3]
    else:
        # No alpha → treat as fully opaque
        alpha = np.ones((fg_crop.shape[0], fg_crop.shape[1], 1), dtype=np.float32)
        fg_rgb = fg_crop

    bg_crop = bg[y1:y2, x1:x2]

    bg[y1:y2, x1:x2] = (bg_crop * (1 - alpha) + fg_rgb * alpha).astype(bg.dtype)

    return bg


def get_valid_images(folder_path):
    """
    Returns a list of valid image files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing images.

    Returns:
        list: A list of valid image file names.
    """
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]
    files = []

    for p in patterns:
        files.extend(glob(os.path.join(folder_path, p)))

    return files


def draw_rounded_rect(img, top_left, bottom_right, color, radius=20, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def is_hovering(button, point):
    px, py = point
    x, y = button.pos

    return x <= px <= x + button.width and y <= py <= y + button.height


def create_centered_grid_buttons(
    frame,
    values,
    button_cls,
    button_size=(100, 100),
    gap=15,
    y_offset=0,
):
    """
    Creates centered grid buttons for any OpenCV frame.

    values example:
    [
        ['7', '8', '9', '*'],
        ['4', '5', '6', '-']
    ]
    """

    frame_h, frame_w = frame.shape[:2]

    rows = len(values)
    cols = len(values[0])

    btn_w, btn_h = button_size

    grid_w = cols * btn_w + (cols - 1) * gap
    grid_h = rows * btn_h + (rows - 1) * gap

    start_x = (frame_w - grid_w) // 2
    start_y = (frame_h - grid_h) // 2 + y_offset

    buttons = []

    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (btn_w + gap)
            y = start_y + row * (btn_h + gap)

            buttons.append(
                button_cls(pos=(x, y), text=values[row][col], size=button_size)
            )

    return buttons


def put_text_rect(
    img,
    text,
    pos,
    scale=3,
    thickness=3,
    colorT=(255, 255, 255),
    colorR=(255, 0, 255),
    font=cv2.FONT_HERSHEY_PLAIN,
    offset=10,
    border=None,
    colorB=(0, 255, 0),
):
    """
    Puts text on the image with a colored rectangle background for better visibility.
    Args:        img: The input image on which to put the text.
        text: The text string to be displayed.
        pos: A tuple (x, y) representing the bottom-left corner of the text.
        scale: The font scale factor (default is 3).
        thickness: The thickness of the text (default is 3).
        colorT: The color of the text (default is white).
        colorR: The color of the rectangle background (default is magenta).
        font: The font type (default is cv2.FONT_HERSHEY_PLAIN).
        offset: The offset for the rectangle padding (default is 10).
        border: The thickness of the border around the rectangle (default is None, no border).
        colorB: The color of the border (default is green).
    Returns:
       The image with the text and rectangle drawn on it.
    """
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = pos[0] - offset, pos[1] + offset
    x2, y2 = pos[0] + w + offset, pos[1] - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, -1)
    if border:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, pos, font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


def draw_wrapped_text(
    img, text, start_pos, font, scale, color, thickness, max_width=40
):
    wrapped = textwrap.wrap(text, width=max_width)
    x, y = start_pos

    for i, line in enumerate(wrapped):
        y_offset = y + i * int(30 * scale)
        cv2.putText(img, line, (x, y_offset), font, scale, color, thickness)


def load_image(image_input):
    """
    Load an image from a file path or return the image if it's already a numpy array.

    Args:
        image_input (str or np.ndarray): The file path of the image or a numpy array.

    Returns:
        np.ndarray: The loaded image.

    Raises:
        ValueError: If the image path is invalid or the file is unreadable.
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    # assume it's a path
    img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image path or unreadable file")

    return img


def move_image(
    img, direction="left_to_right", speed=5, start_pos=(0, 0), window_size=(800, 600)
):
    """
    Moves an image across the screen in a specified direction at a given speed.

    Args:
        img: The file path of the image or a numpy array to be moved.
        direction: The direction of movement (e.g., "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "diag_tl_br", "diag_br_tl").
        speed: The speed of movement in pixels per frame (default is 5).
        start_pos: A tuple (x, y) representing the starting position of the image (default is (0, 0)).
        window_size: A tuple (width, height) representing the size of the display window (default is (800, 600)).
    Returns:
      None
    Usage:
        move_image("path/to/image.png", direction="left_to_right", speed=5, start_pos=(0, 0), window_size=(800, 600))
    """
    img = load_image(img)

    h, w = img.shape[:2]
    canvas_w, canvas_h = window_size

    x, y = start_pos

    while True:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        canvas_x = int(x)
        canvas_y = int(y)

        # place image on canvas safely
        x1, y1 = max(canvas_x, 0), max(canvas_y, 0)
        x2, y2 = min(canvas_x + w, canvas_w), min(canvas_y + h, canvas_h)

        img_x1, img_y1 = max(-canvas_x, 0), max(-canvas_y, 0)
        img_x2, img_y2 = img_x1 + (x2 - x1), img_y1 + (y2 - y1)

        if x1 < x2 and y1 < y2:
            canvas[y1:y2, x1:x2] = img[img_y1:img_y2, img_x1:img_x2]

        cv2.imshow("Moving Image", canvas)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to stop
            break

        # movement logic
        if direction == "left_to_right":
            x += speed
        elif direction == "right_to_left":
            x -= speed
        elif direction == "top_to_bottom":
            y += speed
        elif direction == "bottom_to_top":
            y -= speed
        elif direction == "diag_tl_br":
            x += speed
            y += speed
        elif direction == "diag_br_tl":
            x -= speed
            y -= speed

        time.sleep(0.01)

    cv2.destroyAllWindows()


def stack_images_grid(img_list, cols=2, scale=1.0, labels=None, bg_color=(0, 0, 0)):
    """
    Stack images in a grid.

    Args:
        img_list (list): list of images
        cols (int): number of columns
        scale (float): scaling factor
        labels (list): optional titles for each image
        bg_color (tuple): background color for empty slots

    Returns:
        np.ndarray: stacked image
    """

    if not img_list:
        return None

    # Convert all images to the same size and color format
    h, w = img_list[0].shape[:2]

    processed = []
    for img in img_list:
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert grayscale → BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (w, h))
        processed.append(img)

    # Calculate rows needed and fill remaining slots with blank images
    total = len(processed)
    rows = math.ceil(total / cols)

    # Fill remaining slots with blank images
    blank = np.full((h, w, 3), bg_color, dtype=np.uint8)
    processed += [blank] * (rows * cols - total)

    # Add labels if provided
    if labels:
        for i, text in enumerate(labels):
            if i < len(processed):
                cv2.putText(
                    processed[i],
                    text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    # Stack
    grid = [np.hstack(processed[i * cols : (i + 1) * cols]) for i in range(rows)]
    stacked = np.vstack(grid)

    # Scale final output
    if scale != 1.0:
        stacked = cv2.resize(stacked, (0, 0), fx=scale, fy=scale)

    return stacked


def get_currect_path():
    """
    Get the current file path.

    Returns:
        str: The absolute path of the current file.
    """
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    return path, filename, full_path


def get_calling_folder():
    """
    Get calling folder path.
    Returns:
        str: The absolute path of the calling folder.
    """
    frame = inspect.stack()[1]
    calling_file = frame.filename
    return os.path.dirname(os.path.abspath(calling_file)), calling_file


def find_project_root(start_path, marker="module"):
    """
    Finds the project root directory by looking for a specific marker (e.g., a folder or file).

    Args:
        start_path: The path from which to start searching.
        marker: The name of the folder or file that indicates the project root.

    Returns:
        The absolute path to the project root directory.
    """
    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Project root not found")
        current = parent


def mouse_drag_overlay(event, x, y, flags, param):
    """
    Mouse callback function to enable dragging of an overlay on the video frame. It updates the position of the overlay based on mouse events.

    Args:
      event: The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP).
      x: The x-coordinate of the mouse event.
      y: The y-coordinate of the mouse event.
      flags: Any relevant flags passed by OpenCV (not used in this function).
      param: A dictionary containing the current position of the overlay and its dimensions, as well as a flag to track dragging state.

    Returns:
      None. The function updates the position of the overlay in the param dictionary based on mouse interactions
    """
    drag_position = param["drag_position"]
    overlay_w = param["overlay_w"]
    overlay_h = param["overlay_h"]

    ox = drag_position["x"]
    oy = drag_position["y"]

    inside_overlay = ox <= x <= ox + overlay_w and oy <= y <= oy + overlay_h

    if event == cv2.EVENT_LBUTTONDOWN and inside_overlay:
        drag_position["dragging"] = True
        drag_position["offset_x"] = x - ox
        drag_position["offset_y"] = y - oy

    elif event == cv2.EVENT_MOUSEMOVE and drag_position["dragging"]:
        drag_position["x"] = x - drag_position["offset_x"]
        drag_position["y"] = y - drag_position["offset_y"]

    elif event == cv2.EVENT_LBUTTONUP:
        drag_position["dragging"] = False


def overlay_frame(
    background,
    overlay,
    position="top-left",
    padding=10,
    draggable=False,
    drag_position=None,
    draw_border=True,
):
    """
    Creates an overlay of one image on top of another at a specified position with optional padding and dragging functionality.

    Args:
      background: The background image on which to overlay.
      overlay: The image to be overlaid on the background.
      position: The position to place the overlay (default is "top-left"). Options: "top-left", "top-right", "bottom-left", "bottom-right", "center".
      padding: The padding in pixels from the edges of the background (default is 10).
      draggable: If True, allows the overlay to be dragged (default is False).
      drag_position: A dictionary with "x" and "y" keys to track the current position of the overlay when dragging (default is None).
      draw_border: If True, draws a border around the overlay for better visibility (default is True).

    Returns:
      The background image with the overlay applied at the specified position.
    """

    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    if draggable and drag_position is not None:
        x = drag_position["x"]
        y = drag_position["y"]
    else:
        if position == "top-left":
            x, y = padding, padding
        elif position == "top-right":
            x, y = bg_w - ov_w - padding, padding
        elif position == "bottom-left":
            x, y = padding, bg_h - ov_h - padding
        elif position == "bottom-right":
            x, y = bg_w - ov_w - padding, bg_h - ov_h - padding
        elif position == "center":
            x = (bg_w - ov_w) // 2
            y = (bg_h - ov_h) // 2
        else:
            x, y = padding, padding

    # keep inside image
    x = max(0, min(x, bg_w - ov_w))
    y = max(0, min(y, bg_h - ov_h))

    if draggable and drag_position is not None:
        drag_position["x"] = x
        drag_position["y"] = y

    background[y : y + ov_h, x : x + ov_w] = overlay
    return background


def auto_layout(drawings, frame_shape, cols=4, padding=20):
    frame_h, frame_w = frame_shape[:2]

    max_w = max(d.size[0] for d in drawings)
    max_h = max(d.size[1] for d in drawings)

    cell_w = max_w + padding
    cell_h = max_h + padding

    for idx, drawing in enumerate(drawings):
        row = idx // cols
        col = idx % cols

        x = padding + col * cell_w
        y = padding + row * cell_h

        drawing.origin = (x, y)
