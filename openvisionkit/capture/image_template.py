from collections.abc import Callable

import cv2
import numpy as np
import pyautogui

window_centered = False  # Used to center window only once


def image_template(
    image_path: str,
    custom_logic: Callable[[cv2.typing.MatLike], cv2.typing.MatLike] | None = None,
    window_name: str = "Demo",
    center_window: bool = True,
    show_window: bool = True,
    resolution: tuple[int, int] = (1280, 720),
):
    """
    REUSABLE TEMPLATE for displaying an image with optional custom processing.

    Parameters:
        image_path (str): Path to the image file.
        custom_logic (callable, optional): Function that receives the image and returns the modified image.
        window_name (str): Name of the OpenCV window.
        center_window (bool): If True, automatically centers the window on screen. Default = True
        show_window (bool): If True, displays the image window. Default = True
        resolution (tuple[int, int]): Desired image resolution (width, height). Default = (1280, 720)
    """

    image = cv2.imread(image_path)
    print(image_path)
    if custom_logic is not None:
        image = custom_logic(image)

    if image is None:
        raise ValueError("Image is None. Check file path or loading logic.")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Invalid image type: {type(image)}")

    if image.size == 0:
        raise ValueError("Empty image array")

    # Resize image
    hWIDTH, hHEIGHT = resolution
    resized_img = cv2.resize(image, (hWIDTH, hHEIGHT))

    # ======================
    # NEW: Auto-center window on screen (only once)
    # ======================
    if center_window:
        screen_width, screen_height = pyautogui.size()
        x = int((screen_width - hWIDTH) / 2)
        y = int((screen_height - hHEIGHT) / 2)
        cv2.moveWindow(window_name, x, y)
    # ======================

    if show_window:
        cv2.imshow(window_name, resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
