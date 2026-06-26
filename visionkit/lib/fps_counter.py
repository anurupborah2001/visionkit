import time

import cv2


class FPSCounter:
    """
    A simple FPS (Frames Per Second) counter for video processing.
    This class calculates and displays the FPS of a video feed. It uses the time difference between frames to compute the FPS and overlays it on the video frame.

    Usage:
    fps_counter = FPSCounter()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame, fps = fps_counter.update(frame)
        cv2.imshow('Video Feed', frame)
    """

    def __init__(self):
        self.prev_time = 0

    def update(self, frame):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time else 0
        self.prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return frame, fps
