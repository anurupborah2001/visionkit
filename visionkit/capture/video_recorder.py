import os
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import imageio


@dataclass
class VideoRecorder:
    """
    Advanced Video Recorder Engine:
    - MP4 or GIF export
    - Pause / Resume
    - Timer tracking
    - Multi-source ready (webcam + screen overlay)
    - Plug & play with video_capture_template
    """

    output_path: str = "recordings"
    fps: int = 20
    codec: str = "mp4v"
    output_format: str = "mp4"  # "mp4" or "gif"

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.file_path = os.path.join(
            self.output_path, f"record_{timestamp}.{self.output_format}"
        )

        # VideoWriter (MP4 mode)
        self.writer = None

        # GIF buffer mode
        self.frames = []

        self.is_recording = False
        self.is_paused = False

        self.start_time = None
        self.elapsed_time = 0

    # ----------------------------
    # START RECORDING
    # ----------------------------
    def start(self, frame_shape=None):
        self.is_recording = True
        self.is_paused = False
        self.start_time = time.time()

        self.frames = []

        if self.output_format == "mp4" and frame_shape is not None:
            h, w = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.file_path, fourcc, self.fps, (w, h))

        print(f"🔴 Recording started → {self.file_path}")

    # ----------------------------
    # WRITE FRAME
    # ----------------------------
    def write(self, frame):
        if not self.is_recording or self.is_paused:
            return

        # -------------------------
        # Convert BGR → RGB (IMPORTANT)
        # -------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MP4 mode
        if self.output_format == "mp4" and self.writer:
            self.writer.write(frame)  # MP4 expects BGR (OpenCV standard)

        # GIF mode
        elif self.output_format == "gif":
            self.frames.append(rgb_frame)

    # ----------------------------
    # TIMER
    # ----------------------------
    def get_elapsed_time(self):
        if self.start_time:
            return round(time.time() - self.start_time, 2)
        return 0

    # ----------------------------
    # PAUSE / RESUME
    # ----------------------------
    def pause(self):
        self.is_paused = True
        print("⏸ Recording paused")

    def resume(self):
        self.is_paused = False
        print("▶ Recording resumed")

    # ----------------------------
    # STOP RECORDING
    # ----------------------------
    def stop(self):
        self.is_recording = False

        if self.writer:
            self.writer.release()
            print(f"✅ MP4 saved → {self.file_path}")

        if self.output_format == "gif":
            if not self.frames or len(self.frames) == 0:
                print("⚠️ No frames recorded. Skipping GIF export.")
                return
            safe_fps = self.fps if self.fps and self.fps > 0 else 10
            imageio.mimsave(self.file_path, self.frames, fps=safe_fps)
            print(f"✅ GIF saved → {self.file_path}")

    # ----------------------------
    # AUDIO SYNC (HOOK)
    # ----------------------------
    def attach_audio(self, audio_path: str):
        """
        Placeholder for ffmpeg-based audio sync.
        """
        print(f"🎙 Audio sync not implemented yet: {audio_path}")
