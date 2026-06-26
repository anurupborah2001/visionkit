import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from conftest import blank_bgr

from visionkit.capture.video_template import (
    KeyEventManager,
    save_screenshot,
    video_capture_template,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class FakeCapture:
    """Drop-in for cv2.VideoCapture that returns a fixed sequence of frames."""

    def __init__(self, frames, opened=True):
        self._frames = list(frames)
        self._pos = 0
        self._opened = opened

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self):
        if self._pos < len(self._frames):
            frame = self._frames[self._pos].copy()
            self._pos += 1
            return True, frame
        return False, None

    def release(self):
        self._opened = False


def _run(fake_cap, **kwargs):
    """Call video_capture_template with headless defaults and a FakeCapture."""
    defaults = {"show_window": False, "draw_fps": False, "loop_forever": False}
    defaults.update(kwargs)
    with (
        patch(
            "visionkit.capture.video_template.cv2.VideoCapture", return_value=fake_cap
        ),
        patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
        patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
    ):
        video_capture_template(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# KeyEventManager
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestKeyEventManager:
    def test_register_stores_handler(self):
        km = KeyEventManager()
        cb = MagicMock()
        km.register(ord("r"), cb)
        assert ord("r") in km.handlers

    def test_handle_dispatches_to_registered_handler(self):
        km = KeyEventManager()
        cb = MagicMock()
        km.register(ord("r"), cb)
        frame = blank_bgr()
        state = {"x": 1}
        km.handle(ord("r"), frame, state)
        cb.assert_called_once_with(frame, state)

    def test_handle_ignores_unregistered_key(self):
        km = KeyEventManager()
        km.handle(ord("z"), blank_bgr(), {})  # must not raise

    def test_register_multiple_keys_dispatches_independently(self):
        km = KeyEventManager()
        cb1, cb2 = MagicMock(), MagicMock()
        km.register(ord("a"), cb1)
        km.register(ord("b"), cb2)
        km.handle(ord("a"), blank_bgr(), {})
        km.handle(ord("b"), blank_bgr(), {})
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_handle_passes_state_reference(self):
        km = KeyEventManager()
        seen = {}

        def cb(frame, state):
            seen.update(state)

        km.register(ord("s"), cb)
        km.handle(ord("s"), blank_bgr(), {"level": 7})
        assert seen == {"level": 7}

    def test_later_register_overwrites_same_key(self):
        km = KeyEventManager()
        cb1, cb2 = MagicMock(), MagicMock()
        km.register(ord("r"), cb1)
        km.register(ord("r"), cb2)
        km.handle(ord("r"), blank_bgr(), {})
        cb1.assert_not_called()
        cb2.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# save_screenshot
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSaveScreenshot:
    def test_returns_string_path(self, tmp_path):
        result = save_screenshot(blank_bgr(), output_dir=str(tmp_path), prefix="t")
        assert isinstance(result, str)

    def test_file_exists_after_save(self, tmp_path):
        path = save_screenshot(blank_bgr(), output_dir=str(tmp_path), prefix="t")
        assert Path(path).exists()

    def test_filename_starts_with_prefix(self, tmp_path):
        path = save_screenshot(blank_bgr(), output_dir=str(tmp_path), prefix="myprefix")
        assert Path(path).name.startswith("myprefix_")

    def test_output_is_png(self, tmp_path):
        path = save_screenshot(blank_bgr(), output_dir=str(tmp_path), prefix="t")
        assert path.endswith(".png")

    def test_creates_output_dir_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "subdir")
        save_screenshot(blank_bgr(), output_dir=new_dir, prefix="t")
        assert Path(new_dir).is_dir()

    def test_saved_image_readable_by_opencv(self, tmp_path):
        frame = blank_bgr()
        path = save_screenshot(frame, output_dir=str(tmp_path), prefix="t")
        loaded = cv2.imread(path)
        assert loaded is not None
        assert loaded.shape == frame.shape

    def test_multiple_calls_produce_at_least_one_file(self, tmp_path):
        for _ in range(3):
            save_screenshot(blank_bgr(), output_dir=str(tmp_path), prefix="t")
        files = list(Path(tmp_path).glob("t_*.png"))
        assert len(files) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# video_capture_template — unit tests (no real camera, mocked cv2)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVideoCaptureTemplateUnit:
    def test_exits_immediately_when_capture_not_opened(self):
        _run(FakeCapture([], opened=False), video_source=99)

    def test_custom_logic_called_once_per_frame(self):
        calls = []

        def logic(frame):
            calls.append(True)
            return frame

        _run(
            FakeCapture([blank_bgr(), blank_bgr(), blank_bgr()]),
            video_source=0,
            custom_logic=logic,
        )
        assert len(calls) == 3

    def test_custom_logic_return_value_replaces_frame(self):
        replacement = np.zeros((300, 400, 3), dtype=np.uint8)
        replacement[0, 0] = [9, 9, 9]
        received = []

        def logic(frame):
            received.append(frame.copy())
            return replacement

        _run(FakeCapture([blank_bgr()]), video_source=0, custom_logic=logic)
        assert len(received) == 1
        assert not np.array_equal(received[0], replacement)

    def test_no_custom_logic_does_not_raise(self):
        _run(FakeCapture([blank_bgr()]), video_source=0)

    def test_state_none_initialised_to_empty_dict(self):
        _run(FakeCapture([blank_bgr()]), video_source=0, state=None)

    def test_no_window_calls_when_show_window_false(self):
        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture([blank_bgr()]),
            ),
            patch("visionkit.capture.video_template.cv2.namedWindow") as mock_win,
            patch("visionkit.capture.video_template.cv2.imshow") as mock_show,
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
            )
        mock_win.assert_not_called()
        mock_show.assert_not_called()

    def test_capture_released_on_normal_exit(self):
        fake = FakeCapture([blank_bgr()])
        _run(fake, video_source=0)
        assert not fake._opened

    def test_esc_key_breaks_loop_before_frames_exhausted(self):
        calls = []

        def logic(frame):
            calls.append(True)
            return frame

        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture([blank_bgr() for _ in range(10)]),
            ),
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=27),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=0,
                custom_logic=logic,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
            )
        assert len(calls) == 1

    def test_key_manager_handle_called_each_frame(self):
        km = MagicMock(spec=KeyEventManager)
        _run(FakeCapture([blank_bgr(), blank_bgr()]), video_source=0, key_manager=km)
        assert km.handle.call_count == 2

    def test_draw_fps_true_does_not_raise(self):
        _run(FakeCapture([blank_bgr()]), video_source=0, draw_fps=True)

    def test_mouse_callback_set_when_show_window_true(self):
        cb = MagicMock()
        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture([blank_bgr()]),
            ),
            patch("visionkit.capture.video_template.cv2.namedWindow"),
            patch("visionkit.capture.video_template.cv2.resizeWindow"),
            patch(
                "visionkit.capture.video_template.cv2.setMouseCallback"
            ) as mock_set_cb,
            patch("visionkit.capture.video_template.cv2.imshow"),
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=27),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=0,
                show_window=True,
                center_window=False,
                draw_fps=False,
                loop_forever=False,
                mouse_callback=cb,
            )
        mock_set_cb.assert_called_once()

    def test_auto_recording_creates_recorder_writes_and_stops(self):
        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture([blank_bgr()]),
            ),
            patch("visionkit.capture.video_template.VideoRecorder") as MockVR,
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            instance = MagicMock()
            MockVR.return_value = instance
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
                enable_auto_recording=True,
                record_format="mp4",
            )
        MockVR.assert_called_once_with(output_format="mp4", fps=15)
        instance.start.assert_called_once()
        instance.write.assert_called_once()
        instance.stop.assert_called_once()

    def test_manual_recording_toggled_by_r_key(self):
        # Two frames: R on first toggles recording ON, natural end stops it
        frames = [blank_bgr(), blank_bgr()]
        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture(frames),
            ),
            patch("visionkit.capture.video_template.VideoRecorder") as MockVR,
            patch(
                "visionkit.capture.video_template.cv2.waitKey",
                side_effect=[ord("r"), 0],
            ),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            instance = MagicMock()
            MockVR.return_value = instance
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
                enable_manual_recording=True,
            )
        # Recorder lazily created inside loop once recording=True, then stopped at cleanup
        instance.stop.assert_called()

    def test_auto_screenshot_fires_once_when_repeat_false(self):
        frames = [blank_bgr() for _ in range(3)]
        screenshots = []

        # time.time calls: 1 before loop (start_time), then 1 per frame (now)
        time_values = iter([0.0, 1.0, 6.0, 7.0])

        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture(frames),
            ),
            patch(
                "visionkit.capture.video_template.save_screenshot",
                side_effect=lambda f, **kw: screenshots.append(1) or "/tmp/x.png",
            ),
            patch("visionkit.capture.video_template.time") as mock_time,
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            mock_time.time.side_effect = time_values
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
                enable_screenshot=True,
                auto_screenshot_after_seconds=5.0,
                auto_screenshot_repeat=False,
            )
        assert len(screenshots) == 1

    def test_auto_screenshot_repeats_on_each_interval(self):
        frames = [blank_bgr() for _ in range(4)]
        screenshots = []

        # start=0, frame1=0 (no fire), frame2=6 (fire→last=6), frame3=12 (fire→last=12), frame4=18 (fire)
        time_values = iter([0.0, 0.0, 6.0, 12.0, 18.0, 24.0])

        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture(frames),
            ),
            patch(
                "visionkit.capture.video_template.save_screenshot",
                side_effect=lambda f, **kw: screenshots.append(1) or "/tmp/x.png",
            ),
            patch("visionkit.capture.video_template.time") as mock_time,
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            mock_time.time.side_effect = time_values
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
                enable_screenshot=True,
                auto_screenshot_after_seconds=5.0,
                auto_screenshot_repeat=True,
            )
        assert len(screenshots) >= 2

    def test_manual_screenshot_triggered_by_s_key(self):
        screenshots = []
        with (
            patch(
                "visionkit.capture.video_template.cv2.VideoCapture",
                return_value=FakeCapture([blank_bgr()]),
            ),
            patch(
                "visionkit.capture.video_template.save_screenshot",
                side_effect=lambda f, **kw: screenshots.append(1) or "/tmp/x.png",
            ),
            patch(
                "visionkit.capture.video_template.cv2.waitKey", return_value=ord("s")
            ),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=0,
                show_window=False,
                draw_fps=False,
                loop_forever=False,
                enable_screenshot=True,
            )
        assert len(screenshots) == 1


# ──────────────────────────────────────────────────────────────────────────────
# video_capture_template — integration tests (real video file on disk)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """5-frame 640×480 MP4 written to a temp file."""
    p = tmp_path_factory.mktemp("video") / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, 10, (640, 480))
    for i in range(5):
        frame = np.full((480, 640, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(p)


@pytest.mark.integration
class TestVideoCaptureTemplateIntegration:
    def test_reads_frames_from_real_video_file(self, synthetic_video):
        count = []

        def logic(frame):
            count.append(True)
            return frame

        with (
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=synthetic_video,
                custom_logic=logic,
                loop_forever=False,
                show_window=False,
                draw_fps=False,
            )
        assert count, "expected at least one frame to be processed"

    def test_custom_logic_receives_3channel_bgr_frame(self, synthetic_video):
        shapes = []

        def logic(frame):
            shapes.append(frame.shape)
            return frame

        with (
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=synthetic_video,
                custom_logic=logic,
                loop_forever=False,
                show_window=False,
                draw_fps=False,
            )
        assert all(len(s) == 3 and s[2] == 3 for s in shapes)

    def test_auto_screenshot_writes_png_to_disk(self, synthetic_video, tmp_path):
        shot_dir = str(tmp_path / "shots")
        with (
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=synthetic_video,
                loop_forever=False,
                show_window=False,
                draw_fps=False,
                enable_screenshot=True,
                auto_screenshot_after_seconds=0.0001,
                auto_screenshot_repeat=False,
                screenshot_output_dir=shot_dir,
                screenshot_prefix="inttest",
            )
        files = list(Path(shot_dir).glob("inttest_*.png"))
        assert len(files) == 1

    def test_fps_counter_overlay_does_not_raise(self, synthetic_video):
        with (
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=synthetic_video,
                loop_forever=False,
                show_window=False,
                draw_fps=True,
            )

    def test_auto_recording_creates_output_file(self, synthetic_video, tmp_path):
        from visionkit.capture.video_recorder import VideoRecorder

        original_init = VideoRecorder.__post_init__

        created_paths = []

        def patched_init(self):
            self.output_path = str(tmp_path / "rec")
            original_init(self)
            created_paths.append(self.file_path)

        with (
            patch.object(VideoRecorder, "__post_init__", patched_init),
            patch("visionkit.capture.video_template.cv2.waitKey", return_value=0),
            patch("visionkit.capture.video_template.cv2.destroyAllWindows"),
        ):
            video_capture_template(
                video_source=synthetic_video,
                loop_forever=False,
                show_window=False,
                draw_fps=False,
                enable_auto_recording=True,
                record_format="mp4",
            )
        assert created_paths, "VideoRecorder was not instantiated"
        assert Path(created_paths[0]).exists()
