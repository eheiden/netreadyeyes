import unittest
from unittest import mock

from netrunner_scanner import camera


class CameraFormattingTests(unittest.TestCase):
    def test_format_camera_info_full_metadata(self):
        self.assertEqual(
            camera.format_camera_info({"width": 1920, "height": 1080, "fps": 25}),
            "1920x1080 @ 25.0 fps",
        )

    def test_format_camera_info_partial_metadata(self):
        self.assertEqual(camera.format_camera_info({"width": 1280, "height": 720, "fps": 0}), "1280x720")
        self.assertEqual(camera.format_camera_info({"width": 0, "height": 0, "fps": 59.94}), "59.9 fps")

    def test_format_camera_info_empty_metadata_is_blank(self):
        self.assertEqual(camera.format_camera_info({"width": 0, "height": 0, "fps": 0}), "")
        self.assertEqual(camera.format_camera_info({}), "")

    def test_list_video_sources_does_not_show_unknown_resolution_when_unprobed(self):
        with mock.patch.object(camera, "list_cameras", return_value=["OBS Virtual Camera", "Cam Link 4K"]):
            sources = camera.list_video_sources(probe=False)
        labels = [source["label"] for source in sources]
        self.assertEqual(labels[0], "0: OBS Virtual Camera")
        self.assertEqual(labels[1], "1: Cam Link 4K")
        self.assertFalse(any("resolution unknown" in label.lower() for label in labels))


if __name__ == "__main__":
    unittest.main()
