import unittest

from netrunner_scanner import roi


class RoiTests(unittest.TestCase):
    def test_make_rect_roi_uses_four_ordered_points(self):
        result = roi.make_rect_roi(10, 20, 100, 200, enabled=False, color=(1, 2, 3))
        self.assertEqual(result["type"], "quad")
        self.assertEqual(result["points"], [[10, 20], [110, 20], [110, 220], [10, 220]])
        self.assertFalse(result["enabled"])
        self.assertEqual(result["color"], [1, 2, 3])

    def test_normalize_legacy_rect_roi(self):
        result = roi.normalize_roi([5, 6, 70, 80], frame_width=200, frame_height=200)
        self.assertEqual(result["points"], [[5, 6], [75, 6], [75, 86], [5, 86]])
        self.assertTrue(roi.roi_enabled(result))

    def test_clamps_points_to_frame(self):
        result = roi.normalize_roi(
            {"points": [[-10, -5], [300, 0], [300, 250], [0, 250]], "enabled": True},
            frame_width=100,
            frame_height=80,
        )
        self.assertEqual(result["points"], [[0, 0], [99, 0], [99, 79], [0, 79]])

    def test_roi_label_toggle_round_trip(self):
        original = roi.show_roi_labels()
        try:
            self.assertFalse(roi.set_show_roi_labels(False))
            self.assertFalse(roi.show_roi_labels())
            self.assertTrue(roi.set_show_roi_labels(True))
            self.assertTrue(roi.show_roi_labels())
        finally:
            roi.set_show_roi_labels(original)


if __name__ == "__main__":
    unittest.main()
