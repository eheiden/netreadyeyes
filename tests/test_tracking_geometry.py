import unittest

import numpy as np

from netrunner_scanner import tracking


class TrackingGeometryTests(unittest.TestCase):
    def test_rect_iou_identical(self):
        self.assertEqual(tracking.rect_iou((10, 10, 50, 70), (10, 10, 50, 70)), 1.0)

    def test_rect_iou_disjoint(self):
        self.assertEqual(tracking.rect_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)

    def test_center_distance(self):
        self.assertAlmostEqual(tracking.center_distance((0, 0, 10, 10), (30, 40, 10, 10)), 50.0)

    def test_smooth_box_handles_none(self):
        box = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.intp)
        self.assertIs(tracking.smooth_box(None, box), box)
        self.assertIs(tracking.smooth_box(box, None), box)

    def test_smooth_box_blends_points(self):
        old = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.intp)
        new = np.array([[10, 10], [20, 10], [20, 30], [10, 30]], dtype=np.intp)
        blended = tracking.smooth_box(old, new, alpha=0.5)
        np.testing.assert_array_equal(blended, np.array([[5, 5], [15, 5], [15, 25], [5, 25]], dtype=np.intp))


if __name__ == "__main__":
    unittest.main()
