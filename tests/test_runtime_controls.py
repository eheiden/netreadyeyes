import unittest

from netrunner_scanner import config
from netrunner_scanner import runtime_controls


class RuntimeControlsTests(unittest.TestCase):
    def setUp(self):
        runtime_controls.reset_threshold_values()

    def test_queue_wait_is_clamped(self):
        runtime_controls.set_queue_wait_seconds(-100)
        self.assertEqual(runtime_controls.get_queue_wait_seconds(), float(config.CONTROL_QUEUE_SECONDS_MIN))
        runtime_controls.set_queue_wait_seconds(999)
        self.assertEqual(runtime_controls.get_queue_wait_seconds(), float(config.CONTROL_QUEUE_SECONDS_MAX))

    def test_invalid_mode_is_rejected(self):
        before = runtime_controls.get_settings()["mode"]
        self.assertFalse(runtime_controls.set_mode("banana"))
        self.assertEqual(runtime_controls.get_settings()["mode"], before)

    def test_threshold_setting_updates_config_constant(self):
        self.assertTrue(runtime_controls.set_threshold_value("track_center_match_threshold_px", 123))
        self.assertEqual(runtime_controls.get_threshold_values()["track_center_match_threshold_px"], 123.0)
        self.assertEqual(config.TRACK_CENTER_MATCH_THRESHOLD_PX, 123.0)

    def test_unknown_threshold_is_rejected(self):
        self.assertFalse(runtime_controls.set_threshold_value("not_a_real_threshold", 1))

    def test_threshold_definitions_have_required_fields(self):
        required = {"key", "label", "constant", "type", "group", "help"}
        definitions = runtime_controls.threshold_definitions()
        self.assertGreater(len(definitions), 0)
        keys = set()
        for definition in definitions:
            self.assertTrue(required.issubset(definition.keys()), definition)
            self.assertNotIn(definition["key"], keys, f"duplicate threshold key: {definition['key']}")
            keys.add(definition["key"])
            self.assertTrue(hasattr(config, definition["constant"]), definition)


if __name__ == "__main__":
    unittest.main()
