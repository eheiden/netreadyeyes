import importlib
import unittest


PURE_MODULES = [
    "netrunner_scanner.config",
    "netrunner_scanner.runtime_controls",
    "netrunner_scanner.roi",
    "netrunner_scanner.tracking",
    "netrunner_scanner.obs_queue",
    "netrunner_scanner.obs_output",
    "netrunner_scanner.stability",
    "netrunner_scanner.perf",
]


class ImportSmokeTests(unittest.TestCase):
    def test_core_modules_import(self):
        for module_name in PURE_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
