"""
Unit Tests for PySole Logging Module.
"""

import os
import unittest
import logging
from pysole.logging import setup_logging, logger
from pysole.config import load_config, DEFAULT_CONFIG, create_template_config


class TestLogging(unittest.TestCase):
    """Test suite for PySole logging module and configuration integration."""

    def setUp(self):
        self.test_log_file = "test_pysole_unit.log"
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    def tearDown(self):
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    def test_setup_logging_file_creation(self):
        """Tests that setup_logging creates the specified log file and writes formatted logs."""
        setup_logging(log_file=self.test_log_file, log_level="INFO", console_output=False)
        logger.info("Test Info Message")
        logger.warning("Test Warning Message")

        self.assertTrue(os.path.exists(self.test_log_file))

        with open(self.test_log_file, "r") as f:
            content = f.read()

        self.assertIn("[INFO] Test Info Message", content)
        self.assertIn("[WARNING] Test Warning Message", content)

    def test_logging_level_filtering(self):
        """Tests that messages below configured log_level are filtered out."""
        setup_logging(log_file=self.test_log_file, log_level="WARNING", console_output=False)
        logger.debug("Debug Message (Should be filtered)")
        logger.info("Info Message (Should be filtered)")
        logger.warning("Warning Message (Should be logged)")

        with open(self.test_log_file, "r") as f:
            content = f.read()

        self.assertNotIn("Debug Message", content)
        self.assertNotIn("Info Message", content)
        self.assertIn("[WARNING] Warning Message (Should be logged)", content)

    def test_default_config_includes_log_level(self):
        """Tests that DEFAULT_CONFIG includes log_level under inputs section."""
        self.assertIn("log_level", DEFAULT_CONFIG["inputs"])
        self.assertEqual(DEFAULT_CONFIG["inputs"]["log_level"], "INFO")


if __name__ == "__main__":
    unittest.main()
