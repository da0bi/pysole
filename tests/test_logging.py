"""
Unit Tests for PySole Logging Module.
"""

import inspect
import os
import textwrap
import threading
import time
import unittest
import logging
from unittest.mock import patch

import pysole.logging as pysole_logging
from pysole.logging import _format_clock, activity_indicator, setup_logging, logger
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


class RecordingBar:
    """Fake tqdm bar that records the description lines."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.lines = [kwargs.get("desc", "")]
        self.closed = False
        RecordingBar.instances.append(self)

    def set_description_str(self, desc=None, refresh=True):
        self.lines.append(desc)

    def close(self):
        self.closed = True


class TestActivityIndicator(unittest.TestCase):
    """The elapsed-time spinner only reports; it must appear late, stop cleanly and never leak its thread."""

    def setUp(self):
        RecordingBar.instances = []

    @staticmethod
    def _spinner_threads():
        return [t for t in threading.enumerate() if t.name == "pysole-activity-indicator"]

    def test_disabled_starts_no_thread_and_shows_nothing(self):
        with patch("pysole.logging.tqdm", RecordingBar):
            with activity_indicator("work", disable=True, delay=0.0):
                self.assertEqual(self._spinner_threads(), [])
        self.assertEqual(RecordingBar.instances, [])

    def test_fast_body_shows_nothing_and_stops_the_thread(self):
        with patch("pysole.logging.tqdm", RecordingBar):
            with activity_indicator("work", delay=5.0):
                pass
        self.assertEqual(RecordingBar.instances, [])
        self.assertEqual(self._spinner_threads(), [])

    def test_slow_body_shows_elapsed_time_and_closes_the_bar(self):
        with patch("pysole.logging.tqdm", RecordingBar):
            with activity_indicator("work", delay=0.02, interval=0.02):
                time.sleep(0.2)
        self.assertEqual(len(RecordingBar.instances), 1)
        bar = RecordingBar.instances[0]
        self.assertTrue(bar.closed)
        self.assertTrue(all(line.startswith("work [00:0") for line in bar.lines))
        self.assertGreaterEqual(len(bar.lines), 3)
        self.assertEqual(self._spinner_threads(), [])

    def test_exception_in_body_stops_the_thread_and_propagates(self):
        with patch("pysole.logging.tqdm", RecordingBar):
            with self.assertRaises(RuntimeError):
                with activity_indicator("work", delay=0.0, interval=0.01):
                    time.sleep(0.05)
                    raise RuntimeError("boom")
        self.assertTrue(RecordingBar.instances[0].closed)
        self.assertEqual(self._spinner_threads(), [])

    def test_clock_format(self):
        self.assertEqual(_format_clock(5.9), "00:05")
        self.assertEqual(_format_clock(125), "02:05")
        self.assertEqual(_format_clock(3725), "1:02:05")

    def test_works_with_the_fallback_stub(self):
        """Without tqdm the stub must accept the spinner's calls (the stub is the module-level fallback)."""
        namespace = {}
        source = inspect.getsource(pysole_logging).split("try:\n    from tqdm import tqdm\nexcept ImportError:\n")[1]
        stub_source = textwrap.dedent(source.split("\n\n# Global PySole logger")[0])
        exec(stub_source, namespace)
        bar = namespace["tqdm"](total=None, desc="x", bar_format="{desc}", leave=True)
        with bar:
            bar.update(3)
            bar.refresh()
            bar.set_description_str("y")
        bar.close()
        self.assertEqual(list(namespace["tqdm"]([1, 2, 3], total=3, desc="x", unit="u", ascii=" #", disable=True)), [1, 2, 3])


class TestTerminalUI(unittest.TestCase):
    """Test suite for Terminal UI enhancements: banner centering, section headers, and progress bar spacing."""

    def test_welcome_banner_content(self):
        banner = pysole_logging.get_welcome_banner(version="0.5.2", term_width=80)
        self.assertIn("/ ___/____  / /__", banner)
        self.assertIn("The soul of the sole", banner)
        self.assertIn("v0.5.2", banner)
        self.assertIn("Released under the MIT License.", banner)

    def test_welcome_banner_dynamic_version(self):
        banner_auto = pysole_logging.get_welcome_banner(term_width=80)
        from pysole import __version__
        self.assertIn(f"v{__version__}", banner_auto)

    def test_welcome_banner_centering(self):
        block_width = 33
        for width in [80, 100, 120]:
            banner = pysole_logging.get_welcome_banner(version="0.5.1", term_width=width)
            lines = [l for l in banner.splitlines() if l.strip()]
            tagline = [l for l in lines if "The soul of the sole" in l][0]
            left_space = len(tagline) - len(tagline.lstrip())
            art_margin = (width - block_width) // 2
            expected_left = art_margin + (block_width - len(tagline.strip())) // 2
            self.assertEqual(left_space, expected_left)

    def test_print_welcome_banner(self):
        import io
        buf = io.StringIO()
        pysole_logging.print_welcome_banner(version="0.5.1", file=buf)
        out = buf.getvalue()
        self.assertIn("The soul of the sole", out)
        self.assertIn("Released under the MIT License.", out)

    def test_progress_bar_spacing_enabled(self):
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            pbar = pysole_logging.PySoleProgressBar(range(2), disable=False, file=buf)
            for _ in pbar:
                pass
        finally:
            sys.stdout = old_stdout
        out = buf.getvalue()
        self.assertTrue(out.startswith("\n"))
        self.assertTrue(out.endswith("\n"))

    def test_progress_bar_spacing_disabled(self):
        import io
        buf = io.StringIO()
        pbar = pysole_logging.PySoleProgressBar(range(2), disable=True, file=buf)
        for _ in pbar:
            pass
        self.assertEqual(buf.getvalue(), "")

    def test_log_section_header(self):
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            pysole_logging.log_section_header("TEST HEADLINE TITLE", width=80)
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        self.assertIn("=" * 80, out)
        self.assertIn("TEST HEADLINE TITLE", out)
        # Verify it has top and bottom dividers and centered title
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "=" * 80)
        self.assertEqual(lines[1], "TEST HEADLINE TITLE")
        self.assertEqual(lines[2], "=" * 80)


class TestPipelineLogging(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.addCleanup(self.temp_dir.cleanup)

    def test_cli_log_level_overrides_config(self):
        """run_from_config(log_level='DEBUG') beats the config-file log_level (and the control case does not)."""
        import json
        from pathlib import Path
        import numpy as np
        from pysole.raster import BedrockMap
        from pysole.pipeline import run_from_config

        out_path = Path(self.output_dir)
        dem = np.full((20, 20), 2000.0)
        dem_path = out_path / "dem_log.tif"
        BedrockMap(dem, bounds=(0, 0, 200, 200), crs="EPSG:32633").save(dem_path)
        pts_path = out_path / "pts_log.csv"
        pts = np.array([[50.0, 50.0, 2000.0, 30.0], [100.0, 100.0, 1950.0, 40.0], [150.0, 150.0, 1900.0, 25.0]])
        np.savetxt(pts_path, pts, delimiter=",", header="X,Y,Z_surf,depth", comments="")
        cfg = {
            "inputs": {
                "dem_path": str(dem_path), "survey_data_path": str(pts_path), "survey_data_type": "depth",
                "show_progress": False, "log_level": "WARNING",
            },
            "migration_parameters": {"perform_migration": False},
            "outputs": {"output_dir": str(out_path / "log_out"), "output_format": "tif"},
        }
        cfg_path = out_path / "cfg_log.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        prev_level = logger.level
        self.addCleanup(logger.setLevel, prev_level)
        env = {k: v for k, v in os.environ.items() if k != "PYSOLE_LOG_LEVEL"}
        with patch.dict(os.environ, env, clear=True):
            run_from_config(cfg_path, is_batch=True)
            self.assertEqual(logging.getLogger("pysole").level, logging.WARNING)  # control: config value wins
            run_from_config(cfg_path, is_batch=True, log_level="DEBUG")
            self.assertEqual(logging.getLogger("pysole").level, logging.DEBUG)  # CLI override wins

    def test_single_log_initialization_message(self):
        """Verifies that 'Initialized PySole log file at:' is logged exactly once, after the banner."""
        import json
        from pathlib import Path
        import numpy as np
        from pysole.raster import BedrockMap
        from pysole.pipeline import run_from_config
        from pysole.logging import reset_logging_state

        reset_logging_state()
        out_path = Path(self.output_dir)
        cfg_path = out_path / "pysole_test_log.json"
        dem_path = out_path / "dem.tif"
        survey_path = out_path / "survey.csv"
        out_dir = out_path / "out"

        dem = np.full((20, 20), 1000.0)
        BedrockMap(dem, bounds=(0, 0, 200, 200), crs="EPSG:32633").save(dem_path)
        np.savetxt(survey_path, np.array([[50.0, 50.0, 1000.0, 100.0], [150.0, 150.0, 1000.0, 120.0]]), delimiter=",", header="X,Y,Z,T", comments="")

        cfg = {
            "inputs": {
                "dem_path": str(dem_path),
                "survey_data_path": str(survey_path),
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "outputs": {
                "output_dir": str(out_dir),
            },
            "spatial_parameters": {"dx": 10.0, "dy": 10.0},
            "kriging_parameters": {
                "post_migration": {"method": "ordinary"}
            },
            "optimization_parameters": {"interactive_optimization": False},
            "migration_parameters": {"perform_migration": False},
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        run_from_config(cfg_path, is_batch=True)

        log_file = out_dir / "pysole.log"
        self.assertTrue(log_file.exists())
        with open(log_file, "r") as f:
            log_lines = f.readlines()

        init_lines = [l for l in log_lines if "Initialized PySole log file at:" in l]
        self.assertEqual(len(init_lines), 1, f"Expected exactly 1 log init line in file, got: {init_lines}")

        banner_idx = -1
        init_idx = -1
        for i, l in enumerate(log_lines):
            if "STARTING NEW PYSOLE BEDROCK TOPOGRAPHY CALCULATION" in l:
                banner_idx = i
            elif "Initialized PySole log file at:" in l:
                init_idx = i
        self.assertGreater(banner_idx, -1)
        self.assertGreater(init_idx, banner_idx)


if __name__ == "__main__":
    unittest.main()
