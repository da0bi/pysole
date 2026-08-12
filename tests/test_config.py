"""
Unit tests for pysole.json configuration parsing and run_from_config workflow using Wurtenkees Glacier dataset.
"""

import unittest
import tempfile
import pathlib
import os
import numpy as np
import pysole


class TestConfigWorkflowWuk(unittest.TestCase):
    def test_run_from_config_wuk(self):
        wuk_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/pysole_wuk.json"))

        # Execute complete workflow from Wurtenkees Glacier config
        solver = pysole.Solver.from_config(wuk_config)
        self.assertEqual(solver.dem_grid.shape, (179, 213))
        self.assertEqual(solver.dx, 5.0)
        self.assertEqual(solver.dy, 5.0)

        bedrock_map = pysole.run_from_config(wuk_config)
        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        self.assertEqual(bedrock_map.shape, (179, 213))
        self.assertTrue(np.all(np.isfinite(bedrock_map.grid)))

    def test_save_multi_format_all(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            grid = np.ones((10, 10)) * 500.0
            bedrock_map = pysole.BedrockMap(grid=grid, bounds=(0.0, 0.0, 50.0, 50.0))

            base_file = str(tmp_path / "test_export.tif")
            saved = bedrock_map.save(base_file, formats="all")

            self.assertIsInstance(saved, list)
            self.assertEqual(len(saved), 4)

            expected_exts = [".tif", ".asc", ".csv", ".npy"]
            for ext in expected_exts:
                expected_file = str(tmp_path / f"test_export{ext}")
                self.assertTrue(os.path.exists(expected_file), f"Missing exported file: {expected_file}")


if __name__ == "__main__":
    unittest.main()
