"""
PySole test suite initialization.
Ensures thread-safe headless Matplotlib Agg backend during test discovery and execution.
"""

import matplotlib
matplotlib.use("Agg")
