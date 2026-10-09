"""
Guards the quick-start notebook against API drift.

* Always: every code cell compiles, and every ``pysole.<func>(...)`` / ``Solver.<method>(...)`` keyword
  argument used in the notebook exists in the real signature.
* Opt-in (``PYSOLE_TEST_NOTEBOOK=1`` and ``nbclient`` + ``ipykernel`` installed): the notebook is executed
  end-to-end in a temporary copy of ``examples/``.
"""

import ast
import inspect
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pysole
from pysole import Solver

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOK = EXAMPLES / "pysole_quickstart.ipynb"


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]


def _strip_magics(src: str) -> str:
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!")))


class TestQuickstartNotebook(unittest.TestCase):
    def test_code_cells_compile(self):
        cells = _code_cells()
        self.assertGreater(len(cells), 0)
        for i, src in enumerate(cells):
            compile(_strip_magics(src), f"<cell {i}>", "exec")

    def test_api_calls_match_signatures(self):
        """Keyword arguments of pysole.* and <solver>.* calls must exist in the real signatures."""
        checked = 0
        for src in _code_cells():
            tree = ast.parse(_strip_magics(src))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                owner, name = node.func.value, node.func.attr
                target = None
                if isinstance(owner, ast.Name) and owner.id == "pysole":
                    target = getattr(pysole, name, None)
                    self.assertIsNotNone(target, f"pysole.{name} does not exist")
                elif isinstance(owner, ast.Name) and owner.id == "Solver":
                    target = getattr(Solver, name, None)
                    self.assertIsNotNone(target, f"Solver.{name} does not exist")
                elif isinstance(owner, ast.Name) and owner.id == "solver":
                    target = getattr(Solver, name, None)
                    self.assertIsNotNone(target, f"Solver.{name}() used in the notebook does not exist")
                if target is None or not callable(target):
                    continue
                params = inspect.signature(target).parameters
                has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
                for kw in node.keywords:
                    if kw.arg is not None and not has_var_kw:
                        self.assertIn(kw.arg, params, f"{name}() has no parameter '{kw.arg}'")
                checked += 1
        self.assertGreaterEqual(checked, 5)

    @unittest.skipUnless(os.environ.get("PYSOLE_TEST_NOTEBOOK") == "1", "set PYSOLE_TEST_NOTEBOOK=1 to execute the notebook")
    def test_notebook_executes(self):
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:  # pragma: no cover
            self.skipTest("nbclient/nbformat/ipykernel not installed (pip install -e .[dev])")

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "examples"
            work.mkdir()
            shutil.copytree(EXAMPLES / "wuk", work / "wuk")
            shutil.copy(NOTEBOOK, work / NOTEBOOK.name)
            nb = nbformat.read(work / NOTEBOOK.name, as_version=4)
            client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(work)}})
            client.execute()
            self.assertTrue((work / "wuk_bedrock_tutorial.tif").exists())


if __name__ == "__main__":
    unittest.main()
