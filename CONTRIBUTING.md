# Contributing to PySole

Thank you for your interest in contributing to **PySole**! We welcome contributions from glaciologists, geophysicists, software developers, and researchers of all skill levels.

---

## Code of Conduct

We are committed to providing a welcoming, respectful, and collaborative environment for everyone. Please maintain professionalism, constructive feedback, and mutual respect in all issues, pull requests, and community discussions.

---

## How to Contribute

### 1. Reporting Bugs & Issues
If you encounter a bug, unexpected behavior, or inaccurate calculations:
1. Search existing [GitHub Issues](https://github.com/da0bi/pysole/issues) to verify if the issue has already been reported.
2. If not, open a **New Issue** with a clear title and description.
3. Include:
   - Your Python version and OS platform.
   - Minimal reproducible example or configuration snippet.
   - The full traceback error log.

### 2. Suggesting Features
We welcome proposals for new Universal Kriging drift models, ray migration algorithms, spatial raster formats, or visualization features.
- Open an Issue with tag `feature request`.
- Explain the glaciological or geophysical rationale, the proposed API design, and expected behavior.

### 3. Submitting Pull Requests (PRs)

#### Step 1: Fork and Clone
```bash
git clone https://github.com/da0bi/pysole.git
cd pysole
```

#### Step 2: Set Up Virtual Environment
Create a virtual environment and install development dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

#### Step 3: Create a Feature Branch
```bash
git checkout -b feature/my-new-drift-model
```

#### Step 4: Write Code and Tests
- Ensure all new features or bug fixes include corresponding unit tests in the `tests/` directory.
- Follow Python 3.10+ conventions, including explicit type hints and standard docstrings.
- Enforce canonical drift model names (`"sia"`, `"z_dem"`, `"curvature_dem"`, `"linear_xy"`, `"quadratic_xy"`).

#### Step 5: Run the Unit Test Suite
Before opening a Pull Request, run the full test suite to verify that all tests pass cleanly:
```bash
python -m unittest discover tests
```

#### Step 6: Commit and Push
```bash
git add .
git commit -m "feat(interpolation): add new geomorphological drift model"
git push origin feature/my-new-drift-model
```

#### Step 7: Open a Pull Request
Go to the repository on GitHub and click **Compare & pull request**. Provide a clear summary of your changes, referencing any relevant issues.

---

## Development & Code Style Guidelines

- **Style**: Follow PEP 8 guidelines for formatting.
- **Type Hints**: Use standard type annotations (`np.ndarray`, `float`, `dict[str, Any]`).
- **Documentation**: Keep `README.md`, docstrings, and parameter descriptions up to date.
- **No Backward Compatibility Breaking**: Always verify that changes pass all test cases in `tests/test_*.py`.

Thank you for helping make PySole better for the glaciology and geophysics community!
