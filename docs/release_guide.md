# PySole Release & Distribution Guide

This document outlines the standard two-step procedure for releasing and distributing `PySole`.

---

## Step 1: Colleague Testing via GitHub Direct Installation

Before publishing to PyPI, colleagues and early testers can install `PySole` directly from GitHub using `pip`.

### 1. Direct `pip` Installation from GitHub

To install the latest state from the `main` branch:
```bash
pip install git+https://github.com/da0bi/pysole.git
```

To install a specific release version tag (e.g. `v0.4.3`):
```bash
pip install git+https://github.com/da0bi/pysole.git@v0.4.3
```

### 2. Editable Development Installation for Testers

If testers want to inspect the source code or run example scripts locally:
```bash
# Clone the repository
git clone https://github.com/da0bi/pysole.git
cd pysole

# Install in editable mode with optional dependencies
pip install -e ".[all]"
```

### 3. Verification Commands for Testers

Verify that the CLI entry point and package imports work properly:
```bash
# Check installed PySole version
pysole --version

# Create a template pysole.json configuration file
pysole --init

# Run the included Weighted Universal Kriging (WUK) example
python examples/wuk/run_wuk_example.py
```

---

## Step 2: Official Release on PyPI (pypi.org)

Publishing `PySole` to PyPI makes it installable globally via `pip install pysole`.

### 1. Prerequisites

Ensure `build` and `twine` are installed in your build environment:
```bash
pip install --upgrade build twine
```

### 2. Version Verification

Ensure the version string is synchronized across all three location files:
- `pyproject.toml` (`version = "0.4.3"`)
- `src/pysole/__init__.py` (`__version__ = "0.4.3"`)
- `CHANGELOG.md` (`## [0.4.3] - 2026-10-08`)

### 3. Git Release Tagging

Tag the release commit in your local git repository and push it to GitHub:
```bash
git add .
git commit -m "Release v0.4.3"
git tag -a v0.4.3 -m "PySole Release v0.4.3"
git push origin main --tags
```

### 4. Build Distribution Packages

Clean existing build artifacts and build the Source Distribution (`sdist`) and Binary Wheel (`bdist_wheel`):
```bash
# Remove previous build outputs
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Build sdist and wheel
python -m build
```

Verify that the `dist/` directory contains both `.tar.gz` and `.whl` files:
- `dist/pysole-0.4.3.tar.gz`
- `dist/pysole-0.4.3-py3-none-any.whl`

### 5. Validate Package Metadata with Twine

Run `twine check` to ensure `README.md` markdown renders without syntax errors and metadata compliance is met:
```bash
twine check dist/*
```

Expected output:
```text
Checking dist/pysole-0.4.3-py3-none-any.whl: PASSED
Checking dist/pysole-0.4.3.tar.gz: PASSED
```

### 6. Upload to TestPyPI (Optional Dry-Run)

Before uploading to production PyPI, test the package upload on TestPyPI:
```bash
twine upload --repository testpypi dist/*
```

Test installing from TestPyPI in a clean virtual environment:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pysole
```

### 7. Upload to Official PyPI (pypi.org)

Upload the validated distribution files to production PyPI:
```bash
twine upload dist/*
```

Enter your PyPI API token (`pypi-...`) when prompted for credentials.

Once uploaded, `PySole` is publicly available for anyone to install via:
```bash
pip install pysole
```
