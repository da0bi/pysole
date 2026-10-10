# PySole v0.4.4 — Initial Official Release

This is the **first official release of PySole (v0.4.4)**!

**PySole** is a specialized Python package designed to reconstruct thickness distributions and 3D bedrock topography (sole) for viscous flow phenomena from sparse geophysical datasets. It adapts the **Shallow Ice Approximation (SIA)** to transform limited survey points into robust, physically constrained 3D bedrock models.

---

## 🌟 Key Features

- ⚙️ **JSON Configuration & Command-Line CLI**

- 📂 **5 Supported Digital Elevation Model Formats & Strict Coordinate Reference System Verification of all Input Data**

- 📊 **Automated High-Resolution Diagnostic Plots** for every major processing milestone.

- 💡 **Objective DEM Surface Slope Optimization** by enforcing minimum spatial variance in basal shear stress (Binder et al., 2009).

- 📐 **3D Ray-Based Eikonal Migration** engineered specifically to process geophysical signal traveltimes with sparse spatial coverage to better image steep slopes and bed overdeepenings.

- 🌍 **Physically Constrained Kriging Interpolation** offering a suite of Universal Kriging drift models.

- 🔍 **Interactive Drift Analyzer** provides an automated ranking of the optimum Universal Kriging drift model for the given data set.

- ⚡ **High-Performance Dual Kriging Vector Engine** offering **~180x speedups** over point-by-point solvers.

- 🗺️ **Survey Planner** for unprobed objects - automatically generates survey campaign tracks, derived from a modelled thickness distribution and exported as field-ready GPX and GeoJSON vector files.

- 🏞️ **Outcrop Boundaries, ML Gap Filling & Margin Tapering** to assure a continuous bedrock smoothly embedded in the surface DEM.

> 📖 **Methodological & Mathematical Details**:
> For detailed equations, theoretical derivations, parameter references, and step-by-step guides, please consult the [README.md](README.md) and the [`Documentation Manual`](docs/drift_analyzer_&_survey_planner.md).

---

## 📦 Installation (GitHub)

*(PyPI installation is not available yet. Please install directly from GitHub:)*

### Direct Installation from GitHub (Recommended)

You can install `PySole` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/da0bi/pysole.git
```

### Local / Development Installation

To clone and install `PySole` directly from source in editable mode:

```bash
git clone https://github.com/da0bi/pysole.git
cd pysole
pip install -e .
```
