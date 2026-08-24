# Changelog

All notable changes to this project will be documented in this file.

---

## [1.1.1] - 2026-08-24

### Fixed

Orbit resource files (`SWOT_calval_orbit.nc`, `SWOT_science_orbit.nc`) were packaged as raw Git LFS pointer files instead of their actual binary content in the PyPI/conda-forge distributions, causing OSError: [Errno -51] NetCDF: Unknown file format at runtime. actions/checkout now fetches LFS content (lfs: true) before building the package.
All previous releases (1.0.0 through 1.1.0) are affected and should not be used.

---

## [1.1.0] - 2026-08-21

### Changed
- DOCS: add changelog
- CHORE: expose mission related objects

---

## [1.0.2] - 2026-08-18

### Changed
- CI: add missing dependencies
- DOCS: readme update

---

## [1.0.1] - 2026-08-17

### Changed
- CI fix

---

## [1.0.0] - 2026-08-17

### Changed
- the GUI related part is isolated under a `altimetry.search.gui` module
- the scripts used to generate orbit files are pulled up out of the python module
- the pypi/conda package contains the Python API but excludes the GUI part
