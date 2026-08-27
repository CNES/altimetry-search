# Changelog

All notable changes to this project will be documented in this file.

---

## [2.0.0] - 2026-08-27

### Added
- New `sad` module: fetches SWOT orbit files (`SWOT_calval_orbit.nc`, `SWOT_science_orbit.nc`) from a remote server on first access instead of shipping them inside the package.
- `nb_pass` on MissionProperties: fixed number of passes per cycle for each mission phase (584 for Science, 28 for CalVal), the same set every cycle.

### Changed
Breaking: MissionProperties's orbit_file constructor parameter is renamed to orbit_key

### Removed
Breaking: the .nc orbit files are no longer shipped inside the package

---

## [1.1.0] - 2026-08-21 [YANKED]

Broken: orbit `.nc` files are corrupted Git LFS pointers, not valid NetCDF.

### Changed
- DOCS: add changelog
- CHORE: exposed `MissionType`, `MissionProperties`, `MissionPropertiesLoader`.

---

## [1.0.2] - 2026-08-18 [YANKED]

Broken: same root cause as above.

### Changed
- CI: add missing dependencies
- DOCS: readme update

---

## [1.0.1] - 2026-08-17 [YANKED]

Broken: same root cause as above.

### Changed
- CI fix

---

## [1.0.0] - 2026-08-17 [YANKED]

Broken: same root cause as above.

### Changed
- the GUI related part is isolated under a `altimetry.search.gui` module
- the scripts used to generate orbit files are pulled up out of the python module
- the pypi/conda package contains the Python API but excludes the GUI part
