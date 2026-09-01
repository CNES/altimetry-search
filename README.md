# Altimetry Search

Search and select satellite altimetry passes (SWOT science and cal/val
phases) by date and geographic area.

This package exposes the **API only** (orbit/pass search). The interactive
map-based GUI (`ipyleaflet`/`ipywidgets`) lives in the separate
`altimetry.search.gui` module and is **not** installed with this package —
see [Looking for Swot passes using the GUI](#Looking-for-Swot-passes-using-the-GUI)
below if you need it.

## Installation

**pip**
```bash
pip install altimetry-search
```

**conda**
```bash
conda install -c conda-forge altimetry-search
```

## Quickstart

```python
import numpy
import pyinterp.geometry.geographic as py_geo
from altimetry.search import Mission, get_selected_passes, get_pass_passage_time

# Search all passes starting within one cycle of a given date
selected_passes = get_selected_passes(
    Mission.SWOT_SWATH_SCIENCE,
    date=numpy.datetime64("2024-01-01"),
)

# Restrict passes to those crossing a given area, and get the passage time
# window for each of them
bbox = py_geo.algorithms.from_wkt(
    'POLYGON((-6 36,-6 60,36 60,36 36,-6 36))')

passage_time = get_pass_passage_time(
    Mission.SWOT_SWATH_SCIENCE,
    selected_passes,
    polygon=bbox
)
```

Available missions are :
* `SWOT_SWATH_SCIENCE`
* `SWOT_NADIR_SCIENCE`
* `SWOT_SWATH_CALVAL`
* `SWOT_NADIR_CALVAL`

## Looking for Swot passes using the GUI

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?labpath=main.ipynb)

To launch the application, click on the link below:

* https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?urlpath=voila%2Frender%2Fmain.ipynb

To launch jupyterlab in binder, clink on the link below:

* https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?labpath=main.ipynb

## Changelog

See the [CHANGELOG](CHANGELOG.md) for versions details.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
