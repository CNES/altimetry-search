# altimetry-search

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

from altimetry.search import Mission, get_selected_passes, get_pass_passage_time

# Search all passes starting within one cycle of a given date
selected_passes = get_selected_passes(
    Mission.SWOT_SWATH_SCIENCE,
    date=numpy.datetime64("2024-01-01"),
)

# Restrict passes to those crossing a given area, and get the passage time
# window for each of them
passage_time = get_pass_passage_time(
    Mission.SWOT_SWATH_SCIENCE,
    selected_passes,
    polygon=None,  # or a pyinterp.geometry.geographic.Polygon
)
```

## Looking for Swot passes using the GUI

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?labpath=main.ipynb)

To launch the application, click on the link below:

* https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?urlpath=voila%2Frender%2Fmain.ipynb

To launch jupyterlab in binder, clink on the link below:

* https://mybinder.org/v2/gh/CNES/altimetry-search/HEAD?labpath=main.ipynb

## License

BSD-3-Clause — see [LICENSE](LICENSE).
