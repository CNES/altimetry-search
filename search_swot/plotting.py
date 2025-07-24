import dataclasses
import pathlib as pl

import ipyleaflet as ipy_l
import ipywidgets as ipy_w
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyinterp.geodetic as py_geod
import xarray as xr

from .models import Mission, MissionType

# List of HTML colors
COLORS: list[str] = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beiae',
    'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown',
    'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
    'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
    'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
    'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink',
    'indianred ', 'indigo ', 'ivory', 'khaki', 'lavender', 'lavenderblush',
    'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
    'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow',
    'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
    'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin',
    'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
    'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue',
    'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
    'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
    'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
    'transparent', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen'
]

#: HTML Template for the popup of the marker
POPUP_TEMPLATE = """
<div style="text-align: center; font-weight: bold;">
    <div style="display: inline-block; width: 10px; height: 10px;
    border: 1px solid black; background-color: {color};
    margin-right: 5px;"></div>
    Pass {pass_number}
</div>
"""


@dataclasses.dataclass
class HalfOrbitFootprint:
    #: Marker of the half orbit footprint to display the pass number
    marker: ipy_l.Marker


@dataclasses.dataclass
class SwathFootprint(HalfOrbitFootprint):
    """Swath footprint definition."""
    #: Left polygon of the swath footprint
    left: ipy_l.Polygon
    #: Right polygon of the swath footprint
    right: ipy_l.Polygon


@dataclasses.dataclass
class NadirFootprint(HalfOrbitFootprint):
    """Nadir footprint definition."""
    #: Polygon of the nadir footprint
    line: ipy_l.Polyline


#: Type of a pass polygon/line
PassPolygon = tuple[int, py_geod.Polygon]
PassLine = tuple[int, py_geod.LineString]


def plot_selected_passes(selected_area: py_geod.Polygon, mission: Mission,
                         east: float,
                         df: pd.DataFrame) -> list[SwathFootprint]:
    """Plot the selected passes.

    Args:
        selected_area: Selected area.
        mission: Mission.
        east: East longitude.
        df: Selected passes.

    Returns:
        The half_orbits plotted on the map.
    """
    bbox: py_geod.Polygon = (  # type: ignore[assignment]
        selected_area if selected_area is not None else
        py_geod.Box.whole_earth().as_polygon())

    if mission.mission_type == MissionType.SWATH:
        (left_swath, right_swath) = load_polygons(
            mission, df['pass_number'].values)  # type: ignore[arg-type]

        left_layers: dict[int, ipy_l.Polygon] = {}
        right_layers: dict[int, ipy_l.Polygon] = {}
        markers: dict[int, ipy_l.Marker] = {}

        tuple(
            map(lambda x: plot_swath(*x, bbox, left_layers, markers, east),
                left_swath))
        tuple(
            map(lambda x: plot_swath(*x, bbox, right_layers, markers, east),
                right_swath))
        return [
            SwathFootprint(left=left_layers.get(pass_number, ipy_l.Polygon()),
                           right=right_layers.get(pass_number,
                                                  ipy_l.Polygon()),
                           marker=marker)
            for pass_number, marker in markers.items()
        ]

    nadir = load_lines(mission,
                       df['pass_number'].values)  # type: ignore[arg-type]

    layers: dict[int, ipy_l.Polyline] = {}
    markers: dict[int, ipy_l.Marker] = {}
    tuple(map(lambda x: plot_line(*x, bbox, layers, markers, east), nadir))
    return [
        NadirFootprint(line=layers.get(pass_number, ipy_l.Polyline()),
                       marker=marker)
        for pass_number, marker in markers.items()
    ]


def _load_one_polygon(x: NDArray, y: NDArray) -> py_geod.Polygon:
    """Load a polygon from a set of coordinates.

    Args:
        x: X coordinates.
        y: Y coordinates.

    Returns:
        Polygon.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    return py_geod.Polygon([py_geod.Point(x, y) for x, y in zip(x, y)])


def load_polygons(
        mission: Mission,
        pass_number: NDArray) -> tuple[list[PassPolygon], list[PassPolygon]]:
    """Load the polygons of the selected passes.

    Args:
        mission: Mission selected
        pass_number: Pass numbers to load.

    Returns:
        Left and right polygons of the selected passes.
    """
    index = pass_number - 1

    left_polygon: list[PassPolygon] = []
    right_polygon: list[PassPolygon] = []

    orbit_file = pl.Path(__file__).parent / mission.orbit_file
    with xr.open_dataset(orbit_file.resolve(), decode_timedelta=True) as ds:
        for ix in index:
            left_polygon.append(
                (ix + 1,
                 _load_one_polygon(ds.left_polygon_lon[ix, :].values,
                                   ds.left_polygon_lat[ix, :].values)))
            right_polygon.append(
                (ix + 1,
                 _load_one_polygon(ds.right_polygon_lon[ix, :].values,
                                   ds.right_polygon_lat[ix, :].values)))
    return left_polygon, right_polygon


def _load_one_line(x: NDArray, y: NDArray) -> py_geod.LineString:
    """Load a line from a set of coordinates.

    Args:
        x: X coordinates.
        y: Y coordinates.

    Returns:
        LineString.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    return py_geod.LineString([py_geod.Point(x, y) for x, y in zip(x, y)])


def load_lines(mission: Mission, pass_number: NDArray) -> list[PassLine]:
    """Load the lines of the selected passes.

    Args:
        mission: Mission selected
        pass_number: Pass numbers to load.

    Returns:
        Lines of the selected passes.
    """
    index = pass_number - 1

    lines: list[PassLine] = []

    orbit_file = pl.Path(__file__).parent / mission.orbit_file
    with xr.open_dataset(orbit_file.resolve(), decode_timedelta=True) as ds:
        for ix in index:
            lines.append((ix + 1,
                          _load_one_line(ds.line_string_lon[ix, :].values,
                                         ds.line_string_lat[ix, :].values)))
    return lines


def plot_swath(
    pass_number: int,
    item: py_geod.Polygon,
    bbox: py_geod.Polygon,
    layers: dict[int, ipy_l.Polygon],
    markers: dict[int, ipy_l.Marker],
    east: float,
) -> None:
    """Plot a swath.

    Args:
        pass_number: Pass number.
        item: Polygon to plot.
        bbox: Bounding box of the selected area.
        layers: Layers of the map.
        markers: Markers of the map.
        east: East longitude.
    """
    intersection = item.intersection(bbox)
    if len(intersection) == 0:
        return
    outer = intersection[0].outer

    (lons, lats) = _get_lons_lats(outer, east)

    color_id = pass_number % len(COLORS)
    layers[pass_number] = ipy_l.Polygon(
        locations=[(y, x) for x, y in zip(lons, lats)],
        color=COLORS[color_id],
        fill_color=COLORS[color_id],
    )

    # Add a marker to display the pass number on the map if it does not already
    # exist.
    if pass_number not in markers:
        _set_markers(markers, lons, lats, pass_number, color_id)


def plot_line(
    pass_number: int,
    item: py_geod.LineString,
    bbox: py_geod.Polygon,
    layers: dict[int, ipy_l.Polyline],
    markers: dict[int, ipy_l.Marker],
    east: float,
) -> None:
    """Plot nadir as a line.

    Args:
        pass_number: Pass number.
        item: line to plot.
        bbox: Bounding box of the selected area.
        layers: Layers of the map.
        markers: Markers of the map.
        east: East longitude.
    """
    intersection = item.intersection(bbox)
    if len(intersection) == 0:
        return
    outer = intersection[0]

    (lons, lats) = _get_lons_lats(outer, east)

    color_id = pass_number % len(COLORS)
    layers[pass_number] = ipy_l.Polyline(locations=[
        (y, x) for x, y in zip(lons, lats)
    ],
                                         color=COLORS[color_id],
                                         fill=False)

    # Add a marker to display the pass number on the map if it does not already
    # exist.
    if pass_number not in markers:
        _set_markers(markers, lons, lats, pass_number, color_id)


def _get_lons_lats(outer: list[py_geod.Point], east: float):
    lons = np.array([p.lon for p in outer])
    lats = np.array([p.lat for p in outer])
    lons = np.deg2rad(
        py_geod.normalize_longitudes(np.array([p.lon for p in outer]), east))
    lons = np.unwrap(lons, discont=np.pi)
    lons = np.rad2deg(lons)

    return lons, lats


def _set_markers(markers: dict[int, ipy_l.Marker], lons: np.array,
                 lats: np.array, pass_number: int, color_id: int):
    size = lons.size // 8
    index = max(size, 0) if pass_number % 2 == 0 else min(size * 7, size - 1)
    marker = ipy_l.Marker(location=(lats[index], lons[index]))
    marker.draggable = False
    marker.opacity = 0.8
    marker.popup = ipy_w.HTML(
        POPUP_TEMPLATE.format(color=COLORS[color_id], pass_number=pass_number))
    markers[pass_number] = marker
