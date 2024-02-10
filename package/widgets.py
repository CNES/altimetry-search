# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""IPython widgets used by the application."""
from __future__ import annotations

import base64
from collections.abc import Callable
import dataclasses
import datetime
import traceback

import IPython.display
import ipyleaflet
import ipywidgets
import numpy
from numpy.typing import NDArray
import pandas
import pyinterp.geodetic
import xarray

from . import orbit

#: Default bounds of the map
DEFAULT_BOUNDS = ((-180, -90), (180, 90))

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

#: HTML Template for the popup of the marker
DOWNLOAD_TEMPLATE = """<a href="data:file/csv;base64,{b64}"
download="selected_passes.csv"><button style="background-color: #4285F4;
color: white; border-radius: 4px; padding: 10px 16px; font-size: 14px;
font-weight: bold; border: none; cursor: pointer;">
Download data as a CSV file</button></a>"""

#: HTML Template for the help message
HTML_HELP = """<p style="line-height: 2em;">
Use the widget below to select the area of interest (square
icon). You can also use the
<span style="background-color: lightgray;"><code>+</code></span> and
<span style="background-color: lightgray;"><code>-</code></span> buttons to
zoom in and out and wheel mouse to zoom in and out. Once you have selected the
area of interest, click on the
<span style="background-color: lightgray;"><code>Search</code></span> button to
search for {mission} passes. The results are displayed in the table below and
the swaths that intersect the area of interest are displayed on the map. Click
on the marker to view the pass number.<br>
You can draw multiple bounding boxes, but only the last one will be used for
the search. You can also delete one or all bounding boxes by clicking on the
<span style="background-color: lightgray;"><code>trash</code></span> icon.<br>
At the top right side of the map, you can select the period of interest.
The default period is the last 1 day.</p>"""

#: Type of a pass polygon
PassPolygon = tuple[int, pyinterp.geodetic.Polygon]


@dataclasses.dataclass
class Swath:
    """Swath definition."""
    #: Left polygon of the swath
    left: ipyleaflet.Polygon
    #: Right polygon of the swath
    right: ipyleaflet.Polygon
    #: Marker of the swath to display the pass number
    marker: ipyleaflet.Marker


class InvalidDate(Exception):
    """Invalid date exception."""


@dataclasses.dataclass(frozen=True)
class DateSelection:
    """Date selection widget."""

    #: First date
    start_date: ipywidgets.DatePicker = dataclasses.field(init=False)

    #: Last date
    last_date: ipywidgets.DatePicker = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'start_date',
            ipywidgets.DatePicker(description='First date:',
                                  disabled=False,
                                  value=datetime.date.today()))
        object.__setattr__(
            self, 'last_date',
            ipywidgets.DatePicker(description='Last date:',
                                  disabled=False,
                                  value=datetime.date.today() +
                                  datetime.timedelta(days=1)))

    def display(self) -> ipywidgets.Widget:
        """Display the widget.

        Returns:
            Widget to display.
        """
        return ipywidgets.VBox([self.start_date, self.last_date])

    def values(self) -> tuple[numpy.datetime64, numpy.timedelta64]:
        """Return the values of the widget.

        Returns:
            First date and search duration.
        """
        return numpy.datetime64(self.start_date.value), numpy.datetime64(
            self.last_date.value) - numpy.datetime64(self.start_date.value)


def _setup_map(
    date_selection: DateSelection,
    help: ipywidgets.Button,
    search: ipywidgets.Button,
    on_draw: Callable[[ipywidgets.Widget, str, dict], None],
) -> ipyleaflet.Map:
    """Setup the map.

    Args:
        date_selection: Date selection widget.
        search: Search button.
        help: Help button.
        on_draw: Callback called when the user draws a rectangle.

    Returns:
        Map widget.
    """
    layout = ipywidgets.Layout(width='100%', height='600px')

    draw_control = ipyleaflet.DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}
    draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
    draw_control.circle = {}
    draw_control.edit = False

    m = ipyleaflet.Map(center=[0, 0],
                       zoom=2,
                       layout=layout,
                       projection=ipyleaflet.projections.EPSG4326)
    m.scroll_wheel_zoom = True
    m.add_control(ipyleaflet.FullScreenControl())
    m.add_control(draw_control)
    m.add_control(
        ipyleaflet.WidgetControl(widget=date_selection.display(),
                                 position='topright'))
    m.add_control(
        ipyleaflet.WidgetControl(widget=search, position='bottomright'))
    draw_control.on_draw(on_draw)
    m.add_control(ipyleaflet.WidgetControl(widget=help, position='bottomleft'))
    return m


@dataclasses.dataclass
class MapSelection:
    """Map selection widget."""
    #: Selected area
    selection: pyinterp.geodetic.Polygon | None = None
    #: Bounds of the selected area
    bounds: tuple[tuple[float, float],
                  tuple[float, float]] = dataclasses.field(
                      default_factory=lambda: DEFAULT_BOUNDS)
    #: Swaths to display
    swaths: list[Swath] = dataclasses.field(default_factory=list)
    #: Date selection widget
    date_selection: DateSelection = dataclasses.field(
        default_factory=DateSelection)
    #: Search button
    search: ipywidgets.Button = dataclasses.field(
        default_factory=lambda: ipywidgets.Button(description='Search'))
    #: Help button
    help: ipywidgets.Button = dataclasses.field(
        default_factory=lambda: ipywidgets.Button(description='Help'))
    #: Map widget
    m: ipyleaflet.Map = dataclasses.field(init=False)
    #: Output widget
    out: ipywidgets.Output = dataclasses.field(
        default_factory=ipywidgets.Output)
    #: Main widget
    main_widget: ipywidgets.VBox = dataclasses.field(init=False)
    #: Widget to display a message (information or error)
    widget_message: ipywidgets.VBox | None = None

    def __post_init__(self) -> None:
        self.m = _setup_map(self.date_selection, self.help, self.search,
                            self.handle_draw)
        self.main_widget = ipywidgets.VBox([self.m, self.out])
        self.search.on_click(self.handle_compute)
        self.help.on_click(lambda _args: self.display_message(
            HTML_HELP.format(mission='SWOT'),
            button_style='info',
            width='800px'))

    def display(self) -> ipywidgets.Widget:
        """Display the widget.

        Returns:
            Widget to display.
        """
        return self.main_widget

    def handle_widget_message(self, *_args) -> None:
        """Handle the click on the close button of the message widget."""
        self.m.remove_control(self.m.controls[-1])
        self.widget_message = None
        self.search.disabled = False

    def remove_swaths(self) -> None:
        """Remove the swaths from the map."""
        for item in self.swaths:
            self.m.remove(item.left)
            self.m.remove(item.right)
            self.m.remove(item.marker)
        self.swaths.clear()
        self.out.clear_output()

    def delete_last_selection(self) -> None:
        """Delete the last selection."""
        self.remove_swaths()
        self.selection = None
        self.bounds = DEFAULT_BOUNDS

    def handle_draw(self, _target, action, geo_json) -> None:
        """Handle the draw event.

        Args:
            target: Target of the event.
            action: Action of the event.
            geo_json: GeoJSON object.
        """
        if action == 'deleted':
            self.delete_last_selection()
            return

        if action != 'created':
            return

        self.delete_last_selection()

        try:
            coordinates = geo_json['geometry']['coordinates']

            # Build a polygon with interpolated longitudes between the first and
            # last points to restrict the search area to the latitude of the
            # selected zone.
            x = numpy.array([item[0] for item in coordinates[0]])
            y = numpy.array([item[1] for item in coordinates[0]])
            x0, x1 = x[0], x[2]
            y0, y1 = y[0], y[1]
            xs = numpy.linspace(x0, x1, round(x1 - x0) * 2, endpoint=True)
            self.bounds = ((min(x), min(y)), (max(x), max(y)))
            points = [
                pyinterp.geodetic.Point(item, y0) for item in reversed(xs)
            ] + [pyinterp.geodetic.Point(item, y1) for item in xs]
            points.append(points[0])
            self.selection = pyinterp.geodetic.Polygon(points)
        except (KeyError, IndexError):
            self.selection = None

    def display_message(self,
                        msg,
                        button_style: str | None = None,
                        width: str | None = None) -> None:
        """Display a message on the map.

        Args:
            msg: Message to display.
            button_style: Style of the close button.
        """
        button_style = button_style or 'danger'
        panel = ipywidgets.HTML(
            msg,
            layout=ipywidgets.Layout(
                width=width,
                line_height='1.5',  # Adjust the line height here
            ))
        close = ipywidgets.Button(description='Close.',
                                  disabled=False,
                                  button_style=button_style)
        self.widget_message = ipywidgets.VBox([panel, close])
        assert self.widget_message is not None
        self.widget_message.box_style = 'danger'
        self.widget_message.layout = ipywidgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            border='solid lightgray 2px',
        )
        close.on_click(self.handle_widget_message)
        self.m.add_control(
            ipyleaflet.WidgetControl(widget=self.widget_message,
                                     position='bottomright'))
        # Disable the search button while the message is displayed.
        self.search.disabled = True

    def handle_compute(self, _args) -> None:
        """Handle the click on the search button."""
        self.search.disabled = True
        try:
            if self.selection is None:
                # If no area is selected, display a message and return.
                if self.widget_message is None:
                    self.display_message(
                        'Please select an area by drawing a rectangle on the '
                        'map, then click on the <b>Search</b> button.')
                return

            # Remove the last swaths displayed on the map.
            self.remove_swaths()

            # Display a message to inform the user that the computation is in
            # progress.
            with self.out:
                IPython.display.display('Computing...')

            # Compute the selected passes.
            selected_passes = compute_selected_passes(self.date_selection,
                                                      self)

            # If no pass is found, display a message and return.
            if len(selected_passes) == 0:
                self.out.clear_output()
                self.display_message(
                    'No pass found in the selected area. Please select '
                    'another area or extend the search period.',
                    button_style='warning')
                return

            # Plot the swaths on the map.
            self.swaths = plot_selected_passes(self, selected_passes)

            # Rename the columns of the DataFrame to display them in the
            # output widget.
            selected_passes.rename(
                columns={
                    'first_measurement': 'First date',
                    'last_measurement': 'Last date',
                    'cycle_number': 'Cycle number',
                    'pass_number': 'Pass number'
                },
                inplace=True,
            )

            # Draw the swaths on the map.
            for item in self.swaths:
                self.m.add_layer(item.left)
                self.m.add_layer(item.right)
                self.m.add_layer(item.marker)

            # Finally, display the DataFrame in the output widget.
            self.out.clear_output()
            with self.out:
                IPython.display.display(selected_passes)
                # Generate a link to download the data as a CSV file.
                csv = selected_passes.to_csv(sep=';', index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                IPython.display.display(
                    ipywidgets.HTML(DOWNLOAD_TEMPLATE.format(b64=b64)))
        except InvalidDate as err:
            self.out.clear_output()
            self.display_message(str(err))
        # All exceptions thrown in a callback are lost. To avoid this, we catch
        # all exceptions and display them in the output widget.
        # pylint: disable=broad-exception-caught,broad-exception-caught
        except Exception as err:
            self.out.clear_output()
            self.display_message(
                '<b><font color="red">An error occurred while computing the '
                'selected passes.</font></b>'
                '<pre font-size: 11px; font-family: monospace;>' + str(err) +
                '<br>'.join(traceback.format_exc().splitlines()) + '</pre>',
                button_style='danger',
                width='800px')
        finally:
            self.search.disabled = self.widget_message is not None
        # pylint: enable=broad-exception-caught,broad-exception-caught


@dataclasses.dataclass(frozen=True)
class MainWidget:
    """Main widget."""
    #: Map selection widget
    map_selection: MapSelection = dataclasses.field(
        default_factory=MapSelection)
    #: Date selection widget
    date_selection: DateSelection = dataclasses.field(
        default_factory=DateSelection)
    #: Main widget
    main_widget: ipywidgets.HBox = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'main_widget',
            ipywidgets.HBox(
                [self.map_selection.display(),
                 self.date_selection.display()]))

    def display(self) -> ipywidgets.Widget:
        """Display the widget.

        Returns:
            Widget to display.
        """
        return self.main_widget


def _load_one_polygons(x: NDArray, y: NDArray) -> pyinterp.geodetic.Polygon:
    """Load a polygon from a set of coordinates.

    Args:
        x: X coordinates.
        y: Y coordinates.

    Returns:
        Polygon.
    """
    m = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[m]
    y = y[m]
    return pyinterp.geodetic.Polygon(
        [pyinterp.geodetic.Point(x, y) for x, y in zip(x, y)])


def load_polygons(
        pass_number: NDArray) -> tuple[list[PassPolygon], list[PassPolygon]]:
    """Load the polygons of the selected passes.

    Args:
        pass_number: Pass numbers to load.

    Returns:
        Left and right polygons of the selected passes.
    """
    index = pass_number - 1

    left_polygon: list[PassPolygon] = []
    right_polygon: list[PassPolygon] = []

    with xarray.open_dataset(orbit.DATASET) as ds:
        for ix in index:
            left_polygon.append(
                (ix + 1,
                 _load_one_polygons(ds.left_polygon_lon[ix, :].values,
                                    ds.left_polygon_lat[ix, :].values)))
            right_polygon.append(
                (ix + 1,
                 _load_one_polygons(ds.right_polygon_lon[ix, :].values,
                                    ds.right_polygon_lat[ix, :].values)))
    return left_polygon, right_polygon


def compute_selected_passes(date_selection: DateSelection,
                            map_selection: MapSelection) -> pandas.DataFrame:
    """Compute the selected passes.

    Args:
        date_selection: Date selection widget.
        map_selection: Map selection widget.

    Returns:
        Selected passes.
    """
    if map_selection.selection is None:
        raise ValueError('No area selected.')
    first_date, search_duration = date_selection.values()
    if search_duration < numpy.timedelta64(0, 'D'):  # type: ignore
        raise InvalidDate('First date must be before last date.')
    selected_passes = orbit.get_selected_passes(first_date, search_duration)
    pass_passage_time = orbit.get_pass_passage_time(selected_passes,
                                                    map_selection.selection)
    selected_passes = selected_passes.join(
        pass_passage_time.set_index('pass_number'),
        on='pass_number',
        how='right')
    selected_passes.sort_values(by=['cycle_number', 'pass_number'],
                                inplace=True)
    selected_passes['first_measurement'] += selected_passes['first_time']
    selected_passes['last_measurement'] += selected_passes['last_time']
    selected_passes.drop(columns=['first_time', 'last_time'], inplace=True)
    selected_passes['first_measurement'] = selected_passes[
        'first_measurement'].dt.floor('s')
    selected_passes['last_measurement'] = selected_passes[
        'last_measurement'].dt.floor('s')
    selected_passes.reset_index(drop=True, inplace=True)
    return selected_passes


def plot_swath(
    pass_number: int,
    item: pyinterp.geodetic.Polygon,
    bbox: pyinterp.geodetic.Polygon,
    layers: dict[int, ipyleaflet.Polygon],
    markers: dict[int, ipyleaflet.Marker],
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

    lons = numpy.array([p.lon for p in outer])
    lats = numpy.array([p.lat for p in outer])
    lons = numpy.deg2rad(
        pyinterp.geodetic.normalize_longitudes(
            numpy.array([p.lon for p in outer]), east))
    lons = numpy.unwrap(lons, discont=numpy.pi)
    lons = numpy.rad2deg(lons)

    color_id = pass_number % len(COLORS)
    layers[pass_number] = ipyleaflet.Polygon(
        locations=[(y, x) for x, y in zip(lons, lats)],
        color=COLORS[color_id],
        fill_color=COLORS[color_id],
    )

    # Add a marker to display the pass number on the map if it does not already
    # exist.
    if pass_number not in markers:
        size = lons.size // 8
        index = max(size, 0) if pass_number % 2 == 0 else min(
            size * 7, size - 1)
        marker = ipyleaflet.Marker(location=(lats[index], lons[index]))
        marker.draggable = False
        marker.opacity = 0.8
        marker.popup = ipywidgets.HTML(
            POPUP_TEMPLATE.format(color=COLORS[color_id],
                                  pass_number=pass_number))
        markers[pass_number] = marker


def plot_selected_passes(map_selection: MapSelection,
                         df: pandas.DataFrame) -> list[Swath]:
    """Plot the selected passes.

    Args:
        map_selection: Map selection widget.
        df: Selected passes.

    Returns:
        The swaths plotted on the map.
    """
    polygon = map_selection.selection
    bbox: pyinterp.geodetic.Polygon = (  # type: ignore[assignment]
        polygon if polygon is not None else
        pyinterp.geodetic.Box.whole_earth().as_polygon())

    (left_swath, right_swath) = load_polygons(
        df['pass_number'].values)  # type: ignore[arg-type]

    left_layers: dict[int, ipyleaflet.Polygon] = {}
    right_layers: dict[int, ipyleaflet.Polygon] = {}
    markers: dict[int, ipyleaflet.Marker] = {}
    east = map_selection.bounds[0][0]
    tuple(
        map(lambda x: plot_swath(*x, bbox, left_layers, markers, east),
            left_swath))
    tuple(
        map(lambda x: plot_swath(*x, bbox, right_layers, markers, east),
            right_swath))
    return [
        Swath(left=left_layers.get(pass_number, ipyleaflet.Polygon()),
              right=right_layers.get(pass_number, ipyleaflet.Polygon()),
              marker=marker) for pass_number, marker in markers.items()
    ]
