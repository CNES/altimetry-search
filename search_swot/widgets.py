# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""IPython widgets used by the application."""

import base64
from collections.abc import Callable
import dataclasses
from datetime import date, timedelta
import traceback

from IPython.display import display
import ipyleaflet as ipy_l
import ipywidgets as ipy_w
import numpy as np
import pandas as pd
import pyinterp.geodetic as py_geod

from .models import Mission
from .orbit import get_pass_passage_time, get_selected_passes
from .plotting import HalfOrbitFootprint, plot_selected_passes

#: Default bounds of the map
DEFAULT_BOUNDS = ((-180, -90), (180, 90))

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
the half_orbits that intersect the area of interest are displayed on the map. Click
on the marker to view the pass number.<br>
You can draw multiple bounding boxes, but only the last one will be used for
the search. You can also delete one or all bounding boxes by clicking on the
<span style="background-color: lightgray;"><code>trash</code></span> icon.<br>
At the top right side of the map, you can select the period of interest, and the mission.
The default period is the last 1 day.</p>"""


class InvalidDate(Exception):
    """Invalid date exception."""


@dataclasses.dataclass(frozen=True)
class DateSelection:
    """Date selection widget."""

    #: First date
    start_date: ipy_w.DatePicker = dataclasses.field(init=False)

    #: Last date
    last_date: ipy_w.DatePicker = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'start_date',
            ipy_w.DatePicker(description='First date:',
                             disabled=False,
                             value=date.today()))
        object.__setattr__(
            self, 'last_date',
            ipy_w.DatePicker(description='Last date:',
                             disabled=False,
                             value=date.today() + timedelta(days=1)))

    def display(self) -> ipy_w.Widget:
        """Display the widget.

        Returns:
            Widget to display.
        """
        return ipy_w.VBox([self.start_date, self.last_date])

    def values(self) -> tuple[np.datetime64, np.timedelta64]:
        """Return the values of the widget.

        Returns:
            First date and search duration.
        """
        return np.datetime64(self.start_date.value), np.datetime64(
            self.last_date.value) - np.datetime64(self.start_date.value)


def _setup_map(
    date_selection: DateSelection,
    mission_widget: ipy_w.Dropdown,
    help: ipy_w.Button,
    search: ipy_w.Button,
    on_draw: Callable[[ipy_w.Widget, str, dict], None],
) -> ipy_l.Map:
    """Setup the map.

    Args:
        date_selection: Date selection widget.
        search: Search button.
        help: Help button.
        on_draw: Callback called when the user draws a rectangle.

    Returns:
        Map widget.
    """
    layout = ipy_w.Layout(width='100%', height='600px')

    draw_control = ipy_l.DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}
    draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
    draw_control.circle = {}
    draw_control.edit = False

    m = ipy_l.Map(center=[0, 0],
                  zoom=2,
                  layout=layout,
                  projection=ipy_l.projections.EPSG4326)
    m.scroll_wheel_zoom = True
    m.add_control(ipy_l.FullScreenControl())
    m.add_control(draw_control)
    m.add_control(
        ipy_l.WidgetControl(widget=mission_widget, position='topright'))
    m.add_control(
        ipy_l.WidgetControl(widget=date_selection.display(),
                            position='topright'))
    m.add_control(ipy_l.WidgetControl(widget=search, position='bottomright'))
    draw_control.on_draw(on_draw)
    m.add_control(ipy_l.WidgetControl(widget=help, position='bottomleft'))
    return m


@dataclasses.dataclass
class MapSelection:
    """Map selection."""
    #: Selected area
    selection: py_geod.Polygon | None = None
    #: Bounds of the selected area
    bounds: tuple[tuple[float, float],
                  tuple[float, float]] = dataclasses.field(
                      default_factory=lambda: DEFAULT_BOUNDS)
    #: HalfOrbit footprint to display
    half_orbits: list[HalfOrbitFootprint] = dataclasses.field(
        default_factory=list)
    #: Date selection widget
    date_selection: DateSelection = dataclasses.field(
        default_factory=DateSelection)
    #: Search button
    search: ipy_w.Button = dataclasses.field(
        default_factory=lambda: ipy_w.Button(description='Search'))
    #: Help button
    help: ipy_w.Button = dataclasses.field(
        default_factory=lambda: ipy_w.Button(description='Help'))
    #: Map widget
    m: ipy_l.Map = dataclasses.field(init=False)
    #: Output widget
    out: ipy_w.Output = dataclasses.field(default_factory=ipy_w.Output)
    #: Main widget
    main_widget: ipy_w.VBox = dataclasses.field(init=False)
    #: Widget to display a message (information or error)
    widget_message: ipy_w.VBox | None = None
    # Widget to choose a mission
    mission_widget: ipy_w.Dropdown = dataclasses.field(
        default_factory=lambda: ipy_w.Dropdown(options=[(member.value, member)
                                                        for member in Mission],
                                               description='Mission:'))

    def __post_init__(self) -> None:
        self.m = _setup_map(self.date_selection, self.mission_widget,
                            self.help, self.search, self.handle_draw)
        self.main_widget = ipy_w.VBox([self.m, self.out])
        self.search.on_click(self.handle_compute)
        self.help.on_click(lambda _args: self.display_message(
            HTML_HELP.format(mission=self.mission_widget.value),
            button_style='info',
            width='800px'))

    def display(self) -> ipy_w.Widget:
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

    def remove_half_orbit_footprints(self) -> None:
        """Remove the half_orbits from the map."""
        for item in self.half_orbits:
            for v in vars(item):
                self.m.remove(getattr(item, v))
        self.half_orbits.clear()
        self.out.clear_output()

    def delete_last_selection(self) -> None:
        """Delete the last selection."""
        self.remove_half_orbit_footprints()
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
            x = np.array([item[0] for item in coordinates[0]])
            y = np.array([item[1] for item in coordinates[0]])
            x0, x1 = x[0], x[2]
            y0, y1 = y[0], y[1]
            xs = np.linspace(x0, x1, round(x1 - x0) * 2, endpoint=True)
            self.bounds = ((min(x), min(y)), (max(x), max(y)))
            points = [py_geod.Point(item, y0) for item in reversed(xs)
                      ] + [py_geod.Point(item, y1) for item in xs]
            points.append(points[0])
            self.selection = py_geod.Polygon(points)
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
        panel = ipy_w.HTML(
            msg,
            layout=ipy_w.Layout(
                width=width,
                line_height='1.5',  # Adjust the line height here
            ))
        close = ipy_w.Button(description='Close.',
                             disabled=False,
                             button_style=button_style)
        self.widget_message = ipy_w.VBox([panel, close])
        assert self.widget_message is not None
        self.widget_message.box_style = 'danger'
        self.widget_message.layout = ipy_w.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            border='solid lightgray 2px',
        )
        close.on_click(self.handle_widget_message)
        self.m.add_control(
            ipy_l.WidgetControl(widget=self.widget_message,
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

            # Remove the last half_orbits displayed on the map.
            self.remove_half_orbit_footprints()

            # Display a message to inform the user that the computation is in
            # progress.
            with self.out:
                display('Computing...')

            # Compute the selected passes.
            selected_passes = compute_selected_passes(self)

            # If no pass is found, display a message and return.
            if len(selected_passes) == 0:
                self.out.clear_output()
                self.display_message(
                    'No pass found in the selected area. Please select '
                    'another area or extend the search period.',
                    button_style='warning')
                return

            # Plot the half_orbits on the map.
            self.half_orbits = plot_selected_passes(self.selection,
                                                    self.mission_widget.value,
                                                    self.bounds[0][0],
                                                    selected_passes)

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

            # Draw the half_orbits on the map.
            for item in self.half_orbits:
                for v in vars(item):
                    self.m.add_layer(getattr(item, v))

            # Finally, display the DataFrame in the output widget.
            self.out.clear_output()
            with self.out:
                display(selected_passes)
                # Generate a link to download the data as a CSV file.
                csv = selected_passes.to_csv(sep=';', index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                display(ipy_w.HTML(DOWNLOAD_TEMPLATE.format(b64=b64)))
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


def compute_selected_passes(map_selection: MapSelection) -> pd.DataFrame:
    """Compute the selected passes.

    Args:
        map_selection: Map Selection.

    Returns:
        Selected passes.
    """
    if map_selection.selection is None:
        raise ValueError('No area selected.')
    first_date, search_duration = map_selection.date_selection.values()
    if search_duration < np.timedelta64(0, 'D'):  # type: ignore
        raise InvalidDate('First date must be before last date.')
    mission = map_selection.mission_widget.value
    selected_passes = get_selected_passes(mission, first_date, search_duration)
    pass_passage_time = get_pass_passage_time(mission, selected_passes,
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
