# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""IPython widgets used by the application."""

import base64
from collections.abc import Callable
import dataclasses
import datetime
import traceback

import IPython.display
import ipyleaflet
import ipywidgets
import numpy
import pandas
import pyinterp.geodetic

from . import models, orbit, plotting

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
the half_orbits that intersect the area of interest are displayed on the map.
Click on the marker to view the pass number.<br>
You can draw multiple bounding boxes, but only the last one will be used for
the search. You can also delete one or all bounding boxes by clicking on the
<span style="background-color: lightgray;"><code>trash</code></span> icon.<br>
At the top right side of the map, you can select the period of interest, and the
mission. The default period is the last 1 day.</p>"""


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
            self.last_date.value) - numpy.datetime64(
                self.start_date.value)  # type: ignore[return-value]


def _setup_draw_control(
    on_draw: Callable[[ipywidgets.Widget, str, dict],
                      None]) -> ipyleaflet.Control:
    """Setup the map.

    Args:
        on_draw: Callback called when the user draws a rectangle.

    Returns:
        Map widget.
    """
    draw_control = ipyleaflet.DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}
    draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
    draw_control.circle = {}
    draw_control.edit = False

    draw_control.on_draw(on_draw)

    return draw_control


def _setup_map(date_selection: DateSelection,
               mission_widget: ipywidgets.Dropdown, help: ipywidgets.Button,
               search: ipywidgets.Button,
               draw_control: ipyleaflet.DrawControl) -> ipyleaflet.Map:
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

    m = ipyleaflet.Map(center=[0, 0],
                       zoom=2,
                       layout=layout,
                       projection=ipyleaflet.projections.EPSG4326)
    m.scroll_wheel_zoom = True
    m.add_control(ipyleaflet.FullScreenControl())
    m.add_control(draw_control)
    m.add_control(
        ipyleaflet.WidgetControl(widget=mission_widget, position='topright'))
    m.add_control(
        ipyleaflet.WidgetControl(widget=date_selection.display(),
                                 position='topright'))
    m.add_control(
        ipyleaflet.WidgetControl(widget=search, position='bottomright'))
    m.add_control(ipyleaflet.WidgetControl(widget=help, position='bottomleft'))
    return m


@dataclasses.dataclass
class MapSelection:
    """Map selection."""
    #: Selected area
    selection: pyinterp.geodetic.Polygon | None = None
    #: Bounds of the selected area
    bounds: tuple[tuple[float, float],
                  tuple[float, float]] = dataclasses.field(
                      default_factory=lambda: DEFAULT_BOUNDS)
    #: HalfOrbit footprint to display
    half_orbits: list[plotting.HalfOrbitFootprint] = dataclasses.field(
        default_factory=list)
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
    # Draw control of the map
    draw_control: ipyleaflet.DrawControl = dataclasses.field(init=False)
    #: Output widget
    out: ipywidgets.Output = dataclasses.field(
        default_factory=ipywidgets.Output)
    #: Main widget
    main_widget: ipywidgets.VBox = dataclasses.field(init=False)
    #: Widget to display a message (information or error)
    widget_message: ipywidgets.VBox | None = None
    # Widget to choose a mission
    mission_widget: ipywidgets.Dropdown = dataclasses.field(
        default_factory=lambda: ipywidgets.Dropdown(
            options=[('--- Select a mission ---', None)] +
            [(member.value, member) for member in models.Mission],
            description='Mission:',
            value=None,
        ))

    def __post_init__(self) -> None:
        self.draw_control = _setup_draw_control(self.handle_draw)
        self.m = _setup_map(self.date_selection, self.mission_widget,
                            self.help, self.search, self.draw_control)
        self.main_widget = ipywidgets.VBox([self.m, self.out])
        self.search.on_click(self.handle_compute)
        self.mission_widget.observe(self.mission_widget_callback,
                                    names='value')
        self.help.on_click(lambda _args: self.display_message(
            HTML_HELP.format(mission=self.mission_widget.value),
            button_style='info',
            width='800px'))

    def mission_widget_callback(self, change):
        if not (change['old'] is None or self.mission_widget.value is None):
            self.delete_last_selection()
            self.draw_control.clear_polygons()

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

            # Remove the last half_orbits displayed on the map.
            self.remove_half_orbit_footprints()

            # Display a message to inform the user that the computation is in
            # progress.
            with self.out:
                IPython.display.display('Computing...')

            if self.mission_widget.value is None:
                self.display_message('Please select a mission.')
                return

            mission_properties = models.MissionPropertiesLoader().load(
                self.mission_widget.value)

            first_date, search_duration = self.date_selection.values()

            # Compute the selected passes.
            selected_passes = compute_selected_passes(self.selection,
                                                      first_date,
                                                      search_duration,
                                                      mission_properties)

            # If no pass is found, display a message and return.
            if len(selected_passes) == 0:
                self.out.clear_output()
                self.display_message(
                    'No pass found in the selected area. Please select '
                    'another area or extend the search period.',
                    button_style='warning')
                return

            # Plot the half_orbits on the map.
            self.half_orbits = plotting.plot_selected_passes(
                self.selection, mission_properties, self.bounds[0][0],
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


def compute_selected_passes(
        selected_area: pyinterp.geodetic.Polygon, first_date: numpy.datetime64,
        search_duration: numpy.timedelta64,
        mission: models.Mission | models.MissionProperties
) -> pandas.DataFrame:
    """Compute the selected passes.

    Args:
        selected_area: selected area
        first_date: selected first date
        search_duration: search duration
        mission: selected mission (or mission's properties)

    Returns:
        Selected passes.
    """
    if isinstance(mission, models.Mission):
        mission = models.MissionPropertiesLoader().load(mission)
    if selected_area is None:
        raise ValueError('No area selected.')
    if search_duration < numpy.timedelta64(0, 'D'):  # type: ignore
        raise InvalidDate('First date must be before last date.')
    selected_passes = orbit.get_selected_passes(mission, first_date,
                                                search_duration)
    pass_passage_time = orbit.get_pass_passage_time(mission, selected_passes,
                                                    selected_area)
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
