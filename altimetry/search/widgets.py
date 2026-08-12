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
from pyinterp.geometry import geographic
from traitlets import TraitError

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
You can draw one bounding box, or input its coordinate using the widget on the
top right. Drawing a new box will delete the previous search results.
At the top right side of the map, you can select the period of interest, and the
mission. The default period is the last 1 day.</p>"""

#: Geographical box, given as (lon_min, lat_min, lon_max, lat_max). Coordinates
# are expected in degrees.
Box = tuple[float, float, float, float]


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
                                  value=datetime.date.today() -
                                  datetime.timedelta(days=5)))
        object.__setattr__(
            self, 'last_date',
            ipywidgets.DatePicker(description='Last date:',
                                  disabled=False,
                                  value=datetime.date.today()))

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

    def set_defaults(self, phase_date_end: datetime.date | None) -> None:
        """Set the default values of the date pickers.

        last_date defaults to the end date of the phase if one is defined,
        otherwise to today. start_date always defaults to last_date minus
        5 days (it never defaults to the phase start date).

        Args:
            phase_date_end: End date of the current phase, or None if there
                is no phase selected / the phase has no end date.
        """
        self.last_date.value = (phase_date_end if phase_date_end is not None
                                else datetime.date.today())
        self.start_date.value = self.last_date.value - datetime.timedelta(
            days=5)

    def validate(self, phase_date_start: datetime.date | None,
                 phase_date_end: datetime.date | None) -> None:
        """Check that the selected dates lie within the phase bounds.

        Args:
            phase_date_start: Start date of the current phase, or None if
                there is no lower bound.
            phase_date_end: End date of the current phase, or None if there
                is no upper bound.

        Raises:
            InvalidDate: If start_date is before phase_date_start, or
                last_date is after phase_date_end.
        """
        error_date = '<b><font color="#D32F2F">{}</font></b>'
        phase_date = '<b>{}</b>'

        if (phase_date_start is not None
                and self.start_date.value < phase_date_start):
            raise InvalidDate('The first date ' +
                              error_date.format(self.start_date.value) +
                              ' is before the start of the phase: ' +
                              phase_date.format(phase_date_start))
        if (phase_date_end is not None
                and self.last_date.value > phase_date_end):
            raise InvalidDate('The last date ' +
                              error_date.format(self.last_date.value) +
                              ' is after the end of the phase: ' +
                              phase_date.format(phase_date_end))


class GeoBox(ipywidgets.VBox):
    """GeoBox widget for inputing and displaying the Rectangle bounds.

    Args:
        min_lon: Minimum longitude of the geographical box (must be < max_lon).
        Given in degrees.
        min_lat: Minimum latitude of the geographical box (must be < max_lat and
        between -90° and 90°). Given in degrees.
        max_lon: Maximum longitude of the geographical box (must be > min_lon).
        Given in degrees.
        max_lat: Maximum latitude of the geographical box (must be > min_lat and
        between -90° and 90°). Given in degrees.
    """

    def __init__(
        self,
        min_lon=--65.0,
        min_lat=-30.0,
        max_lon=10.0,
        max_lat=60.0,
    ):
        self._initializing = True

        self.min_lon = ipywidgets.FloatText(
            value=min_lon,
            description='Min Lon',
        )
        self.min_lat = ipywidgets.FloatText(
            value=min_lat,
            description='Min Lat',
        )
        self.max_lon = ipywidgets.FloatText(
            value=max_lon,
            description='Max Lon',
        )
        self.max_lat = ipywidgets.FloatText(
            value=max_lat,
            description='Max Lat',
        )

        self.draw_button = ipywidgets.Button(
            description='Draw on map',
            icon='square-o',
            button_style='info',
        )

        self.error = ipywidgets.HTML()

        grid = ipywidgets.VBox([
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat,
        ])

        super().__init__([
            grid,
            self.draw_button,
            self.error,
        ])

        for w in (
                self.min_lon,
                self.min_lat,
                self.max_lon,
                self.max_lat,
        ):
            w.observe(self._on_change, names='value')

        self._initializing = False
        self._validate()

    def _on_change(self, change):
        # Observe new coordinates with a validator.
        self._validate()

    def _validate(self):
        # Only accept [-90, 90] latitudes, and ensure minimum bounds are always
        # inferior to maximum bounds (for both longitudes and latitudes). In
        # case the selection is not valid, the draw button will be disabled to
        # prevent firing the callbacks on an invalid shape.
        if self._initializing:
            return

        try:
            if not (-90 <= self.min_lat.value <= 90):
                raise TraitError('Min latitude must be between -90 and 90.')

            if not (-90 <= self.max_lat.value <= 90):
                raise TraitError('Max latitude must be between -90 and 90.')

            if self.min_lon.value > self.max_lon.value:
                raise TraitError('Min longitude must be <= max longitude.')

            if self.min_lat.value > self.max_lat.value:
                raise TraitError('Min latitude must be <= max latitude.')

            self.error.value = ''
            self.draw_button.disabled = False

        except TraitError as e:
            self.error.value = (f'<span style="color:red">{e}</span>')
            self.draw_button.disabled = True

    def set_bbox(self, min_lon: float, min_lat: float, max_lon: float,
                 max_lat: float):
        """Geographical box setter.

        Can be used to synchronize the GeoBox instance with other components
        defining a box.

        See Also:
            MapSelection: For handling both a GeoBox instance and DrawControl,
            and sharing their box definition.
        """
        self.min_lon.value = min_lon
        self.min_lat.value = min_lat
        self.max_lon.value = max_lon
        self.max_lat.value = max_lat

    @property
    def bbox(self) -> Box:
        """Geographical box as (lon_min, lat_min, lon_max, lat_max)."""
        return (
            self.min_lon.value,
            self.min_lat.value,
            self.max_lon.value,
            self.max_lat.value,
        )

    def on_draw(self, callback: Callable[[Box], None]):
        """Register a callback to fire when the geographical box is updated.

        Parameters
        ----------
        callback
        """
        self.draw_button.on_click(lambda _: callback(self.bbox))


def _setup_draw_control(
    on_draw: Callable[[ipywidgets.Widget, str, dict],
                      None]) -> ipyleaflet.Control:
    """Setup the map.

    Draw control is enabled to draw rectangles only. In addition, the rectangle
    is not owned by the draw control - said control is expected to be cleared
    once the rectangle ownership has been changed - so deletion and edition of
    the features are disabled.

    Args:
        on_draw: Callback called when the user draws a rectangle. It will be
        registered to the returned control widget.

    Returns:
        Draw control widget.
    """
    draw_control = ipyleaflet.DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}
    draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
    draw_control.circle = {}
    draw_control.edit = False
    draw_control.remove = False

    draw_control.on_draw(on_draw)

    return draw_control


def _setup_map(date_selection: DateSelection,
               mission_widget: ipywidgets.Dropdown, help: ipywidgets.Button,
               search: ipywidgets.Button, draw_control: ipyleaflet.DrawControl,
               geo_box: GeoBox) -> ipyleaflet.Map:
    """Setup the map.

    Args:
        date_selection: Date selection widget.
        search: Search button.
        help: Help button.
        on_draw: Callback called when the user draws a rectangle.
        geo_box: Geographical box widget.

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
    m.add_control(ipyleaflet.WidgetControl(widget=geo_box,
                                           position='topright'))
    return m


@dataclasses.dataclass
class MapSelection:
    """Map selection."""
    #: Selected area
    selection: geographic.Polygon | None = None
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
    # Single Rectangle instance shared by both the DrawControl and GeoBox
    # widgets.
    current_rect: ipyleaflet.Rectangle | None = None
    # GeoBox widget for displaying the current Rectangle and inputing another
    # geographical box.
    geo_box: GeoBox = dataclasses.field(default_factory=GeoBox)

    def __post_init__(self) -> None:
        self.draw_control = _setup_draw_control(self.handle_draw)
        self.m = _setup_map(self.date_selection, self.mission_widget,
                            self.help, self.search, self.draw_control,
                            self.geo_box)
        self.main_widget = ipywidgets.VBox([self.m, self.out])
        self.search.on_click(self.handle_compute)
        self.mission_widget.observe(self.mission_widget_callback,
                                    names='value')
        self.help.on_click(lambda _args: self.display_message(
            HTML_HELP.format(mission=self.mission_widget.value),
            button_style='info',
            width='800px'))
        self.geo_box.on_draw(self.draw_bbox)

    def mission_widget_callback(self, change):
        if not (change['old'] is None or self.mission_widget.value is None):
            self.delete_last_selection()
            self.draw_control.clear_polygons()

        if self.mission_widget.value is None:
            self.date_selection.set_defaults(None)
        else:
            mission_properties = models.MissionPropertiesLoader().load(
                self.mission_widget.value)
            self.date_selection.set_defaults(mission_properties.date_end)

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
        if self.current_rect is not None:
            self.m.remove(self.current_rect)
        self.current_rect = None

    def draw_bbox(self, bbox: Box):
        """Draw a geographical box on the map.

        Drawing the box will clear the existing selection, search results and
        related features on the map.

        Args:
            bbox: geographical box to draw

        See Also:
            delete_last_selection: Deletion existing selection, search results
            and related features on the map.
        """
        x0, y0, x1, y1 = bbox
        self.delete_last_selection()

        self.current_rect = ipyleaflet.Rectangle(
            bounds=[
                [y0, x0],
                [y1, x1],
            ],
            color='red',
            fill_opacity=0.1,
        )

        # Build a polygon with interpolated longitudes between the first and
        # last points to restrict the search area to the latitude of the
        # selected zone.
        xs = numpy.linspace(x0, x1, round(x1 - x0) * 2, endpoint=True)
        lons = list(reversed(xs)) + list(xs)
        lats = [y0] * len(xs) + [y1] * len(xs)
        # Close the polygon by adding the first
        # point at the end of the list.
        lons.append(lons[0])
        lats.append(lats[0])
        self.selection = geographic.Polygon(
            geographic.Ring(
                numpy.array(lons, dtype=numpy.float64),
                numpy.array(lats, dtype=numpy.float64),
            ))

        self.m.add(self.current_rect)

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

        try:
            coordinates = geo_json['geometry']['coordinates']
            x = numpy.array([item[0] for item in coordinates[0]])
            y = numpy.array([item[1] for item in coordinates[0]])

            bbox = min(x), min(y), max(x), max(y)
            self.geo_box.set_bbox(*bbox)
            self.draw_bbox(self.geo_box.bbox)
            self.draw_control.clear()
        except (KeyError, IndexError):
            self.delete_last_selection()

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

            self.date_selection.validate(mission_properties.date_start,
                                         mission_properties.date_end)

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
                self.selection, mission_properties, self.geo_box.bbox[0],
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
        selected_area: geographic.Polygon, first_date: numpy.datetime64,
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
