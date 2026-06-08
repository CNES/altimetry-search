# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Calculate the ephemeredes of satellites."""
from __future__ import annotations

import pathlib

import numpy
from numpy.typing import NDArray
import pandas
from pyinterp import TemporalAxis
from pyinterp.geometry import geographic
import xarray

from . import models, orf


def get_cycle_duration(dataset: xarray.Dataset) -> numpy.timedelta64:
    """Return the duration of a cycle.

    Args:
        dataset: Dataset containing the orbit file.

    Returns:
        Duration of a cycle.
    """
    start_time = dataset.start_time[0].values
    end_time = dataset.end_time[-1].values
    return end_time - start_time


def calculate_cycle_axis(
        cycle_duration: numpy.timedelta64,
        mission_properties: models.MissionProperties) -> TemporalAxis:
    """Calculate the cycle axis.

    Args:
        cycle_duration: Duration of a cycle.
        mission_properties: Selected mission's properties.

    Returns:
        Temporal axis of the cycle.
    """
    cycles = orf.load_json(pathlib.Path(mission_properties.orf_file))

    cycle_first_measurement = numpy.full(
        (mission_properties.nb_cycle, ),
        numpy.datetime64('NAT'),
        dtype='M8[ns]',
    )

    keys = sorted(cycles)
    for item in keys:
        cycle_first_measurement[
            item - (mission_properties.first_cycle)] = cycles[item]

    # Linear interpolation of NAT surrounded by known values,
    # and extrapolation of the tail with arange
    indices = numpy.arange(len(cycle_first_measurement))
    known = ~numpy.isnat(cycle_first_measurement)
    last_known_idx = indices[known][-1]

    known_indices = indices[known]
    known_values = cycle_first_measurement[known].astype('i8')
    interpolated = numpy.interp(indices, known_indices,
                                known_values).astype('M8[ns]')

    interior_nat = ~known & (indices <= last_known_idx)
    cycle_first_measurement[interior_nat] = interpolated[interior_nat]

    tail_nat = ~known & (indices > last_known_idx)
    tail_count = tail_nat.sum()
    if tail_count > 0:
        cycle_first_measurement[tail_nat] = numpy.full(
            (tail_count, ), cycle_duration, dtype='m8[ns]') * numpy.arange(
                1, 1 + tail_count) + cycles[keys[-1]]

    return TemporalAxis(cycle_first_measurement)


def get_selected_passes(
        mission: models.Mission | models.MissionProperties,
        date: numpy.datetime64,
        search_duration: numpy.timedelta64 | None = None) -> pandas.DataFrame:
    """Return the selected passes.

    Args:
        mission: Selected mission (or mission's properties)
        date: Date of the first pass.
        search_duration: Duration of the search.

    Returns:
        Temporal axis of the selected passes.
    """
    if isinstance(mission, models.MissionProperties):
        mission_properties = mission
    elif isinstance(mission, models.Mission):
        mission_properties = models.MissionPropertiesLoader().load(mission)

    # To avoid getting a warning from xarray about decoding timedeltas, we set
    # decode_timedelta to True. The warning appears because orbit files do not
    # have the appropriate attributes to decode timedeltas.
    # TODO rewrite the auxiliary data with the proper encoding
    # (dtype='timedelta64[ns]')
    with xarray.open_dataset(mission_properties.orbit_file,
                             decode_timedelta=True) as ds:
        passes_per_cycle = ds.sizes['pass_number']

        cycle_duration = get_cycle_duration(ds)
        search_duration = search_duration or cycle_duration
        axis = calculate_cycle_axis(cycle_duration, mission_properties)
        print(axis)
        dates = numpy.array([date, date + search_duration])
        print(dates)
        indices = axis.find_indexes(dates).ravel()
        print(indices)
        print(passes_per_cycle)
        cycle_numbers = numpy.repeat(
            numpy.arange(indices[0], indices[-1]) +
            mission_properties.first_cycle, passes_per_cycle)
        print(cycle_numbers)
        axis_slice = axis[indices[0]:indices[-1] + 1]
        first_date_of_cycle = numpy.repeat(axis_slice, passes_per_cycle)
        pass_numbers = numpy.tile(numpy.arange(1, passes_per_cycle + 1),
                                  indices[-1] - indices[0])
        dates_of_selected_passes = numpy.vstack(
            (ds.start_time.values, ) * len(axis_slice)).T + axis_slice
        dates_of_selected_passes = dates_of_selected_passes.T.ravel()
        selected_passes = TemporalAxis(dates_of_selected_passes).find_indexes(
            dates).ravel()
        size = selected_passes[-1] - selected_passes[0]

        result: numpy.ndarray = numpy.ndarray(
            (size, ),
            dtype=[('cycle_number', numpy.uint16),
                   ('pass_number', numpy.uint16),
                   ('first_measurement', 'M8[ns]'),
                   ('last_measurement', 'M8[ns]')])
        axis_slice = slice(selected_passes[0], selected_passes[-1])
        result['cycle_number'] = cycle_numbers[axis_slice]
        result['pass_number'] = pass_numbers[axis_slice]
        result['first_measurement'] = first_date_of_cycle[axis_slice]
        result['last_measurement'] = first_date_of_cycle[axis_slice]
        return pandas.DataFrame(result)


def _get_time_bounds(
    lat_nadir: NDArray,
    selected_time: NDArray,
    intersection: geographic.LineString,
) -> tuple[numpy.datetime64, numpy.datetime64]:
    """Return the time bounds of the selected pass.

    Args:
        lat_nadir: Latitude of the nadir.
        selected_time: Time of the selected pass.
        intersection: Intersection of the pass with the polygon.

    Returns:
        Time bounds of the selected pass.
    """
    # Remove NaN values
    selected_time = selected_time[numpy.isfinite(lat_nadir)]
    lat_nadir = lat_nadir[numpy.isfinite(lat_nadir)]

    if lat_nadir[0] > lat_nadir[-1]:
        lat_nadir = lat_nadir[::-1]
        selected_time = selected_time[::-1]

    y0 = intersection[0].lat
    y1 = intersection[len(intersection) -
                      1].lat if len(intersection) > 1 else y0
    t0 = numpy.searchsorted(lat_nadir, y0)
    t1 = numpy.searchsorted(lat_nadir, y1)
    bounds = (
        selected_time[min(t0, t1)],
        selected_time[max(t0, t1)],
    )
    return min(bounds), max(bounds)


def get_pass_passage_time(
        mission: models.Mission | models.MissionProperties,
        selected_passes: pandas.DataFrame,
        polygon: geographic.Polygon | None) -> pandas.DataFrame:
    """Return the passage time of the selected passes.

    Args:
        mission: Selected mission (or mission's properties)
        selected_passes: Selected passes.
        polygon: Polygon used to select the passes.

    Returns:
        Passage time of the selected passes.
    """
    if isinstance(mission, models.MissionProperties):
        mission_properties = mission
    elif isinstance(mission, models.Mission):
        mission_properties = models.MissionPropertiesLoader().load(mission)

    passes = numpy.array(sorted(set(selected_passes['pass_number']))) - 1

    # To avoid getting a warning from xarray about decoding timedeltas, we set
    # decode_timedelta to True. The warning appears because orbit files do not
    # have the appropriate attributes to decode timedeltas.
    # TODO rewrite the auxiliary data with the proper encoding
    # (dtype='timedelta64[ns]')
    with xarray.open_dataset(mission_properties.orbit_file,
                             decode_timedelta=True) as ds:
        lon = ds.line_string_lon.values[passes, :]
        lat = ds.line_string_lat.values[passes, :]
        pass_time = ds.pass_time.values[passes, :]
        lat_nadir = ds.lat_nadir.values[passes, :]

    result: NDArray[numpy.void] = numpy.ndarray(
        (len(passes), ),
        dtype=[('pass_number', numpy.uint16), ('first_time', 'm8[ns]'),
               ('last_time', 'm8[ns]')],
    )

    jx = 0
    wgs84 = geographic.Spheroid()

    for ix, pass_index in enumerate(passes):
        mask = numpy.isfinite(lon[ix, :]) & numpy.isfinite(lat[ix, :])
        line_string = geographic.LineString(
            lon[ix, mask].astype(numpy.float64),
            lat[ix, mask].astype(numpy.float64),
        )
        intersection_list = geographic.algorithms.intersection(
            line_string, polygon,
            spheroid=wgs84) if polygon else [line_string]
        if len(intersection_list) > 0:
            row: NDArray[numpy.void] = result[jx]
            row['pass_number'] = pass_index + 1
            row['first_time'], row['last_time'] = _get_time_bounds(
                lat_nadir[ix, :],
                pass_time[ix, :],
                # Assuming that only one intersection is possible,
                # since polygon is a rectangle.
                intersection_list[0],
            )
            jx += 1

    return pandas.DataFrame(result[:jx])
