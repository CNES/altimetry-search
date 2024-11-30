# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Calculate the ephemeredes of the SWOT satellite."""
from __future__ import annotations

import pathlib

import numpy
from numpy.typing import NDArray
import pandas
import pyinterp
import xarray

from . import orf

#: Orbit Repetition File
ORF = pathlib.Path(__file__).parent / 'SWOT_ORF.json'

#: Orbit File
DATASET = pathlib.Path(__file__).parent / 'SWOT_orbit.nc'

#: Number of passes per cycle
PASSES_PER_CYCLE = 584


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
        cycle_duration: numpy.timedelta64) -> pyinterp.TemporalAxis:
    """Calculate the cycle axis.

    Args:
        cycle_duration: Duration of a cycle

    Returns:
        Temporal axis of the cycle.
    """
    cycles = orf.load_json(ORF)

    cycle_first_measurement = numpy.full(
        (200, ),
        numpy.datetime64('NAT'),
        dtype='M8[ns]',
    )
    keys = sorted(cycles)
    for item in keys:
        cycle_first_measurement[item - 1] = cycles[item]
    undefined = numpy.isnat(cycle_first_measurement)
    cycle_first_measurement[undefined] = numpy.full(
        (undefined.sum(), ), cycle_duration, dtype='m8[ns]') * numpy.arange(
            1, 1 + undefined.sum()) + cycles[keys[-1]]
    return pyinterp.TemporalAxis(cycle_first_measurement)


def get_selected_passes(
        date: numpy.datetime64,
        search_duration: numpy.timedelta64 | None = None) -> pandas.DataFrame:
    """Return the selected passes.

    Args:
        date: Date of the first pass.
        search_duration: Duration of the search.

    Returns:
        Temporal axis of the selected passes.
    """
    with xarray.open_dataset(DATASET.resolve()) as ds:
        cycle_duration = get_cycle_duration(ds)
        search_duration = search_duration or cycle_duration
        axis = calculate_cycle_axis(cycle_duration)
        dates = numpy.array([date, date + search_duration])
        indices = axis.find_indexes(dates).ravel()
        cycle_numbers = numpy.repeat(
            numpy.arange(indices[0], indices[-1]) + 1, PASSES_PER_CYCLE)
        axis_slice = axis[indices[0]:indices[-1] + 1]
        first_date_of_cycle = numpy.repeat(axis_slice, PASSES_PER_CYCLE)
        pass_numbers = numpy.tile(numpy.arange(1, PASSES_PER_CYCLE + 1),
                                  indices[-1] - indices[0])
        dates_of_selected_passes = numpy.vstack(
            (ds.start_time.values, ) * len(axis_slice)).T + axis_slice
        dates_of_selected_passes = dates_of_selected_passes.T.ravel()
        selected_passes = pyinterp.TemporalAxis(
            dates_of_selected_passes).find_indexes(dates).ravel()
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
    intersection: pyinterp.geodetic.LineString,
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
        selected_passes: pandas.DataFrame,
        polygon: pyinterp.geodetic.Polygon | None) -> pandas.DataFrame:
    """Return the passage time of the selected passes.

    Args:
        selected_passes: Selected passes.
        polygon: Polygon used to select the passes.

    Returns:
        Passage time of the selected passes.
    """
    passes = numpy.array(sorted(set(selected_passes['pass_number']))) - 1
    with xarray.open_dataset(DATASET) as ds:
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

    for ix, pass_index in enumerate(passes):
        line_string = pyinterp.geodetic.LineString([
            pyinterp.geodetic.Point(x, y)
            for x, y in zip(lon[ix, :], lat[ix, :])
            if numpy.isfinite(x) and numpy.isfinite(y)
        ])
        intersection = polygon.intersection(
            line_string) if polygon else line_string
        if intersection:
            row: NDArray[numpy.void] = result[jx]
            row['pass_number'] = pass_index + 1
            row['first_time'], row['last_time'] = _get_time_bounds(
                lat_nadir[ix, :],
                pass_time[ix, :],
                intersection,
            )
            jx += 1

    return pandas.DataFrame(result[:jx])
