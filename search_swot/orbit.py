# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Calculate the ephemeredes of satellites."""
from __future__ import annotations

import typing as tp
import pathlib as pl

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pyinterp import TemporalAxis
import pyinterp.geodetic as py_geod
import xarray as xr

from .models import MissionProperties
from .orf import load_json


def get_cycle_duration(dataset: xr.Dataset) -> np.timedelta64:
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
        cycle_duration: np.timedelta64,
        mission_properties: MissionProperties) -> TemporalAxis:
    """Calculate the cycle axis.

    Args:
        cycle_duration: Duration of a cycle.
        mission_properties: Selected mission's properties.

    Returns:
        Temporal axis of the cycle.
    """
    orf_file = pl.Path(__file__).parent / mission_properties.orf_file
    cycles = load_json(orf_file.resolve())

    cycle_first_measurement = np.full(
        (200, ),
        np.datetime64('NAT'),
        dtype='M8[ns]',
    )
    keys = sorted(cycles)
    for item in keys:
        cycle_first_measurement[item - 1] = cycles[item]
    undefined = np.isnat(cycle_first_measurement)
    cycle_first_measurement[undefined] = np.full(
        (undefined.sum(), ), cycle_duration, dtype='m8[ns]') * np.arange(
            1, 1 + undefined.sum()) + cycles[keys[-1]]
    return TemporalAxis(cycle_first_measurement)


def get_selected_passes(
        mission_properties: MissionProperties,
        date: np.datetime64,
        search_duration: np.timedelta64 | None = None) -> pd.DataFrame:
    """Return the selected passes.

    Args:
        mission_properties: Selected mission's properties.
        date: Date of the first pass.
        search_duration: Duration of the search.

    Returns:
        Temporal axis of the selected passes.
    """
    orbit_file = pl.Path(__file__).parent / mission_properties.orbit_file
    with xr.open_dataset(orbit_file.resolve(), decode_timedelta=True) as ds:
        cycle_duration = get_cycle_duration(ds)
        search_duration = search_duration or cycle_duration
        axis = calculate_cycle_axis(cycle_duration, mission_properties)
        dates = np.array([date, date + search_duration])
        indices = axis.find_indexes(dates).ravel()
        cycle_numbers = np.repeat(
            np.arange(indices[0], indices[-1]) + 1,
            mission_properties.passes_per_cycle)
        axis_slice = axis[indices[0]:indices[-1] + 1]
        first_date_of_cycle = np.repeat(axis_slice,
                                        mission_properties.passes_per_cycle)
        pass_numbers = np.tile(
            np.arange(1, mission_properties.passes_per_cycle + 1),
            indices[-1] - indices[0])
        dates_of_selected_passes = np.vstack(
            (ds.start_time.values, ) * len(axis_slice)).T + axis_slice
        dates_of_selected_passes = dates_of_selected_passes.T.ravel()
        selected_passes = TemporalAxis(dates_of_selected_passes).find_indexes(
            dates).ravel()
        size = selected_passes[-1] - selected_passes[0]

        result: np.ndarray = np.ndarray((size, ),
                                        dtype=[('cycle_number', np.uint16),
                                               ('pass_number', np.uint16),
                                               ('first_measurement', 'M8[ns]'),
                                               ('last_measurement', 'M8[ns]')])
        axis_slice = slice(selected_passes[0], selected_passes[-1])
        result['cycle_number'] = cycle_numbers[axis_slice]
        result['pass_number'] = pass_numbers[axis_slice]
        result['first_measurement'] = first_date_of_cycle[axis_slice]
        result['last_measurement'] = first_date_of_cycle[axis_slice]
        return pd.DataFrame(result)


def _get_time_bounds(
    lat_nadir: NDArray,
    selected_time: NDArray,
    intersection: py_geod.LineString,
) -> tuple[np.datetime64, np.datetime64]:
    """Return the time bounds of the selected pass.

    Args:
        lat_nadir: Latitude of the nadir.
        selected_time: Time of the selected pass.
        intersection: Intersection of the pass with the polygon.

    Returns:
        Time bounds of the selected pass.
    """
    # Remove NaN values
    selected_time = selected_time[np.isfinite(lat_nadir)]
    lat_nadir = lat_nadir[np.isfinite(lat_nadir)]

    if lat_nadir[0] > lat_nadir[-1]:
        lat_nadir = lat_nadir[::-1]
        selected_time = selected_time[::-1]

    y0 = intersection[0].lat
    y1 = intersection[len(intersection) -
                      1].lat if len(intersection) > 1 else y0
    t0 = np.searchsorted(lat_nadir, y0)
    t1 = np.searchsorted(lat_nadir, y1)
    bounds = (
        selected_time[min(t0, t1)],
        selected_time[max(t0, t1)],
    )
    return min(bounds), max(bounds)


def get_pass_passage_time(mission_properties: MissionProperties,
                          selected_passes: pd.DataFrame,
                          polygon: py_geod.Polygon | None) -> pd.DataFrame:
    """Return the passage time of the selected passes.

    Args:
        mission_properties: Selected mission's properties.
        selected_passes: Selected passes.
        polygon: Polygon used to select the passes.

    Returns:
        Passage time of the selected passes.
    """
    passes = np.array(sorted(set(selected_passes['pass_number']))) - 1
    orbit_file = pl.Path(__file__).parent / mission_properties.orbit_file
    with xr.open_dataset(orbit_file.resolve(), decode_timedelta=True) as ds:
        lon = ds.line_string_lon.values[passes, :]
        lat = ds.line_string_lat.values[passes, :]
        pass_time = ds.pass_time.values[passes, :]
        lat_nadir = ds.lat_nadir.values[passes, :]

    result: NDArray[np.void] = np.ndarray(
        (len(passes), ),
        dtype=[('pass_number', np.uint16), ('first_time', 'm8[ns]'),
               ('last_time', 'm8[ns]')],
    )

    jx = 0

    for ix, pass_index in enumerate(passes):
        line_string = py_geod.LineString([
            py_geod.Point(x, y) for x, y in zip(lon[ix, :], lat[ix, :])
            if np.isfinite(x) and np.isfinite(y)
        ])
        intersection = polygon.intersection(
            line_string) if polygon else line_string
        if intersection:
            row: NDArray[np.void] = result[jx]
            row['pass_number'] = pass_index + 1
            row['first_time'], row['last_time'] = _get_time_bounds(
                lat_nadir[ix, :],
                pass_time[ix, :],
                intersection,
            )
            jx += 1

    return pd.DataFrame(result[:jx])
