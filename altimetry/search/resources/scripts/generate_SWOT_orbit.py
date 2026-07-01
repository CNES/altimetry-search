"""
Generate SWOT orbit.nc from local L3 SSH database (geometry + timing)
====================================================================
Uses fcollections to query a local L3 SSH database for all passes over a list
of cycles. Requested cycles must be present in the file_path, and the first
cycle should contain all passes.
The Altimetry Downloader Aviso (https://cnes.github.io/altimetry-downloader-aviso/)
can be used to retrieve official data from Aviso.

For each pass, geometry (nadir, swath polygons, nadir line_string)
is taken from the first available cycle, while timing (start_time, end_time,
pass_time) is derived from the real product and averaged across the requested
cycles.

Each pass is queried only once, using the ``stack='CYCLES_PASSES'`` option so
that all requested cycles for a pass come back in a single dataset of shape:
    time      (cycle_number, num_lines)
    longitude (cycle_number, num_lines, num_pixels)
    latitude  (cycle_number, num_lines, num_pixels)

Not every pass is present in every cycle: the timing average is computed over
the cycles that actually contain the pass (>= 1).

Requires: fcollections, xarray, numpy

Usage:
        python generate_swot_orbit.py \
               --l3_db_path file_path \
               --cycles 478 479 482 559 577 \
               --num_passes 28 \
               --output SWOT_calval_orbit.nc

        python generate_swot_orbit.py \
                --l3_db_path file_path \
                --cycles 7 28 29 33 45 \
                --num_passes 584 \
                --output SWOT_science_orbit.nc
"""
import argparse
import logging
import pathlib
import warnings

from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3
import numpy
import xarray

logger = logging.getLogger(__name__)

# Fixed dimensions (property of SWOT satellite altitude ~857 km)
NUM_LINES = 9860  # along-track points at full resolution per half-orbit
NUM_POINTS = 345  # sub-sampled points for polygons and line_string
L3_VERSION = '3.0'
L3_SUBSET = 'Expert'

# NetCDF compression
_COMPRESS = {'zlib': True, 'complevel': 4, 'shuffle': True}


# ───────────────────────────────────────────────────────────────────────────────
# 1. Geometry calculation
# ───────────────────────────────────────────────────────────────────────────────
def _resample(arr: numpy.ndarray, x_orig: numpy.ndarray,
              x_new: numpy.ndarray) -> numpy.ndarray:
    """Linearly interpolate ``arr`` (sampled at ``x_orig``) onto ``x_new``.

    Args:
        arr: Values to interpolate.
        x_orig: Sample positions of ``arr`` (monotonically increasing).
        x_new: Positions to interpolate onto.

    Returns:
        ``arr`` resampled at ``x_new``.
    """
    return numpy.interp(x_new, x_orig, arr)


def _swath_polygon(swath_lon, swath_lat, x_orig, col_hi, col_lo):
    """Build a closed swath-edge polygon from two columns of the L3 grid.

    The polygon runs along the higher-index column then back along the
    lower-index column, sub-sampled to ``NUM_POINTS`` points and closed on
    itself. Passing ``col_hi > col_lo`` for both swaths guarantees a consistent
    (clockwise) winding, which ``geographic.algorithms.intersection`` interprets
    as the swath interior (a counter-clockwise ring would be read as its
    complement).

    Args:
        swath_lon: Longitudes, shape (num_lines, num_pixels).
        swath_lat: Latitudes, shape (num_lines, num_pixels).
        x_orig: Normalised along-track positions of the input rows.
        col_hi: Higher pixel-column index of the swath edge.
        col_lo: Lower pixel-column index of the swath edge.

    Returns:
        ``(polygon_lon, polygon_lat)`` as float32 arrays of length ``NUM_POINTS``.
    """
    n_half = NUM_POINTS // 2
    x_half = numpy.linspace(0.0, 1.0, n_half)

    outer_lon = _resample(swath_lon[:, col_hi], x_orig, x_half)
    outer_lat = _resample(swath_lat[:, col_hi], x_orig, x_half)
    inner_lon = _resample(swath_lon[:, col_lo], x_orig, x_half)
    inner_lat = _resample(swath_lat[:, col_lo], x_orig, x_half)

    outer_lon = outer_lon[::-1]
    outer_lat = outer_lat[::-1]
    inner_lon = inner_lon[::-1]
    inner_lat = inner_lat[::-1]

    p_lon = numpy.concatenate([outer_lon, inner_lon[::-1], [outer_lon[0]]])
    p_lat = numpy.concatenate([outer_lat, inner_lat[::-1], [outer_lat[0]]])

    return p_lon.astype(numpy.float32), p_lat.astype(numpy.float32)


def extract_geometry(lon: numpy.ndarray, lat: numpy.ndarray) -> dict:
    """Derive nadir track, swath polygons and nadir line_string from a grid.

    The nadir is the middle pixel column. Nadir arrays are padded to
    ``NUM_LINES``; polygons and the line_string are sub-sampled to
    ``NUM_POINTS``. The line_string is resampled in [0, 360] before being
    converted back to signed longitudes, to avoid interpolation artefacts
    across the +-180 degree jump.

    Args:
        lon: Longitudes in [-180, 180], shape (num_lines, num_pixels).
        lat: Latitudes, shape (num_lines, num_pixels).

    Returns:
        Dict of geometry arrays: ``lon_nadir``, ``lat_nadir``,
        ``line_string_lon``/``line_string_lat``,
        ``left_polygon_lon``/``left_polygon_lat`` and
        ``right_polygon_lon``/``right_polygon_lat``.
    """
    n_lines, n_pixels = lon.shape
    nadir_idx = n_pixels // 2

    lon_nadir_raw = lon[:, nadir_idx].astype(numpy.float32)
    lat_nadir_raw = lat[:, nadir_idx].astype(numpy.float32)

    lon_nadir = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float32)
    lat_nadir = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float32)
    n_fill = min(n_lines, NUM_LINES)
    lon_nadir[:n_fill] = lon_nadir_raw[:n_fill]
    lat_nadir[:n_fill] = lat_nadir_raw[:n_fill]

    x_orig = numpy.linspace(0.0, 1.0, n_lines)

    left_lon, left_lat = _swath_polygon(lon,
                                        lat,
                                        x_orig,
                                        col_hi=nadir_idx - 1,
                                        col_lo=0)
    right_lon, right_lat = _swath_polygon(lon,
                                          lat,
                                          x_orig,
                                          col_hi=n_pixels - 1,
                                          col_lo=nadir_idx)

    # LineString: resample the nadir in [0, 360] BEFORE going back to signed.
    # numpy.interp does linear interpolation: in [-180, 180] the +-180 jump
    # creates spurious values; in [0, 360] the crossing at 180 deg is continuous.
    x_pts = numpy.linspace(0.0, 1.0, NUM_POINTS)
    lon_nadir_0360 = lon[:, nadir_idx].astype(numpy.float64) % 360.0
    ls_lon_0360 = _resample(lon_nadir_0360, x_orig, x_pts)
    ls_lon = numpy.where(ls_lon_0360 > 180, ls_lon_0360 - 360,
                         ls_lon_0360).astype(numpy.float32)
    ls_lat = _resample(lat_nadir_raw, x_orig, x_pts).astype(numpy.float32)

    return {
        'lon_nadir': lon_nadir,
        'lat_nadir': lat_nadir,
        'line_string_lon': ls_lon,
        'line_string_lat': ls_lat,
        'left_polygon_lon': left_lon,
        'left_polygon_lat': left_lat,
        'right_polygon_lon': right_lon,
        'right_polygon_lat': right_lat,
    }


# ───────────────────────────────────────────────────────────────────────────────
# 2. Query one pass over all cycles (single query, stacked)
# ───────────────────────────────────────────────────────────────────────────────
def query_pass_stacked(db, cycles, pass_number):
    """Query one pass over several cycles in a single stacked dataset.

    Uses the database ``stack='CYCLES_PASSES'`` option so every requested cycle
    comes back at once. Only cycles that actually contain the pass appear in the
    result.

    Args:
        db: L3 SSH database instance.
        cycles: Cycle numbers to query.
        pass_number: Pass (half-orbit) number.

    Returns:
        Tuple ``(lon, lat, time, cycle_numbers)``:
            lon, lat: shape (n_cyc, num_lines, num_pixels); longitude in [0, 360].
            time: shape (n_cyc, num_lines), datetime64[ns].
            cycle_numbers: shape (n_cyc,); cycles actually present.
    """
    ds = db.query(
        cycle_number=list(cycles),
        pass_number=pass_number,
        selected_variables=['time', 'longitude', 'latitude'],
        version=L3_VERSION,
        subset=L3_SUBSET,
        stack='CYCLES_PASSES',
    )

    # Drop the size-1 pass_number dimension if it is carried on the data vars.
    if 'pass_number' in ds['longitude'].dims:
        ds = ds.isel(pass_number=0)

    lon = numpy.asarray(ds['longitude'].values)
    lat = numpy.asarray(ds['latitude'].values)
    time = numpy.asarray(ds['time'].values).astype('datetime64[ns]')
    cycnums = numpy.atleast_1d(numpy.asarray(
        ds['cycle_number'].values)).astype(int)

    # A single-cycle stacked query drops the leading cycle axis. Restore it so
    # the rest of the pipeline can always index cycles along axis 0.
    if lon.ndim == 2:
        lon = lon[None]
        lat = lat[None]
    if time.ndim == 1:
        time = time[None]

    return lon, lat, time, cycnums


# ───────────────────────────────────────────────────────────────────────────────
# 4. Timing from the real `time` variable, averaged over cycles
# ───────────────────────────────────────────────────────────────────────────────
def cycle_start_from_pass1(time2d: numpy.ndarray,
                           cycnums: numpy.ndarray) -> dict:
    """Map each cycle number to its start time, derived from pass 1.

    The cycle start is the first valid time of pass 1 (the first half-orbit) in
    that cycle. Values are kept as datetime64[ns] so that later subtractions
    stay exact.

    Args:
        time2d: Pass-1 line times, shape (n_cyc, num_lines), datetime64[ns].
        cycnums: Cycle numbers matching the rows of ``time2d``.

    Returns:
        Dict ``{cycle_number: start_time}`` (datetime64[ns]); cycles whose
        pass 1 is entirely NaT are omitted.
    """
    cmap = {}
    for i, cyc in enumerate(cycnums):
        row = time2d[i]
        valid = ~numpy.isnat(row)
        if valid.any():
            cmap[int(cyc)] = row[valid][0]  # datetime64[ns]
    return cmap


def _nanmean(arr: numpy.ndarray, axis=0) -> numpy.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return numpy.nanmean(arr, axis=axis)


def average_timing(time2d: numpy.ndarray, cycnums: numpy.ndarray,
                   cycle_start_map: dict):
    """Average pass timing over cycles, anchored on the start of the cycle.

    For each cycle the line times are expressed relative to that cycle's start,
    then averaged:

        pass_time[line] = mean_cycle(time[line] - cycle_start)

    ``pass_time`` is therefore the time of each line since the beginning of the
    cycle;
    ``start_time``/``end_time`` are the first/last finite ``pass_time``,
    which guarantees by construction that a pass's last valid
    ``pass_time`` equals its ``end_time``.

    Args:
        time2d: Line times, shape (n_cyc, num_lines), datetime64[ns].
        cycnums: Cycle numbers matching the rows of ``time2d``.
        cycle_start_map: ``{cycle_number: start_time}`` from
            :func:`cycle_start_from_pass1`; cycles missing from it are skipped.

    Returns:
        Tuple ``(pass_time_ns, start, end)``:
            pass_time_ns: float array of length ``NUM_LINES`` (ns, NaN-padded).
            start, end: numpy.timedelta64[ns], or NaT if no cycle could be
                anchored to its start.
    """
    n_cyc, n_lines = time2d.shape

    rel = numpy.full((n_cyc, n_lines), numpy.nan)
    for i, cyc in enumerate(cycnums):
        c0 = cycle_start_map.get(int(cyc))
        if c0 is None:
            continue  # cannot anchor this cycle -> skip it
        delta = time2d[i] - c0  # timedelta64[ns], NaT where time NaT
        rel[i] = numpy.where(numpy.isnat(delta), numpy.nan,
                             delta.astype('int64').astype(numpy.float64))

    pass_time = _nanmean(rel, axis=0)  # (n_lines,)
    pass_time_full = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float64)
    n_fill = min(n_lines, NUM_LINES)
    pass_time_full[:n_fill] = pass_time[:n_fill]

    # start/end = first/last finite pass_time -> self-consistent with pass_time.
    finite = numpy.flatnonzero(numpy.isfinite(pass_time_full))
    if finite.size:
        start = numpy.timedelta64(int(round(pass_time_full[finite[0]])), 'ns')
        end = numpy.timedelta64(int(round(pass_time_full[finite[-1]])), 'ns')
    else:
        start = numpy.timedelta64('NaT', 'ns')
        end = numpy.timedelta64('NaT', 'ns')

    return pass_time_full, start, end


# ───────────────────────────────────────────────────────────────────────────────
# 5. Per-pass record (geometry + averaged timing)
# ───────────────────────────────────────────────────────────────────────────────
def get_geometry(lon, lat) -> dict:
    """Extract pass geometry.

    Thelongitudes are converted from [0, 360] to signed before
    being passed to :func:`extract_geometry`.

    Args:
        lon, lat: ``(lon, lat)`` from
            :func:`query_pass_stacked`.

    Returns:
        The geometry dict produced by :func:`extract_geometry`.
    """
    lon_signed = numpy.where(lon > 180, lon - 360, lon)
    return extract_geometry(lon_signed, lat)


def get_average_timing(time, cycnums, cycle_start_map) -> dict:
    """Average pass timing over the queried cycles.

    Thin adapter over :func:`average_timing`:
    packages the averaged times, plus the number of contributing cycles, into
    a record.

    Args:
        time, cycnums: ``(time, cycnums)`` from
            :func:`query_pass_stacked`.
        cycle_start_map: ``{cycle_number: start_time}`` for timing anchoring.

    Returns:
        Dict with ``ptime_ns``, ``start``, ``end`` and ``n_cycles`` (number of
        cycles that contained the pass).
    """
    ptime, start, end = average_timing(time, cycnums, cycle_start_map)
    return {
        'ptime_ns': ptime,
        'start': start,
        'end': end,
        'n_cycles': len(cycnums)
    }


# ───────────────────────────────────────────────────────────────────────────────
# 6. Collect all passes from the local L3 database
# ───────────────────────────────────────────────────────────────────────────────


def collect_all_passes(db,
                       cycles,
                       num_passes,
                       cycle_start_map,
                       pass1_query=None):
    """Build a record for every pass 1..num_passes, querying each once.

    Args:
        db: L3 SSH database instance.
        cycles: Cycle numbers to average over.
        num_passes: Total number of half-orbits per cycle.
        cycle_start_map: ``{cycle_number: start_time}`` for timing anchoring.
        pass1_query: Cached :func:`query_pass_stacked` result for pass 1, reused
            to avoid querying pass 1 twice (it also anchors the cycle starts).

    Returns:
        List of per-pass records, in pass order.
    """
    records = []
    logger.info('Reading %d passes over cycles %s', num_passes, list(cycles))

    for pass_number in range(1, num_passes + 1):
        # Reuse the pass-1 query made to establish the cycle reference.
        if pass_number == 1 and pass1_query is not None:
            lon, lat, time, cycnums = pass1_query
        else:
            lon, lat, time, cycnums = query_pass_stacked(
                db, cycles, pass_number)

        rec = {
            **get_geometry(lon[0], lat[0]),
            **get_average_timing(time, cycnums, cycle_start_map)
        }
        records.append(rec)

        if pass_number % 10 == 0 or pass_number == num_passes:
            logger.info('  %d/%d passes', pass_number, num_passes)

    missing = sum(1 for r in records if r['n_cycles'] == 0)
    logger.info('Done: %d/%d passes with data (%d missing)',
                num_passes - missing, num_passes, missing)
    return records


# ───────────────────────────────────────────────────────────────────────────────
# 7. Build xarray Dataset
# ───────────────────────────────────────────────────────────────────────────────


def build_dataset(records, pass_numbers):
    """Assemble per-pass geometry and timing into the orbit Dataset.

    Args:
        records: Per-pass records from :func:`collect_all_passes`.
        pass_numbers: Pass numbers labelling the ``num_passes`` dimension.

    Returns:
        The orbit :class:`xarray.Dataset` (nadir, swath polygons, line_string
        and start/end/pass times).
    """

    def stack(key) -> numpy.ndarray:
        """Stack field ``key`` across all records into a single array."""
        return numpy.stack([r[key] for r in records])

    num_passes = numpy.array(list(pass_numbers), dtype=numpy.int32)
    start_time = numpy.array([r['start'] for r in records],
                             dtype='timedelta64[ns]')
    end_time = numpy.array([r['end'] for r in records],
                           dtype='timedelta64[ns]')

    return xarray.Dataset(
        {
            'start_time':
            (['num_passes'], start_time, {
                'long_name': 'Start of half-orbit since beginning of cycle'
            }),
            'end_time':
            (['num_passes'], end_time, {
                'long_name': 'End of half-orbit since beginning of cycle'
            }),
            'pass_time':
            xarray.Variable(
                ['num_passes', 'num_lines'], stack('ptime_ns'), {
                    'units': 'nanoseconds',
                    'long_name': 'Time along pass since beginning of cycle'
                }),
            'lon_nadir': (['num_passes', 'num_lines'], stack('lon_nadir'), {
                'units': 'degrees_east'
            }),
            'lat_nadir': (['num_passes', 'num_lines'], stack('lat_nadir'), {
                'units': 'degrees_north'
            }),
            'line_string_lon':
            (['num_passes', 'num_points'], stack('line_string_lon'), {
                'units': 'degrees_east'
            }),
            'line_string_lat':
            (['num_passes', 'num_points'], stack('line_string_lat'), {
                'units': 'degrees_north'
            }),
            'left_polygon_lon':
            (['num_passes', 'num_points'], stack('left_polygon_lon'), {
                'units': 'degrees_east'
            }),
            'left_polygon_lat':
            (['num_passes', 'num_points'], stack('left_polygon_lat'), {
                'units': 'degrees_north'
            }),
            'right_polygon_lon':
            (['num_passes', 'num_points'], stack('right_polygon_lon'), {
                'units': 'degrees_east'
            }),
            'right_polygon_lat':
            (['num_passes', 'num_points'], stack('right_polygon_lat'), {
                'units': 'degrees_north'
            }),
        },
        coords={
            'num_passes': (['pass_number'], num_passes, {
                'long_name': 'Half-orbit number within cycle'
            }),
        },
    )


def _encoding(ds: xarray.Dataset) -> dict:
    """Build per-variable NetCDF encoding (compression, dtypes, fill values).

    Timedelta variables are stored as int64 nanoseconds with a NaT sentinel as
    fill value; floating variables use NaN. Every variable is zlib-compressed.

    Args:
        ds: Dataset to encode.

    Returns:
        ``{variable_name: encoding_dict}`` for :meth:`xarray.Dataset.to_netcdf`.
    """
    enc = {}
    for name, var in ds.data_vars.items():
        e = dict(_COMPRESS)
        if numpy.issubdtype(var.dtype, numpy.timedelta64):
            e['dtype'] = 'int64'
            e['units'] = 'nanoseconds'
            e['_FillValue'] = numpy.iinfo('int64').min  # NaT sentinel
        else:
            e['dtype'] = var.dtype
            if numpy.issubdtype(var.dtype, numpy.floating):
                e['_FillValue'] = numpy.nan
        enc[name] = e
    return enc


# ───────────────────────────────────────────────────────────────────────────────
# 8. Main
# ───────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and write the orbit file.

    Reads geometry and timing for every pass from the local L3 database
    over a list of cycles, anchors timing on pass 1, builds the dataset
    and writes it to NetCDF.
    """
    parser = argparse.ArgumentParser(
        description='Generate orbit file from local L3 SSH database '
        '(geometry + timing averaged over cycles)')
    parser.add_argument('--l3_db_path',
                        type=pathlib.Path,
                        required=True,
                        help='Path to local L3 SSH database directory')
    parser.add_argument(
        '--cycles',
        type=int,
        nargs='+',
        required=True,
        help='List of cycle numbers to average over (e.g., --cycles 7 28 45)')
    parser.add_argument(
        '--num_passes',
        type=int,
        required=True,
        help='Total number of half-orbits per cycle (28 CalVal, 584 Science)')
    parser.add_argument('--output',
                        type=pathlib.Path,
                        default=pathlib.Path('SWOT_orbit.nc'),
                        help='Output NetCDF file (default: SWOT_orbit.nc)')
    args = parser.parse_args()

    # ── Initialize database ──────────────────────────────────────────────────
    logger.info('[0/3] Initializing L3 database: %s', args.l3_db_path)
    if not args.l3_db_path.exists():
        raise FileNotFoundError(
            f'L3 database path not found: {args.l3_db_path}')
    db = NetcdfFilesDatabaseSwotLRL3(str(args.l3_db_path),
                                     follow_symlinks=True)
    logger.info('      Database ready')

    # ── Establish the per-cycle reference time from pass 1 ───────────────────
    # pass 1 is the first half-orbit, so its start defines the cycle start.
    logger.info('[1/3] Querying pass 1 to anchor cycle start times…')
    pass1_query = query_pass_stacked(db, args.cycles, pass_number=1)
    _, _, time1, cyc1 = pass1_query
    cycle_start_map = cycle_start_from_pass1(time1, cyc1)
    logger.info('      Cycle start anchored on %d cycle(s)',
                len(cycle_start_map))

    # ── Loop over all passes (one query per pass) ────────────────────────────
    logger.info('[2/3] Querying local L3 database for geometry + timing…')
    records = collect_all_passes(db,
                                 args.cycles,
                                 args.num_passes,
                                 cycle_start_map,
                                 pass1_query=pass1_query)

    # ── Build and write ──────────────────────────────────────────────────────
    logger.info(
        '[3/3] Building Dataset and writing NetCDF (zlib complevel=4)…')
    ds = build_dataset(records, range(1, args.num_passes + 1))
    ds.to_netcdf(args.output, encoding=_encoding(ds))

    size_mb = args.output.stat().st_size / 1024 / 1024
    logger.info('Written: %s (%.1f MB)', args.output, size_mb)
    logger.debug('Dataset:\n%s', ds)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    main()
