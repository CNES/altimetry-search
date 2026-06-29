"""
Generate SWOT orbit.nc from local L3 SSH database (geometry + timing)
====================================================================
Uses fcollections to query a local L3 SSH database for all passes over a list
of cycles. For each pass, geometry (nadir, swath polygons, nadir line_string)
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
               --l3_db_path /work/HELPDESK_SWOTLR/commun/data/aviso/swot_products/l3_karin_nadir/l3_lr_ssh/ \
               --cycles 7 28 45 \
               --num_passes 28 \
               --output SWOT_calval_orbit.nc

        python generate_swot_orbit.py \
                --l3_db_path /work/HELPDESK_SWOTLR/commun/data/aviso/swot_products/l3_karin_nadir/l3_lr_ssh/ \
                --cycles 7 8 9 \
                --num_passes 584 \
                --output SWOT_science_orbit.nc
"""
import argparse
import logging
import pathlib
import warnings

import numpy
import xarray

from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3

logger = logging.getLogger(__name__)

# Fixed dimensions (property of SWOT satellite altitude ~857 km)
NUM_LINES  = 9860   # along-track points at full resolution per half-orbit
NUM_POINTS = 345    # sub-sampled points for polygons and line_string
L3_VERSION = "3.0"
L3_SUBSET  = "Expert"
STACK_DIM  = "CYCLES_PASSES"

# NetCDF compression
_COMPRESS = {'zlib': True, 'complevel': 4, 'shuffle': True}


# ───────────────────────────────────────────────────────────────────────────────
# 1. Resample / averaging utilities
# ───────────────────────────────────────────────────────────────────────────────

def _resample(arr: numpy.ndarray, x_orig: numpy.ndarray,
              x_new: numpy.ndarray) -> numpy.ndarray:
    """1-D linear interpolation."""
    return numpy.interp(x_new, x_orig, arr)


def _nanmean(arr: numpy.ndarray, axis=0) -> numpy.ndarray:
    """nanmean that silently returns NaN on all-NaN slices (no warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return numpy.nanmean(arr, axis=axis)


def _swath_polygon(swath_lon, swath_lat, x_orig, col_outer, col_inner, reverse=True):
    # The left and right swaths are built from opposite edges of the swath
    # grid (col_outer=0 vs col_outer=-1), which naturally produces polygons
    # with opposite winding orders. geographic.algorithms.intersection
    # interprets a counter-clockwise polygon as the complement of its
    # interior, resulting in the bounding box minus the swath being displayed
    # instead of the swath itself. reverse=True corrects the winding order of
    # the left swath to match the right swath (clockwise).
    n_half = NUM_POINTS // 2
    x_half = numpy.linspace(0.0, 1.0, n_half)

    outer_lon = _resample(swath_lon[:, col_outer], x_orig, x_half)
    outer_lat = _resample(swath_lat[:, col_outer], x_orig, x_half)
    inner_lon = _resample(swath_lon[:, col_inner], x_orig, x_half)
    inner_lat = _resample(swath_lat[:, col_inner], x_orig, x_half)

    outer_lon = outer_lon[::-1]; outer_lat = outer_lat[::-1]
    inner_lon = inner_lon[::-1]; inner_lat = inner_lat[::-1]

    p_lon = numpy.concatenate([outer_lon, inner_lon[::-1], [outer_lon[0]]])
    p_lat = numpy.concatenate([outer_lat, inner_lat[::-1], [outer_lat[0]]])

    if reverse:
        p_lon = p_lon[::-1]
        p_lat = p_lat[::-1]

    return p_lon.astype(numpy.float32), p_lat.astype(numpy.float32)


# ───────────────────────────────────────────────────────────────────────────────
# 2. Query one pass over all cycles (single query, stacked)
# ───────────────────────────────────────────────────────────────────────────────

def query_pass_stacked(db, cycles, pass_number):
    """Query one pass over a list of cycles in a single stacked dataset.

    Returns (lon, lat, time, cycle_numbers):
        lon, lat : (n_cyc, num_lines, num_pixels)  -- longitude in [0, 360]
        time     : (n_cyc, num_lines)              -- datetime64[ns]
        cycles   : (n_cyc,)                         -- cycle numbers actually present
    """
    ds = db.query(
        cycle_number=list(cycles),
        pass_number=pass_number,
        selected_variables=['time', 'longitude', 'latitude'],
        version=L3_VERSION,
        subset=L3_SUBSET,
        stack=STACK_DIM,
    )

    # Drop the size-1 pass_number dimension if it is carried on the data vars.
    if 'pass_number' in ds['longitude'].dims:
        ds = ds.isel(pass_number=0)

    lon = numpy.asarray(ds['longitude'].values)
    lat = numpy.asarray(ds['latitude'].values)
    time = numpy.asarray(ds['time'].values).astype('datetime64[ns]')
    cycnums = numpy.atleast_1d(numpy.asarray(ds['cycle_number'].values)).astype(int)

    return lon, lat, time, cycnums


# ───────────────────────────────────────────────────────────────────────────────
# 3. Extract geometry from an averaged (num_lines, num_pixels) lon/lat grid
# ───────────────────────────────────────────────────────────────────────────────

def extract_geometry(lon: numpy.ndarray, lat: numpy.ndarray) -> dict:
    """Build nadir, swath polygons and nadir line_string from a lon/lat grid.

    ``lon`` must be in [-180, 180] (signed).
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

    half   = n_pixels // 2
    x_orig = numpy.linspace(0.0, 1.0, n_lines)

    left_lon,  left_lat  = _swath_polygon(lon, lat, x_orig, col_outer=0,  col_inner=half - 1, reverse=True)
    right_lon, right_lat = _swath_polygon(lon, lat, x_orig, col_outer=-1, col_inner=half,     reverse=False)

    # LineString: resample the nadir in [0, 360] BEFORE going back to signed.
    # numpy.interp does linear interpolation: in [-180, 180] the +-180 jump
    # creates spurious values; in [0, 360] the crossing at 180 deg is continuous.
    x_pts        = numpy.linspace(0.0, 1.0, NUM_POINTS)
    lon_nadir_0360 = lon[:, nadir_idx].astype(numpy.float64) % 360.0
    ls_lon_0360  = _resample(lon_nadir_0360, x_orig, x_pts)
    ls_lon       = numpy.where(ls_lon_0360 > 180,
                               ls_lon_0360 - 360,
                               ls_lon_0360).astype(numpy.float32)
    ls_lat       = _resample(lat_nadir_raw, x_orig, x_pts).astype(numpy.float32)

    return {
        'lon_nadir':         lon_nadir,
        'lat_nadir':         lat_nadir,
        'line_string_lon':   ls_lon,
        'line_string_lat':   ls_lat,
        'left_polygon_lon':  left_lon,
        'left_polygon_lat':  left_lat,
        'right_polygon_lon': right_lon,
        'right_polygon_lat': right_lat,
    }


# ───────────────────────────────────────────────────────────────────────────────
# 4. Timing from the real `time` variable, averaged over cycles
# ───────────────────────────────────────────────────────────────────────────────

def cycle_start_from_pass1(time2d: numpy.ndarray, cycnums: numpy.ndarray) -> dict:
    """Map cycle_number -> absolute start time of the cycle, taken as the first
    valid time of pass 1 in that cycle (the first half-orbit starts the cycle).
    Values are kept as datetime64[ns] so that later subtractions stay exact.
    """
    cmap = {}
    for i, cyc in enumerate(cycnums):
        row = time2d[i]
        valid = ~numpy.isnat(row)
        if valid.any():
            cmap[int(cyc)] = row[valid][0]   # datetime64[ns]
    return cmap


def average_timing(time2d: numpy.ndarray, cycnums: numpy.ndarray,
                   cycle_start_map: dict):
    """Average timing over cycles, anchored on the start of the cycle.

    pass_time[line] = mean_cycle( time[line] - cycle_start )
        -> time of each line since the beginning of the cycle. It does NOT
        restart at 0 for each pass: pass_time[0] == start_time.
    start_time = first finite pass_time, end_time = last finite pass_time.
        Deriving them from pass_time guarantees, by construction, that the last
        valid pass_time of a pass equals its end_time.

    The subtraction is done in datetime64 (integer) arithmetic before the cast
    to float, so the delta stays exact (a float cast of raw ns-since-epoch,
    ~1.7e18, would carry ~512 ns of rounding noise).

    Returns (pass_time_ns[NUM_LINES] float, start_td, end_td) where start/end
    are numpy.timedelta64[ns] (NaT if no cycle could be anchored to its start).
    """
    n_cyc, n_lines = time2d.shape

    rel = numpy.full((n_cyc, n_lines), numpy.nan)
    for i, cyc in enumerate(cycnums):
        c0 = cycle_start_map.get(int(cyc))
        if c0 is None:
            continue                      # cannot anchor this cycle -> skip it
        delta = time2d[i] - c0            # timedelta64[ns], NaT where time NaT
        rel[i] = numpy.where(numpy.isnat(delta),
                             numpy.nan,
                             delta.astype('int64').astype(numpy.float64))

    pass_time = _nanmean(rel, axis=0)                     # (n_lines,)
    pass_time_full = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float64)
    n_fill = min(n_lines, NUM_LINES)
    pass_time_full[:n_fill] = pass_time[:n_fill]

    # start/end = first/last finite pass_time -> self-consistent with pass_time.
    finite = numpy.flatnonzero(numpy.isfinite(pass_time_full))
    if finite.size:
        start = numpy.timedelta64(int(round(pass_time_full[finite[0]])),  'ns')
        end   = numpy.timedelta64(int(round(pass_time_full[finite[-1]])), 'ns')
    else:
        start = numpy.timedelta64('NaT', 'ns')
        end   = numpy.timedelta64('NaT', 'ns')

    return pass_time_full, start, end


# ───────────────────────────────────────────────────────────────────────────────
# 5. Per-pass record (averaged geometry + timing)
# ───────────────────────────────────────────────────────────────────────────────

def average_pass(query_result, cycle_start_map) -> dict:
    """Build one pass record: geometry from the first cycle, timing averaged."""
    lon, lat, time2d, cycnums = query_result

    # Geometry: first available cycle only (no averaging).
    lon0 = lon[0]                                   # (num_lines, num_pixels) in [0, 360]
    lat0 = lat[0]
    lon0_signed = numpy.where(lon0 > 180, lon0 - 360, lon0)

    geom = extract_geometry(lon0_signed, lat0)
    ptime, start, end = average_timing(time2d, cycnums, cycle_start_map)

    return {**geom, 'ptime_ns': ptime, 'start': start, 'end': end,
            'n_cycles': len(cycnums)}


# ───────────────────────────────────────────────────────────────────────────────
# 6. Collect all passes from the local L3 database
# ───────────────────────────────────────────────────────────────────────────────

def collect_all_passes(db, cycles, num_passes, cycle_start_map, pass1_query=None):
    """Loop over passes 1..num_passes, querying each once over all cycles."""
    records = []
    logger.info('Reading %d passes over cycles %s', num_passes, list(cycles))

    for pass_number in range(1, num_passes + 1):
        # Reuse the pass-1 query made to establish the cycle reference.
        if pass_number == 1 and pass1_query is not None:
            result = pass1_query
        else:
            result = query_pass_stacked(db, cycles, pass_number)

        rec = average_pass(result, cycle_start_map)
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
    """Assemble geometry + timing into the orbit Dataset."""
    def stack(key) -> numpy.ndarray:
        return numpy.stack([r[key] for r in records])

    num_passes = numpy.array(list(pass_numbers), dtype=numpy.int32)
    start_time = numpy.array([r['start'] for r in records], dtype='timedelta64[ns]')
    end_time   = numpy.array([r['end']   for r in records], dtype='timedelta64[ns]')

    return xarray.Dataset(
        {
            'start_time':       (['num_passes'], start_time,
                                 {'long_name': 'Start of half-orbit since beginning of cycle'}),
            'end_time':         (['num_passes'], end_time,
                                 {'long_name': 'End of half-orbit since beginning of cycle'}),
            'pass_time': xarray.Variable(
                ['num_passes', 'num_lines'], stack('ptime_ns'),
                {'units': 'nanoseconds',
                 'long_name': 'Time along pass since beginning of cycle'}),
            'lon_nadir':        (['num_passes', 'num_lines'], stack('lon_nadir'),
                                 {'units': 'degrees_east'}),
            'lat_nadir':        (['num_passes', 'num_lines'], stack('lat_nadir'),
                                 {'units': 'degrees_north'}),
            'line_string_lon':  (['num_passes', 'num_points'], stack('line_string_lon'),
                                 {'units': 'degrees_east'}),
            'line_string_lat':  (['num_passes', 'num_points'], stack('line_string_lat'),
                                 {'units': 'degrees_north'}),
            'left_polygon_lon': (['num_passes', 'num_points'], stack('left_polygon_lon'),
                                 {'units': 'degrees_east'}),
            'left_polygon_lat': (['num_passes', 'num_points'], stack('left_polygon_lat'),
                                 {'units': 'degrees_north'}),
            'right_polygon_lon':(['num_passes', 'num_points'], stack('right_polygon_lon'),
                                 {'units': 'degrees_east'}),
            'right_polygon_lat':(['num_passes', 'num_points'], stack('right_polygon_lat'),
                                 {'units': 'degrees_north'}),
        },
        coords={
            'num_passes': (['pass_number'], num_passes,
                           {'long_name': 'Half-orbit number within cycle'}),
        },
    )


def _encoding(ds: xarray.Dataset) -> dict:
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
    parser = argparse.ArgumentParser(
        description='Generate orbit file from local L3 SSH database '
                    '(geometry + timing averaged over cycles)')
    parser.add_argument(
        '--l3_db_path', type=pathlib.Path, required=True,
        help='Path to local L3 SSH database directory')
    parser.add_argument(
        '--cycles', type=int, nargs='+', required=True,
        help='List of cycle numbers to average over (e.g., --cycles 7 28 45)')
    parser.add_argument(
        '--num_passes', type=int, required=True,
        help='Total number of half-orbits per cycle (28 CalVal, 584 Science)')
    parser.add_argument(
        '--output', type=pathlib.Path,
        default=pathlib.Path('SWOT_orbit.nc'),
        help='Output NetCDF file (default: SWOT_orbit.nc)')
    args = parser.parse_args()

    # ── Initialize database ──────────────────────────────────────────────────
    logger.info('[0/3] Initializing L3 database: %s', args.l3_db_path)
    if not args.l3_db_path.exists():
        raise FileNotFoundError(f'L3 database path not found: {args.l3_db_path}')
    db = NetcdfFilesDatabaseSwotLRL3(str(args.l3_db_path), follow_symlinks=True)
    logger.info('      Database ready')

    # ── Establish the per-cycle reference time from pass 1 ───────────────────
    # pass 1 is the first half-orbit, so its start defines the cycle start.
    logger.info('[1/3] Querying pass 1 to anchor cycle start times…')
    pass1_query = query_pass_stacked(db, args.cycles, pass_number=1)
    _, _, time1, cyc1 = pass1_query
    cycle_start_map = cycle_start_from_pass1(time1, cyc1)
    logger.info('      Cycle start anchored on %d cycle(s)', len(cycle_start_map))

    # ── Loop over all passes (one query per pass) ────────────────────────────
    logger.info('[2/3] Querying local L3 database for geometry + timing…')
    records = collect_all_passes(
        db, args.cycles, args.num_passes, cycle_start_map, pass1_query=pass1_query)

    # ── Build and write ──────────────────────────────────────────────────────
    logger.info('[3/3] Building Dataset and writing NetCDF (zlib complevel=4)…')
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