"""
Generate SWOT orbit.nc from local L3 SSH database + ephemeris timing
====================================================================
Uses fcollections to query a local L3 SSH database for all passes in a cycle,
extracts swath geometry, and combines with ephemeris timing.

Requires: fcollections, pyinterp, xarray, netcdf4, numpy

Usage:    python generate_orbit_from_fcollections.py \
              --l3_db_path /path/to/SWOT_L3_LR_SSH \
              --cycle 28 \
              --ephemeris ephemeris.txt \
              --output SWOT_orbit.nc
"""
import argparse
import pathlib

from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3
import numpy
import pyinterp.orbit
import xarray

# Fixed dimensions (property of SWOT satellite altitude ~857 km)
NUM_LINES = 9860  # along-track points at full resolution per half-orbit
NUM_POINTS = 345  # sub-sampled points for polygons and line_string
L3_VERSION = '3.0'
L3_SUBSET = 'Expert'

# NetCDF compression
_COMPRESS = {'zlib': True, 'complevel': 4, 'shuffle': True}

# ───────────────────────────────────────────────────────────────────────────────
# 1. Load ephemeris
# ───────────────────────────────────────────────────────────────────────────────


def load_ephemeris(
    path: pathlib.Path,
) -> tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray,
           numpy.timedelta64]:
    """Load a CNES text ephemeris file."""
    with open(path, encoding='utf-8') as fh:
        lines = fh.readlines()

    settings: dict[str, float] = {}
    for line in lines[:2]:
        key, value = line[1:].split('=')
        settings[key.strip()] = float(value)
    del lines[:2]

    eph = numpy.loadtxt(
        lines,
        delimiter=' ',
        dtype={
            'names': ('time', 'longitude', 'latitude', 'height'),
            'formats': ('f8', 'f8', 'f8', 'f8')
        },
    )
    cycle_duration = numpy.timedelta64(
        int(settings['cycle_duration'] * 86_400 * 1e9), 'ns')
    return (
        settings['height'],
        eph['longitude'],
        eph['latitude'],
        eph['time'].astype('timedelta64[s]'),
        cycle_duration,
    )


# ───────────────────────────────────────────────────────────────────────────────
# 2. Resample utilities
# ───────────────────────────────────────────────────────────────────────────────


def _resample(arr: numpy.ndarray, x_orig: numpy.ndarray,
              x_new: numpy.ndarray) -> numpy.ndarray:
    """1-D linear interpolation."""
    return numpy.interp(x_new, x_orig, arr)


def _swath_polygon(swath_lon,
                   swath_lat,
                   x_orig,
                   col_outer,
                   col_inner,
                   reverse=True):
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

    outer_lon = outer_lon[::-1]
    outer_lat = outer_lat[::-1]
    inner_lon = inner_lon[::-1]
    inner_lat = inner_lat[::-1]

    p_lon = numpy.concatenate([outer_lon, inner_lon[::-1], [outer_lon[0]]])
    p_lat = numpy.concatenate([outer_lat, inner_lat[::-1], [outer_lat[0]]])

    if reverse:
        p_lon = p_lon[::-1]
        p_lat = p_lat[::-1]

    return p_lon.astype(numpy.float32), p_lat.astype(numpy.float32)


# ───────────────────────────────────────────────────────────────────────────────
# 3. Read one pass from local L3 database and extract geometry
# ───────────────────────────────────────────────────────────────────────────────


def read_l3_pass_from_db(
    db: NetcdfFilesDatabaseSwotLRL3,
    cycle_number: int,
    pass_number: int,
) -> dict[str, numpy.ndarray]:
    """Query local L3 SSH database for one pass and extract geometry.

    Args:
        db: NetcdfFilesDatabaseSwotLRL3 instance
        cycle_number: cycle number (e.g., 28)
        pass_number: pass number (1-based, 1..28 for CalVal or 1..584 for Science)

    Returns:
        Dictionary with geometry data for the pass.
    """
    # Query the local database
    try:
        data = db.query(cycle_number=cycle_number,
                        pass_number=pass_number,
                        selected_variables=['time', 'longitude', 'latitude'],
                        version=L3_VERSION,
                        subset=L3_SUBSET)
    except Exception as e:
        raise RuntimeError(
            f'L3 database query failed for cycle {cycle_number}, pass {pass_number}: {e}'
        )

    lon_raw = data[
        'longitude'].values  # ← conserver [0, 360] pour la line_string
    lat = data['latitude'].values
    n_lines, n_pixels = lon_raw.shape

    # ── Conversion [0, 360] → [-180, 180] pour polygones et nadir ────────────
    lon = numpy.where(lon_raw > 180, lon_raw - 360, lon_raw)

    nadir_idx = n_pixels // 2
    lon_nadir_raw = lon[:, nadir_idx].astype(numpy.float32)
    lat_nadir_raw = lat[:, nadir_idx].astype(numpy.float32)

    lon_nadir = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float32)
    lat_nadir = numpy.full(NUM_LINES, numpy.nan, dtype=numpy.float32)
    lon_nadir[:n_lines] = lon_nadir_raw
    lat_nadir[:n_lines] = lat_nadir_raw

    half = n_pixels // 2
    x_orig = numpy.linspace(0.0, 1.0, n_lines)

    left_lon, left_lat = _swath_polygon(lon,
                                        lat,
                                        x_orig,
                                        col_outer=0,
                                        col_inner=half - 1,
                                        reverse=True)
    right_lon, right_lat = _swath_polygon(lon,
                                          lat,
                                          x_orig,
                                          col_outer=-1,
                                          col_inner=half,
                                          reverse=False)

    # ── Line_string : rééchantillonner en [0, 360] AVANT la conversion ───────
    # numpy.interp interpole linéairement : en [-180, 180] le saut ±180° crée
    # des valeurs parasites ; en [0, 360] le passage par 180° est continu.
    x_pts = numpy.linspace(0.0, 1.0, NUM_POINTS)
    ls_lon_0360 = _resample(lon_raw[:, nadir_idx].astype(numpy.float64),
                            x_orig, x_pts)
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
        'n_valid_lines': n_lines,
    }


# ───────────────────────────────────────────────────────────────────────────────
# 4. Get timing from ephemeris for all passes in a cycle
# ───────────────────────────────────────────────────────────────────────────────


def compute_pass_timing(orbit: pyinterp.orbit.Orbit, ) -> list[dict]:
    """Calcule start_time, end_time, pass_time pour toutes les passes du cycle.

    Utilise swath.time issu de l'éphéméride pyinterp,
    en nanosecondes depuis le début du cycle.
    Retourne un enregistrement par passe (1 → n_passes_total),
    indexable par timing_records[pass_number - 1].
    """
    n_passes = orbit.passes_per_cycle()
    records = []

    for pass_number in range(1, n_passes + 1):
        pass_data = pyinterp.orbit.calculate_pass(pass_number, orbit)
        if pass_data is None:
            raise RuntimeError(f'Pass {pass_number} returned None from orbit')

        swath = pyinterp.orbit.calculate_swath(pass_data,
                                               half_swath=60.0,
                                               half_gap=10.0)

        n_orig = len(swath.time)
        t_ns = swath.time.astype('timedelta64[ns]').astype(numpy.float64)

        # Resample pass_time to NUM_LINES if needed
        if n_orig != NUM_LINES:
            print(
                f'  ! Pass {pass_number} : n_orig={n_orig} ≠ NUM_LINES={NUM_LINES}, rééchantillonnage'
            )
            x_orig = numpy.linspace(0.0, 1.0, n_orig)
            x_full = numpy.linspace(0.0, 1.0, NUM_LINES)
            ptime_ns = _resample(t_ns, x_orig, x_full)
        else:
            ptime_ns = t_ns

        records.append({
            'start': swath.time[0],
            'end': swath.time[-1],
            'ptime_ns': ptime_ns,
        })

    return records


# ───────────────────────────────────────────────────────────────────────────────
# 5. Collect all passes from local L3 database
# ───────────────────────────────────────────────────────────────────────────────


def collect_all_passes(db, cycle_number, timing_records, n_passes):
    pass_list = range(1, n_passes + 1)
    records = []
    print(f'Reading {len(pass_list)} passes (cycle {cycle_number})...')

    for i, pass_number in enumerate(pass_list):
        try:
            geom = read_l3_pass_from_db(db, cycle_number, pass_number)
            timing = timing_records[pass_number - 1]

            # Aligner les NaN de pass_time sur ceux de lon_nadir
            n_valid = geom.pop('n_valid_lines')
            ptime_ns = timing['ptime_ns'].copy()
            ptime_ns[n_valid:] = numpy.nan

            records.append({**geom, **timing, 'ptime_ns': ptime_ns})

            if (i + 1) % 10 == 0 or (i + 1) == len(pass_list):
                print(f'  {i+1}/{len(pass_list)} passes')

        except Exception as e:
            print(f'  X Pass {pass_number}: {e}')
            if pass_number <= pass_list[0] + 4:
                raise

    print(f'Done : {len(records)}/{len(pass_list)} passes')
    return records


# ───────────────────────────────────────────────────────────────────────────────
# 6. Build xarray Dataset
# ───────────────────────────────────────────────────────────────────────────────


def build_dataset(records, pass_numbers):
    """Assemble geometry + timing into the orbit Dataset."""

    def stack(key) -> numpy.ndarray:
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
                ['num_passes', 'num_lines'],
                stack('ptime_ns'),
                {
                    'units': 'nanoseconds',  # ← ns
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
    enc = {}
    for name, var in ds.data_vars.items():
        e = dict(_COMPRESS)
        if numpy.issubdtype(var.dtype, numpy.timedelta64):
            e['dtype'] = 'int64'
            e['units'] = 'nanoseconds'
        else:
            e['dtype'] = var.dtype
            if numpy.issubdtype(var.dtype, numpy.floating):
                e['_FillValue'] = numpy.nan
        enc[name] = e  # ← dans la boucle
    return enc


# ───────────────────────────────────────────────────────────────────────────────
# 7. Main
# ───────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate orbit file from local L3 SSH database + ephemeris'
    )
    parser.add_argument(
        '--l3_db_path',
        type=pathlib.Path,
        required=True,
        help=
        'Path to local L3 SSH database directory (e.g., /data/SWOT_L3_LR_SSH)')
    parser.add_argument(
        '--cycle',
        type=int,
        required=True,
        help='Cycle number (e.g., 28 for CalVal, 1 for Science cycle 001)')
    parser.add_argument('--ephemeris',
                        type=pathlib.Path,
                        required=True,
                        help='Ephemeris text file for timing data')
    parser.add_argument('--output',
                        type=pathlib.Path,
                        default=pathlib.Path('SWOT_orbit.nc'),
                        help='Output NetCDF file (default: SWOT_orbit.nc)')
    args = parser.parse_args()

    # Create fcollections database object from local path
    print(f'[0/4] Initializing L3 database: {args.l3_db_path}')
    if not args.l3_db_path.exists():
        raise FileNotFoundError(
            f'L3 database path not found: {args.l3_db_path}')

    db = NetcdfFilesDatabaseSwotLRL3(str(args.l3_db_path),
                                     follow_symlinks=True)
    print(f'      Database ready')

    print(f'[1/4] Loading ephemeris: {args.ephemeris}')
    height, lon, lat, time, cycle_duration = load_ephemeris(args.ephemeris)
    print(f'      height={height:.0f} m  |  cycle_duration={cycle_duration}')

    print('[2/4] Computing orbit timing from ephemeris…')
    orbit = pyinterp.orbit.calculate_orbit(height, lon, lat, time,
                                           cycle_duration)
    n_passes = orbit.passes_per_cycle()
    print(f'      {n_passes} half-orbits per cycle')
    timing_records = compute_pass_timing(orbit)

    print(f'[3/4] Querying local L3 database for geometry…')
    records = collect_all_passes(db, args.cycle, timing_records, n_passes)

    print('[4/4] Building Dataset and writing NetCDF (zlib complevel=4)…')
    ds = build_dataset(records, range(1, len(records) + 1))
    ds.to_netcdf(args.output, encoding=_encoding(ds))

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f'\n✓  Written: {args.output}  ({size_mb:.1f} MB)\n')
    print(ds)


if __name__ == '__main__':
    main()
