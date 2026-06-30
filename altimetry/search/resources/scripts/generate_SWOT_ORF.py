"""
Generate a SWOT ORF (Orbit Reference File) from the local L3 SSH database
========================================================================
For every cycle in a range, reads the start time of pass 1 (the first
half-orbit, which defines the cycle start) from the local L3 SSH database and
writes a JSON mapping ``{cycle_number: cycle_start_iso}``.

Pass 1 of all requested cycles is fetched in a single stacked query
(``stack='CYCLES_PASSES'``). Cycles that do not contain pass 1 are simply
absent from the output.

NOTE: pass 1 is missing in the very first cycle of BOTH SWOT phases (cycle 1 of
the science phase and cycle 474 of the CalVal phase). That cycle therefore does
not appear in the generated ORF and its start time must be added by hand to the
output JSON.

Requires: fcollections, numpy

Usage:
        # Science phase
        python generate_swot_ORF.py \
               --l3-db-path /work/.../l3_lr_ssh/ \
               --cycle-start 1 --cycle-end 52 \
               --output SWOT_science_ORF.json

        # CalVal phase
        python generate_swot_ORF.py \
               --l3-db-path /work/.../l3_lr_ssh/ \
               --cycle-start 474 --cycle-end 578 \
               --output SWOT_calval_ORF.json
"""
import argparse
import json
import logging
import pathlib

import numpy

from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3

logger = logging.getLogger(__name__)

L3_VERSION = "3.0"
L3_SUBSET  = "Expert"
STACK_DIM  = "CYCLES_PASSES"
TIME_UNIT  = "ms"   # precision of the serialised ISO timestamps


# ───────────────────────────────────────────────────────────────────────────────
# 1. Query one pass over several cycles
# ───────────────────────────────────────────────────────────────────────────────

def query_pass_stacked(db, cycles, pass_number):
    """Query the line times of one pass over several cycles in a single dataset.

    Uses the database ``stack='CYCLES_PASSES'`` option so every requested cycle
    comes back at once. Only cycles that actually contain the pass appear in the
    result. Only ``time`` is requested, since the ORF needs nothing else.

    Args:
        db: L3 SSH database instance.
        cycles: Cycle numbers to query.
        pass_number: Pass (half-orbit) number.

    Returns:
        Tuple ``(time, cycle_numbers)``:
            time: shape (n_cyc, num_lines), datetime64[ns].
            cycle_numbers: shape (n_cyc,); cycles actually present.
    """
    ds = db.query(
        cycle_number=list(cycles),
        pass_number=pass_number,
        selected_variables=['time'],
        version=L3_VERSION,
        subset=L3_SUBSET,
        stack=STACK_DIM,
    )

    # Drop the size-1 pass_number dimension if it is carried on the data var.
    if 'pass_number' in ds['time'].dims:
        ds = ds.isel(pass_number=0)

    time = numpy.asarray(ds['time'].values).astype('datetime64[ns]')
    cycnums = numpy.atleast_1d(numpy.asarray(ds['cycle_number'].values)).astype(int)

    # A single-cycle stacked query drops the leading cycle axis. Restore it so
    # cycles can always be indexed along axis 0.
    if time.ndim == 1:
        time = time[None]

    return time, cycnums


# ───────────────────────────────────────────────────────────────────────────────
# 2. Cycle start times
# ───────────────────────────────────────────────────────────────────────────────

def cycle_start_from_pass1(time2d: numpy.ndarray, cycnums: numpy.ndarray) -> dict:
    """Map each cycle number to its start time, derived from pass 1.

    The cycle start is the first valid time of pass 1 (the first half-orbit) in
    that cycle. Values are kept as datetime64[ns].

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
            cmap[int(cyc)] = row[valid][0]   # datetime64[ns]
    return cmap


def to_serializable(cycle_start_map: dict) -> dict:
    """Convert ``{cycle: datetime64}`` to a JSON-ready ``{cycle: iso_string}``.

    Cycles are sorted and timestamps formatted to ``TIME_UNIT`` precision.

    Args:
        cycle_start_map: ``{cycle_number: start_time}`` (datetime64[ns]).

    Returns:
        Ordered dict ``{cycle_number: iso8601_string}``.
    """
    return {
        int(cyc): numpy.datetime_as_string(start, unit=TIME_UNIT)
        for cyc, start in sorted(cycle_start_map.items())
    }


# ───────────────────────────────────────────────────────────────────────────────
# 3. Main
# ───────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and write the ORF JSON.

    Reads the start time of pass 1 for every cycle in ``[cycle_start,
    cycle_end]`` from the local L3 database and serialises the result to JSON.
    """
    parser = argparse.ArgumentParser(
        description='Generate a SWOT ORF (cycle start times) from the local '
                    'L3 SSH database')
    parser.add_argument(
        '--l3-db-path', type=pathlib.Path, required=True,
        help='Path to local L3 SSH database directory')
    parser.add_argument(
        '--cycle-start', type=int, required=True,
        help='First cycle number (inclusive)')
    parser.add_argument(
        '--cycle-end', type=int, required=True,
        help='Last cycle number (inclusive)')
    parser.add_argument(
        '--output', type=pathlib.Path,
        default=pathlib.Path('SWOT_ORF.json'),
        help='Output JSON file (default: SWOT_ORF.json)')
    args = parser.parse_args()

    cycles = range(args.cycle_start, args.cycle_end + 1)

    logger.info('[1/3] Initializing L3 database: %s', args.l3_db_path)
    if not args.l3_db_path.exists():
        raise FileNotFoundError(f'L3 database path not found: {args.l3_db_path}')
    db = NetcdfFilesDatabaseSwotLRL3(str(args.l3_db_path), follow_symlinks=True)

    logger.info('[2/3] Querying pass 1 over cycles %d..%d…',
                args.cycle_start, args.cycle_end)
    time1, cyc1 = query_pass_stacked(db, cycles, pass_number=1)
    cycle_start_map = cycle_start_from_pass1(time1, cyc1)
    logger.info('      %d cycle start(s) found', len(cycle_start_map))

    logger.info('[3/3] Writing ORF JSON: %s', args.output)
    serializable = to_serializable(cycle_start_map)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    logger.info('Done: %d cycles written to %s', len(serializable), args.output)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    main()