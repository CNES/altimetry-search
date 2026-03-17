# Copyright (c) 2023 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Read the ORF file."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import json
import os
import pathlib
import re

import numpy

#: Regular expression to parse a line of the ORF file
ENTRY: Callable[[str], re.Match | None] = re.compile(
    r'(?P<year>\d{4})\/(?P<month>\d{2})\/(?P<day>\d{2})\s+'
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+(?P<cycle>\d+)\s+'
    r'(?P<pass>\d+)\s+\d+\s+(?P<lon>-?\d+\.\d+)\s+'
    r'(?P<lat>-?\d+\.\d+)').search


@dataclasses.dataclass(frozen=True)
class Entry:
    """Store a single entry of the ORF file."""
    #: Date of the event
    date: numpy.datetime64
    #: Cycle number
    cycle_number: int
    #: Pass number
    pass_number: int
    #: Latitude of the event
    latitude: float
    #: Longitude of the event
    longitude: float

    @classmethod
    def from_line(cls, line) -> Entry | None:
        """Create an entry from a line of the ORF file.

        Args:
            line: Line of the ORF file

        Returns:
            Entry or None if the line is not valid
        """
        match: re.Match[str] | None = ENTRY(line)
        if match is None:
            return None
        return cls(
            date=numpy.datetime64(
                f"{match['year']}-{match['month']}-{match['day']}T"
                f"{match['time']}", 'ms'),
            cycle_number=int(match['cycle']),
            pass_number=int(match['pass']),
            latitude=float(match['lat']),
            longitude=float(match['lon']),
        )


def load(filename: os.PathLike) -> dict[int, numpy.datetime64]:
    """Load an ORF file and return for each cycle numbers the first measurement
    of the pass.

    Args:
        filename: Path to the ORF file

    Returns:
        Dictionary of cycle numbers and dates.
    """
    filename = pathlib.Path(filename)
    with filename.open(encoding='UTF-8') as stream:
        entries: dict[int, numpy.datetime64] = {}
        previous_cycle: int = -1
        for line in stream:
            entry: Entry | None = Entry.from_line(line)
            if entry is None or entry.cycle_number == 0:
                continue
            # Ignore entries for information stored at the poles.
            if entry.latitude == 0:
                continue
            if previous_cycle != entry.cycle_number:
                entries[entry.cycle_number] = entry.date
            previous_cycle = entry.cycle_number
    return entries


class Datetime64Encoder(json.JSONEncoder):
    """JSON encoder for numpy.datetime64."""

    def default(self, obj):
        if isinstance(obj, numpy.datetime64):
            return obj.astype(str)
        return json.JSONEncoder.default(self, obj)


def write_json(filename: os.PathLike, orf: os.PathLike) -> None:
    """Write a JSON file with the first measurement of each cycle.

    Args:
        filename: Path to the CSV file
        orf: Path to the ORF file
    """
    cycles_list = load(orf)
    with open(filename, 'w') as stream:
        json.dump(cycles_list, stream, cls=Datetime64Encoder, indent=2)


def load_json(filename: os.PathLike) -> dict[int, numpy.datetime64]:
    """Load a JSON file with the first measurement of each cycle.

    Args:
        filename: Path to the JSON file

    Returns:
        Dictionary of cycle numbers and dates.
    """
    with open(filename) as stream:
        data = json.load(stream)
    return {int(k): numpy.datetime64(v) for k, v in data.items()}


def usage() -> argparse.Namespace:
    """Print the usage of the script."""
    parser = argparse.ArgumentParser(
        description='Convert the SSALTO/DUACS ORF file to JSON')
    parser.add_argument('orf', type=pathlib.Path, help='ORF file')
    parser.add_argument('json', type=pathlib.Path, help='JSON file')
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = usage()
    write_json(args.json, args.orf)


if __name__ == '__main__':
    main()
