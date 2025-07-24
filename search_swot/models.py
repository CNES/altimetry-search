from enum import Enum, auto


class MissionType(Enum):
    SWATH = auto()
    NADIR = auto()


class Mission(Enum):
    SWOT_SWATH = ('Swot - swath', MissionType.SWATH, 'resources/SWOT_ORF.json',
                  'resources/SWOT_orbit.nc', 584)
    SWOT_NADIR = ('Swot - nadir', MissionType.NADIR, 'resources/SWOT_ORF.json',
                  'resources/SWOT_orbit.nc', 584)
    JASON3 = ('Jason 3', MissionType.NADIR, 'resources/SWOT_ORF.json',
              'resources/SWOT_orbit.nc', 584)

    def __new__(cls, display: str, mission_type: MissionType, orf_file: str,
                orbit_file: str, passes_per_cycle: int):
        obj = object.__new__(cls)
        obj._value_ = display
        obj.mission_type = mission_type
        obj.orf_file = orf_file
        obj.orbit_file = orbit_file
        obj.passes_per_cycle = passes_per_cycle
        return obj

    def __str__(self):
        return self._value_
