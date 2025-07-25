import dataclasses
from enum import Enum, StrEnum, auto


class Mission(StrEnum):
    SWOT_SWATH = ('Swot - swath')
    SWOT_NADIR = ('Swot - nadir')


class MissionType(Enum):
    SWATH = auto()
    NADIR = auto()


@dataclasses.dataclass
class MissionProperties:
    mission_type: MissionType = MissionType.SWATH
    orf_file: str = dataclasses.field(default_factory=str)
    orbit_file: str = dataclasses.field(default_factory=str)
    passes_per_cycle: int = dataclasses.field(default_factory=int)


missions_properties = {
    Mission.SWOT_SWATH:
    MissionProperties(MissionType.SWATH, 'resources/SWOT_ORF.json',
                      'resources/SWOT_orbit.nc', 584),
    Mission.SWOT_NADIR:
    MissionProperties(MissionType.NADIR, 'resources/SWOT_ORF.json',
                      'resources/SWOT_orbit.nc', 584)
}


class MissionPropertiesLoader:

    def load(self, m: Mission):
        return missions_properties[m]
