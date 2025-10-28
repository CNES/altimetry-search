import dataclasses
from enum import Enum, StrEnum, auto
import pathlib


class Mission(StrEnum):
    SWOT_SWATH = ('Swot - swath')
    SWOT_NADIR = ('Swot - nadir')


class MissionType(Enum):
    SWATH = auto()
    NADIR = auto()


@dataclasses.dataclass
class MissionProperties:
    mission_type: MissionType
    orf_file: str = dataclasses.field(default_factory=str)
    orbit_file: str = dataclasses.field(default_factory=str)

    def __post_init__(self):
        orf_file_path = pathlib.Path(__file__).parent / self.orf_file
        self.orf_file = orf_file_path.resolve(strict=True)

        orbit_file_path = pathlib.Path(__file__).parent / self.orbit_file
        self.orbit_file = orbit_file_path.resolve(strict=True)


missions_properties = {
    Mission.SWOT_SWATH:
    MissionProperties(MissionType.SWATH, 'resources/SWOT_ORF.json',
                      'resources/SWOT_orbit.nc'),
    Mission.SWOT_NADIR:
    MissionProperties(MissionType.NADIR, 'resources/SWOT_ORF.json',
                      'resources/SWOT_orbit.nc')
}


class MissionPropertiesLoader:

    def load(self, m: Mission):
        return missions_properties[m]
