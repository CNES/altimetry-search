import dataclasses
from enum import Enum, StrEnum, auto
import pathlib


class Mission(StrEnum):
    """The mission selected in the application."""
    SWOT_SWATH_SCIENCE = ('Swot Science - swath')
    SWOT_NADIR_SCIENCE = ('Swot Science - nadir')
    SWOT_SWATH_CALVAL = ('Swot Calval - swath')
    SWOT_NADIR_CALVAL = ('Swot Calval - nadir')


class MissionType(Enum):
    """The type of the mission: either nadir, or swath mission."""
    SWATH = auto()
    NADIR = auto()


@dataclasses.dataclass
class MissionProperties:
    """Represents the properties of a mission."""
    mission_type: MissionType
    orf_file: str = dataclasses.field(default_factory=str)
    orbit_file: str = dataclasses.field(default_factory=str)
    nb_cycle: int = 200
    first_cycle: int = 1

    def __post_init__(self):
        """Checks that orf and orbit file exist, raises a FileNotFoundError if
        not."""
        orf_file_path = pathlib.Path(__file__).parent / self.orf_file
        self.orf_file = orf_file_path.resolve(strict=True)

        orbit_file_path = pathlib.Path(__file__).parent / self.orbit_file
        self.orbit_file = orbit_file_path.resolve(strict=True)


missions_properties = {
    Mission.SWOT_SWATH_SCIENCE:
    MissionProperties(
        MissionType.SWATH, 'resources/SWOT_science_ORF.json',
        '/home/atonneau/workspace/TESTS/SEARCH_SWOT/SWOT_science_orbit.nc'),
    Mission.SWOT_NADIR_SCIENCE:
    MissionProperties(MissionType.NADIR, 'resources/SWOT_science_ORF.json',
                      'resources/SWOT_science_orbit.nc'),
    Mission.SWOT_SWATH_CALVAL:
    MissionProperties(
        MissionType.SWATH,
        '/home/atonneau/workspace/TESTS/SEARCH_SWOT/SWOT_calval_ORF.json',
        '/home/atonneau/workspace/TESTS/SEARCH_SWOT/SWOT_calval_orbit.nc',
        nb_cycle=105,
        first_cycle=474),
    Mission.SWOT_NADIR_CALVAL:
    MissionProperties(
        MissionType.NADIR,
        '/home/atonneau/workspace/TESTS/SEARCH_SWOT/SWOT_calval_ORF.json',
        '/home/atonneau/workspace/TESTS/SEARCH_SWOT/SWOT_calval_orbit.nc',
        nb_cycle=105,
        first_cycle=474)
}


class MissionPropertiesLoader:
    """Utility class to load a mission's properties."""

    def load(self, m: Mission) -> MissionProperties:
        """Loads mission properties.

        Parameters
        ----------
        m: Mission
            a mission.

        Returns
        -------
            the mission properties.
        """
        return missions_properties[m]
