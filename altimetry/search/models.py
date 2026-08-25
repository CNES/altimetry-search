import dataclasses
import datetime
from enum import Enum, StrEnum, auto
import pathlib

from . import sad


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
    #: Key identifying the orbit file to fetch (see `sad.OrbitFiles`):
    #: 'swot_calval' or 'swot_science'
    orbit_key: str = dataclasses.field(default_factory=str)
    first_cycle: int = dataclasses.field(default_factory=int)
    #: Fixed number of cycles for this mission phase
    nb_cycle: int = dataclasses.field(default_factory=int)
    #: Fixed number of passes per cycle for this mission phase
    nb_pass: int = dataclasses.field(default_factory=int)
    #: Lower bound of the selectable period
    date_start: datetime.date = dataclasses.field(
        default_factory=datetime.date.today)
    #: Upper bound of the selectable period (None: no upper bound)
    date_end: datetime.date | None = None

    _orbit_file_cache: pathlib.Path | None = dataclasses.field(default=None,
                                                               init=False,
                                                               repr=False,
                                                               compare=False)

    def __post_init__(self):
        """Checks that the (packaged) orf file exists, raises a
        FileNotFoundError if not.

        The orbit file is resolved lazily on
        first access instead (see `orbit_file`), since it may need to be
        downloaded rather than simply looked up on disk.
        """
        orf_file_path = pathlib.Path(__file__).parent / self.orf_file
        self.orf_file = orf_file_path.resolve(strict=True)

    @property
    def orbit_file(self) -> pathlib.Path:
        """Local path to this mission's orbit file, downloading it on first
        access if not already present locally (see
        `altimetry.search.sad.OrbitFiles`)."""
        if self._orbit_file_cache is None:
            self._orbit_file_cache = sad.OrbitFiles()[self.orbit_key]
        return self._orbit_file_cache


missions_properties = {
    Mission.SWOT_SWATH_SCIENCE:
    MissionProperties(
        MissionType.SWATH,
        'resources/SWOT_science_ORF.json',
        'swot_science',
        first_cycle=1,
        nb_cycle=399,
        nb_pass=584,
        date_start=datetime.date(2023, 7, 21),
    ),
    Mission.SWOT_NADIR_SCIENCE:
    MissionProperties(
        MissionType.NADIR,
        'resources/SWOT_science_ORF.json',
        'swot_science',
        first_cycle=1,
        nb_cycle=399,
        nb_pass=584,
        date_start=datetime.date(2023, 7, 21),
    ),
    Mission.SWOT_SWATH_CALVAL:
    MissionProperties(MissionType.SWATH,
                      'resources/SWOT_calval_ORF.json',
                      'swot_calval',
                      first_cycle=474,
                      nb_cycle=105,
                      nb_pass=28,
                      date_start=datetime.date(2023, 3, 29),
                      date_end=datetime.date(2023, 7, 10)),
    Mission.SWOT_NADIR_CALVAL:
    MissionProperties(MissionType.NADIR,
                      'resources/SWOT_calval_ORF.json',
                      'swot_calval',
                      first_cycle=474,
                      nb_cycle=105,
                      nb_pass=28,
                      date_start=datetime.date(2023, 3, 29),
                      date_end=datetime.date(2023, 7, 10)),
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
