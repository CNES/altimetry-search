"""Vendored auxiliary-data fetcher, adapted from fcollections.sad
(https://github.com/CNES/fcollections/tree/main/src/fcollections/sad).

Used to retrieve SWOT orbit files (.nc) from a remote server, if
not already present in the local look-up folders.
"""
from __future__ import annotations

import abc
import logging
import os
from pathlib import Path
import re

import requests

logger = logging.getLogger(__name__)


class IAuxiliaryDataFetcher(abc.ABC):
    """Interface for an auxiliary data source definition.

    Parameters
    ----------
    preferred_target_folder
        The folder where data will be downloaded if it is missing. Default to
        the user home (~/.config/sad)
    """

    PATTERN = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

    def __init__(self, preferred_target_folder: Path | None = None):
        self.preferred_target_folder = preferred_target_folder

    @property
    @abc.abstractmethod
    def keys(self) -> set[str]:
        """Keys identifying single downloadable elements.

        This key is used to build the file names to retrieve from the
        remote source.
        """

    @abc.abstractmethod
    def _download(self, remote_file: str, target_folder: Path):
        """Download the given remote_file into a target folder."""

    @abc.abstractmethod
    def _file_name(self, key: str) -> str:
        """This method contains the mapping between the keys and the files that
        can be downloaded.

        For example: dict(foo='file_foo.nc', bar='file_bar.txt')
        """

    @property
    def name(self) -> str:
        """Name of the auxiliary data.

        It is set as the class name. However, because the name can be
        used as a key, it is converted to snake_case for improved
        ergonomy.
        """
        name = self.__class__.__name__
        return self.PATTERN.sub('_', name).lower()

    def __getitem__(self, key: str) -> Path:
        """Get the file path matching the input key.

        If the file is not found in the local look-up folders, it is downloaded
        from the remote sources into the user .config/sad folder

        Parameters
        ----------
        key
            The key matching the file to download

        Returns
        -------
        :
            The file path matching the key, ensuring it is present on the local
            file system
        """
        if key not in self.keys:
            raise KeyError(
                f'Unknown {key}. Possible choices include {self.keys}')

        candidate = self.file(key)

        # Last folder is user config
        if not candidate.exists():
            self._download(candidate.name, candidate.parent)

        return candidate

    def lookup_folders(self) -> list[Path]:
        """Lists the folders that may contain the files we seek.

        In order to provide flexibility for the system setup, we scan multiple
        folders centered around the SAD_DATA environment variable. For an
        auxiliary data class named AtomicTime, the order of priority is given
        as followed
        ${SAD_DATA_ATOMIC_TIME} > ${SAD_DATA}/atomic_time > ${SAD_DATA} >
        ${HOME}/.config/sad

        Returns
        -------
        :
            A list of folders to scan. Candidate folders that do not exists are
            omitted from the list, with the exception of the user folder which
            serves as a fallback
        """
        folders = []

        try:
            # Try to see if the system is configured with a specific folder
            # for our data.
            # It can happen if we have a baseline of static data set up,
            # but which is only partially filled. Allowing multiple folders
            # gives some flexibility in the overall system setup
            if len(os.environ[f'SAD_DATA_{self.name.upper()}']) > 0:
                folders.append(
                    Path(os.environ[f'SAD_DATA_{self.name.upper()}']))
        except KeyError:
            pass

        try:
            # Else, we scan a more generic environment variable that encompasses
            # all auxiliary data types, with a flat layout or subfolders
            folders.append(Path(os.environ['SAD_DATA']) / self.name.lower())
            if len(os.environ['SAD_DATA']) > 0:
                folders.append(Path(os.environ['SAD_DATA']))
        except KeyError:
            pass

        # The user config folder fallback. data will be downloaded here if it
        # is not available in the previous shared folders
        user_folder = (Path('~') / '.config' / 'sad').expanduser()
        user_folder.mkdir(parents=True, exist_ok=True)
        folders.append(user_folder)

        return [folder for folder in folders if folder.exists()]

    def file(self, key: str) -> Path:
        """Look for a file in local.

        The file is identified by its key and if it not found on the local file
        system, a fallback path is returned that points to the user space.

        Parameters
        ----------
        key
            Identifier for the file to download

        Returns
        -------
        :
            The file path on the local file system. The existence is not
            guaranteed and must be handled by this method caller.
        """
        for folder in self.lookup_folders():
            file_name = self._file_name(key)
            candidate = folder / file_name
            if candidate.exists():
                return candidate
        return (candidate if self.preferred_target_folder is None else
                self.preferred_target_folder / file_name)


class OrbitFiles(IAuxiliaryDataFetcher):
    """SWOT orbit files (.nc), one per mission phase.

    Each mission phase has its own orbit, thus its own file. Available
    keys are the lower-case mission phases: 'calval' and 'science'.

    Parameters
    ----------
    preferred_target_folder
        The folder where data will be downloaded if it is missing. Default
        to the user home (~/.config/sad).
    """

    HTTP_URL = (
        'https://data.aviso.altimetry.fr/aviso-gateway/data/.swot_orbit_files')

    @property
    def keys(self) -> set[str]:
        return {'swot_calval', 'swot_science'}

    def _download(self, remote_file: str, target_folder: Path) -> None:
        fetch_http_file(self.HTTP_URL, remote_file, target_folder)

    def _file_name(self, key: str) -> str:
        mission, phase = key.split('_', 1)
        return f'{mission.upper()}_{phase}_orbit.nc'


def fetch_http_file(url: str, filename: str, target_folder: Path) -> None:
    """Download a single flat file over HTTP into target_folder.

    Adapted from fcollections.sad._karin.fetch_http_file.
    """
    full_url = f'{url}/{filename}'
    logger.info('Downloading %s...', full_url)
    response = requests.get(full_url, timeout=60)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        msg = f'Failed to download file from {full_url}'
        raise RuntimeError(msg) from e

    target_folder.mkdir(parents=True, exist_ok=True)
    with open(target_folder / filename, 'wb') as f:
        f.write(response.content)
    logger.info('Downloading %s... Done', full_url)
