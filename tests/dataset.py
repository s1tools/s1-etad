"""Utility functions for unit testing."""

import shutil
import pathlib

import pooch

BASEURL = "https://sentinels.copernicus.eu/documents/d/sentinel"
DATADIR = pathlib.Path(__file__).parent / "data"
REGISTRY = DATADIR.joinpath("registry.txt")


datarepo = pooch.create(
    path=DATADIR,
    base_url=BASEURL,
    # retry_if_failed=3,
    # version="0.5"
    registry=None,
    env="S1ETAD_TEST_DATADIR",
)
datarepo.load_registry(REGISTRY)


def download_all(datarepo: pooch.Pooch = datarepo):
    for item in datarepo.registry_files:
        datarepo.fetch(item, processor=pooch.Untar(), progressbar=True)


def clean_cache(datarepo: pooch.Pooch = datarepo):
    for item in datarepo.registry_files:
        shutil.rmtree(
            datarepo.path.joinpath(f"{item}.untar"), ignore_errors=True
        )
        datarepo.path.joinpath(item).unlink(missing_ok=True)
