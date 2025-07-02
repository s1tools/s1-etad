"""pytest fixtures for s1-etad testing."""

import pathlib

import pooch
import pytest

import s1etad

from . import dataset

S1_ETAD_IW_NAME = "s1_setap_3-0-0_2025-05-30_etad_iw-tar"
S1_ETAD_SM_NAME = "s1_setap_3-0-0_2025-05-30_etad_sm-tar"


def _get_product_path(name: str) -> pathlib.Path:
    files = dataset.datarepo.fetch(name, pooch.Untar())
    assert files
    return pathlib.Path(files[0]).parent


@pytest.fixture
def sm_etad_filename() -> pathlib.Path:
    return _get_product_path(S1_ETAD_SM_NAME)


@pytest.fixture(scope="session")
def sm_etad_product():
    filename = _get_product_path(S1_ETAD_SM_NAME)
    return s1etad.Sentinel1Etad(filename)


@pytest.fixture(scope="session")
def sm_etad_swath(sm_etad_product):
    swath_id = sm_etad_product.swath_list[0]
    return sm_etad_product[swath_id]


@pytest.fixture(scope="session")
def sm_etad_burst(sm_etad_swath):
    burst_id = sm_etad_swath.burst_list[0]
    return sm_etad_swath[burst_id]


@pytest.fixture
def iw_etad_filename() -> pathlib.Path:
    return _get_product_path(S1_ETAD_IW_NAME)


@pytest.fixture(scope="session")
def iw_etad_product():
    filename = _get_product_path(S1_ETAD_IW_NAME)
    return s1etad.Sentinel1Etad(filename)


@pytest.fixture(scope="session")
def iw_etad_swath(iw_etad_product):
    swath_id = iw_etad_product.swath_list[0]
    return iw_etad_product[swath_id]


@pytest.fixture(scope="session")
def iw_etad_burst(iw_etad_swath):
    burst_id = iw_etad_swath.burst_list[0]
    return iw_etad_swath[burst_id]
