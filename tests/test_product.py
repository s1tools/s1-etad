"""Unit tests for ETAD products."""

import re
import pathlib
import datetime

import pandas as pd
import pytest
import netCDF4
import shapely
from pytest_lazy_fixtures import lf as lazy_fixture

import s1etad

COLUMNS = (
    "bIndex",
    "pIndex",
    "sIndex",
    "productID",
    "swathID",
    "azimuthTimeMin",
    "azimuthTimeMax",
)


CORRECTION_SETTINGS = {
    "troposphericDelayCorrection",
    "ionosphericDelayCorrection",
    "solidEarthTideCorrection",
    "bistaticAzimuthCorrection",
    "dopplerShiftRangeCorrection",
    "FMMismatchAzimuthCorrection",
}


def _get_mode_id(path):
    name = pathlib.Path(path).name
    mobj = re.match(r"S1[AB]_(?P<mode_id>[SIE][W0-9])_ETA_.*\.SAFE", name)
    assert mobj is not None
    mode_id = mobj.group("mode_id")
    return "SM" if mode_id.startswith("S") else mode_id


def _get_n_swaths(mode_id):
    if mode_id == "SM":
        n_swaths = 1
    elif mode_id == "IW":
        n_swaths = 3
    elif mode_id == "EW":
        n_swaths = 5
    else:
        raise ValueError(f"invalid mode_id: {mode_id}")
    return n_swaths


@pytest.mark.network
@pytest.mark.parametrize(
    "filename",
    [
        pytest.param(lazy_fixture("sm_etad_filename"), id="SM"),
        pytest.param(lazy_fixture("iw_etad_filename"), id="IW"),
    ],
)
def test_open_etad_product(filename):
    eta = s1etad.Sentinel1Etad(filename)
    assert eta


@pytest.mark.network
@pytest.mark.parametrize(
    "etad_product",
    [
        pytest.param(lazy_fixture("sm_etad_product"), id="SM"),
        pytest.param(lazy_fixture("iw_etad_product"), id="IW"),
    ],
)
class TestEtadProduct:
    @staticmethod
    def test_burst_catalogue(etad_product):
        burst_catalogue = etad_product.burst_catalogue
        assert isinstance(burst_catalogue, pd.DataFrame)
        assert len(burst_catalogue) > 0
        if _get_mode_id(etad_product.product) == "SM":
            assert len(burst_catalogue) == 1
        assert tuple(burst_catalogue.columns) == COLUMNS

    @staticmethod
    def test_ds(etad_product):
        ds = etad_product.ds
        assert isinstance(ds, netCDF4.Dataset)

    @staticmethod
    def test_get_footprint(etad_product):
        footprint = etad_product.get_footprint()
        assert isinstance(footprint, shapely.MultiPolygon)
        for geom in footprint.geoms:
            assert geom.has_z

    @staticmethod
    @pytest.mark.parametrize(
        "correction_type",
        [
            "tropospheric",
            "ionospheric",
            "geodetic",
            "bistatic",
            "doppler",
            "fmrate",
            "sum",
        ],
    )
    def test_get_statistics(etad_product, correction_type):
        if _get_mode_id(etad_product.product) == "SM" and correction_type in {
            "doppler",
            "fmrate",
        }:
            pytest.xfail(reason="layer not available for SM")
        stats = etad_product.get_statistics(correction_type)
        assert set(stats.keys()).issubset({"x", "y", "unit"})
        assert stats["unit"] == "s"

    @staticmethod
    def test_get_statistics_units(etad_product):
        assert etad_product.get_statistics("sum", meter=True)["unit"] == "m"

    @staticmethod
    def test_grid_sampling(etad_product):
        grid_sampling = etad_product.grid_sampling
        assert isinstance(grid_sampling, dict)
        assert set(grid_sampling.keys()) == {"x", "y", "unit"}
        assert grid_sampling["unit"] == "s"

    @staticmethod
    def test_grid_spacing(etad_product):
        grid_spacing = etad_product.grid_spacing
        assert isinstance(grid_spacing, dict)
        assert set(grid_spacing.keys()) == {"x", "y", "unit"}
        assert grid_spacing["unit"] == "m"

    @staticmethod
    def test_intersects(etad_product):
        assert etad_product.intersects(etad_product.get_footprint().centroid)
        assert not etad_product.intersects(shapely.Point(0, 90, 0))

    @staticmethod
    def test_iter_bursts(etad_product):
        bursts = list(etad_product.iter_bursts())
        assert len(bursts) > 0
        if _get_mode_id(etad_product.product) == "SM":
            assert len(bursts) == 1
        assert all(
            isinstance(burst, s1etad.Sentinel1EtadBurst)
            for burst in etad_product.iter_bursts()
        )

    @staticmethod
    def test_iter_swaths(etad_product):
        swaths = list(etad_product.iter_swaths())
        assert len(swaths) == etad_product.number_of_swath
        assert all(
            isinstance(swath, s1etad.Sentinel1EtadSwath)
            for swath in etad_product.iter_swaths()
        )

    @staticmethod
    def test_azimuth_time(etad_product):
        assert isinstance(etad_product.min_azimuth_time, datetime.datetime)
        assert isinstance(etad_product.max_azimuth_time, datetime.datetime)
        assert etad_product.min_azimuth_time <= etad_product.max_azimuth_time

    @staticmethod
    def test_range_time(etad_product):
        assert isinstance(etad_product.min_range_time, float)
        assert isinstance(etad_product.max_range_time, float)
        etad_product.min_range_time <= etad_product.max_range_time

    @staticmethod
    def test_number_of_swaths(etad_product):
        n_swaths = _get_n_swaths(_get_mode_id(etad_product.product))
        assert etad_product.number_of_swath == n_swaths

    @staticmethod
    def test_processing_setting(etad_product):
        assert set(etad_product.processing_setting()) == CORRECTION_SETTINGS

    @staticmethod
    def test_s1_etad_product_list(etad_product):
        assert len(etad_product.s1_product_list()) == 1
        assert all(
            re.match(r"S1[AB]_([SA]\d|IW|EW)_SLC_\w*\.SAFE", etad_product)
            for etad_product in etad_product.s1_product_list()
        )
        assert sorted(etad_product.ds.groups.keys()) == sorted(
            etad_product.swath_list
        )

    @staticmethod
    def test_swath_list(etad_product):
        mode_id = _get_mode_id(etad_product.product)
        if mode_id.startswith("S"):
            pattern = r"S\d"
        else:
            pattern = mode_id
        assert all(
            re.match(pattern, swath_id) for swath_id in etad_product.swath_list
        )

    @staticmethod
    def test_vg(etad_product):
        assert isinstance(etad_product.vg, float)
        assert 6000 <= etad_product.vg <= 8000


@pytest.mark.network
@pytest.mark.parametrize(
    ("etad_product", "etad_filename"),
    [
        (lazy_fixture("sm_etad_product"), lazy_fixture("sm_etad_filename")),
        (lazy_fixture("iw_etad_product"), lazy_fixture("iw_etad_filename")),
    ],
)
def test_product(etad_product, etad_filename):
    assert isinstance(etad_product.product, pathlib.Path)
    assert etad_product.product == pathlib.Path(etad_filename)


"""
TODO:

    Sentinel1Etad
        merge_correction(name, selection, set_auto_mask, meter, direction)
        query_burst(first_time, product_name, last_time, swath, geometry)
        __getitem__
        __iter__
        __repr__
        __str__
        + iter_bursts(selection)
        + iter_swaths(selection)
        + get_footprint(selection, merge)
"""
