import re

import pytest
import shapely
from pytest_lazy_fixtures import lf as lazy_fixture

import s1etad


@pytest.mark.network
@pytest.mark.parametrize(
    "etad_swath",
    [
        pytest.param(lazy_fixture("sm_etad_swath"), id="SM"),
        pytest.param(lazy_fixture("iw_etad_swath"), id="IW"),
    ],
)
class TestSwath:
    @staticmethod
    def test_burst_list(etad_swath):
        burst_list = etad_swath.burst_list
        assert all(isinstance(burst_id, int) for burst_id in burst_list)
        assert len(burst_list) >= 0
        if etad_swath.swath_id.startswith("S"):
            assert len(burst_list) == 1

    @staticmethod
    def test_get_footprint(etad_swath):
        footprint = etad_swath.get_footprint()
        assert isinstance(footprint, shapely.MultiPolygon)
        for geom in footprint.geoms:
            assert geom.has_z

        footprint_no_merge = etad_swath.get_footprint(merge=False)
        assert isinstance(footprint_no_merge, shapely.MultiPolygon)
        for geom in footprint_no_merge.geoms:
            assert geom.has_z
        assert footprint == footprint_no_merge

    @staticmethod
    def test_get_footprint_merge(etad_swath):
        footprint = etad_swath.get_footprint(merge=True)
        assert isinstance(footprint, shapely.Polygon)
        assert footprint.has_z

    @staticmethod
    def test_intersects(etad_swath):
        assert etad_swath.intersects(etad_swath.get_footprint().centroid)
        assert not etad_swath.intersects(shapely.Point(0, 90, 0))

    @staticmethod
    def test_iter_bursts(etad_swath):
        bursts = list(etad_swath.iter_bursts())
        assert len(bursts) > 0
        if etad_swath.swath_id.startswith("S"):
            assert len(bursts) == 1
        assert all(
            isinstance(burst, s1etad.Sentinel1EtadBurst)
            for burst in etad_swath.iter_bursts()
        )

    @staticmethod
    def test_number_of_burst(etad_swath):
        if etad_swath.swath_id.startswith("S"):
            assert etad_swath.number_of_burst == 1
        else:
            assert etad_swath.number_of_burst > 1

    @staticmethod
    def test_sampling(etad_swath):
        sampling = etad_swath.sampling
        assert isinstance(sampling, dict)
        assert set(sampling.keys()) == {"x", "y", "units"}
        assert sampling["units"] == "s"

    @staticmethod
    def test_sampling_start(etad_swath):
        sampling_start = etad_swath.sampling_start
        assert isinstance(sampling_start, dict)
        assert set(sampling_start.keys()) == {"x", "y", "units"}
        assert sampling_start["units"] == "s"

    @staticmethod
    def test_swath_id(etad_swath):
        assert isinstance(etad_swath.swath_id, str)
        assert re.match(r"S\d|IW|EW", etad_swath.swath_id)

    @staticmethod
    def test_swath_index(etad_swath):
        assert isinstance(etad_swath.swath_index, int)
        assert etad_swath.swath_index > 0


"""
TODO:

    Sentinel1EtadSwath
        + get_footprint(selection)
        + iter_bursts(selection)
        merge_correction(name, selection, set_auto_mask, meter, direction)

        __getitem__
        __iter__
        __repr__
"""
