import numpy as np
import pytest
import shapely
from pytest_lazy_fixtures import lf as lazy_fixture


@pytest.mark.network
@pytest.mark.parametrize(
    "etad_burst",
    [
        pytest.param(lazy_fixture("sm_etad_burst"), id="SM"),
        pytest.param(lazy_fixture("iw_etad_burst"), id="IW"),
    ],
)
class TestBurst:
    @staticmethod
    def test_burst_id(etad_burst):
        assert isinstance(etad_burst.burst_id, str)
        assert etad_burst.burst_id.startswith("Burst")

    @staticmethod
    def test_burst_index(etad_burst):
        assert isinstance(etad_burst.burst_index, int)
        assert etad_burst.burst_index > 0

    @staticmethod
    def test_get_burst_grid(etad_burst):
        az, rg = etad_burst.get_burst_grid()
        assert isinstance(az, np.ndarray)
        assert isinstance(rg, np.ndarray)
        assert az.dtype == np.float64
        assert rg.dtype == np.float64
        assert len(az) == etad_burst.lines
        assert len(rg) == etad_burst.samples

    @staticmethod
    def test_get_footprint(etad_burst):
        footprint = etad_burst.get_footprint()
        assert isinstance(footprint, shapely.Polygon)
        assert footprint.has_z

    @staticmethod
    @pytest.mark.parametrize("channel", ["H", "V"])
    def test_get_polarimetric_channel_offset(etad_burst, channel):
        channel = f"{etad_burst.reference_polarization[0]}{channel}"
        offset = etad_burst.get_polarimetric_channel_offset(channel)
        assert set(offset.keys()) == {"x", "y", "units"}
        assert offset["units"] == "s"
        assert isinstance(offset["x"], float)
        assert isinstance(offset["y"], float)

    @staticmethod
    def test_get_polarimetric_channel_offset_invalid_channel(etad_burst):
        with pytest.raises(ValueError):
            etad_burst.get_polarimetric_channel_offset("invalid")

    @staticmethod
    def test_get_timing_calibration_constants(etad_burst):
        k = etad_burst.get_timing_calibration_constants()
        assert set(k.keys()) == {"x", "y", "units"}
        assert k["units"] == "s"
        assert isinstance(k["x"], float)
        assert isinstance(k["y"], float)

    @staticmethod
    def test_reference_polarization(etad_burst):
        rp = etad_burst.reference_polarization
        assert isinstance(rp, str)
        assert rp in {"HH", "HV", "VH", "VV"}

    @staticmethod
    def test_lines(etad_burst):
        assert isinstance(etad_burst.lines, int)
        assert etad_burst.lines > 0

    @staticmethod
    def test_product_id(etad_burst):
        assert isinstance(etad_burst.product_id, str)
        assert etad_burst.product_id.startswith("S1")

    @staticmethod
    def test_product_index(etad_burst):
        assert isinstance(etad_burst.product_index, int)
        assert etad_burst.product_index >= 0

    @staticmethod
    def test_samples(etad_burst):
        assert isinstance(etad_burst.samples, int)
        assert etad_burst.samples > 0

    @staticmethod
    def test_sampling(etad_burst):
        s = etad_burst.sampling
        assert set(s.keys()) == {"x", "y", "units"}
        assert s["units"] == "s"
        assert isinstance(s["x"], float)
        assert isinstance(s["y"], float)
        assert s["x"] >= 0
        assert s["y"] >= 0

    @staticmethod
    def test_sampling_start(etad_burst):
        sstart = etad_burst.sampling_start
        assert set(sstart.keys()) == {"x", "y", "units"}
        assert sstart["units"] == "s"
        assert isinstance(sstart["x"], float)
        assert isinstance(sstart["y"], float)
        assert sstart["x"] >= 0
        assert sstart["y"] >= 0

    @staticmethod
    def test_swath_id(etad_burst):
        assert isinstance(etad_burst.swath_id, str)
        assert 2 <= len(etad_burst.swath_id) <= 3
        assert etad_burst.swath_id[0] in {"E", "I", "S", "W"}

    @staticmethod
    def test_swath_index(etad_burst):
        assert isinstance(etad_burst.swath_index, int)
        assert etad_burst.swath_index > 0

    @staticmethod
    def test_vg(etad_burst):
        assert isinstance(etad_burst.vg, float)
        assert etad_burst.vg > 0

    # @staticmethod
    # @pytest.mark.parametrize("name", list(s1etad.ECorrectionType))
    # def test_get_correction_name(etad_burst, name):
    #     data = etad_burst.get_correction(name)
    #     assert isinstance(data, np.ndarray)
    #     assert data.dtype == np.float64


"""
TODO:

    Sentinel1EtadBurst
        get_correction(name, set_auto_mask, transpose, meter, direction)
        get_lat_lon_height(transpose)
        geodetic_to_radar(lat, lon, h, deg)
        image_to_radar
        intersects(geometry)
        radar_to_geodetic(tau, t, deg)
        __repr__
"""
