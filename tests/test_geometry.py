import pytest
import pymap3d
from numpy import testing as npt

import s1etad.geometry


# Interface tests on geodetic_to_ecef and ecef_to_geodetic functions
class TestBaseCoordinateConversion:
    ELL = pymap3d.Ellipsoid.from_name("wgs84")
    TEST_DATA = [
        (0, 0, 0, ELL.semimajor_axis, 0, 0),
        (0, 0, 1, ELL.semimajor_axis + 1, 0, 0),
        (0, +90, 0, 0, +ELL.semimajor_axis, 0),
        (0, -90, 0, 0, -ELL.semimajor_axis, 0),
        (0, +180, 0, -ELL.semimajor_axis, 0, 0),
        # (0, -180, 0, -ELL.semimajor_axis, 0, 0),  # ambiguity not relevant
        (+90, 0, 0, 0, 0, +ELL.semiminor_axis),
        (-90, 0, 0, 0, 0, -ELL.semiminor_axis),
    ]

    @staticmethod
    @pytest.mark.parametrize(["lat", "lon", "h", "x", "y", "z"], TEST_DATA)
    def test_geodetic_to_ecef(lat, lon, h, x, y, z):
        xyz = s1etad.geometry.geodetic_to_ecef(lat, lon, h)
        npt.assert_allclose(xyz, (x, y, z), atol=1e-9)

    @staticmethod
    @pytest.mark.parametrize(["lat", "lon", "h", "x", "y", "z"], TEST_DATA)
    def test_ecef_to_geodetic(lat, lon, h, x, y, z):
        llh = s1etad.geometry.ecef_to_geodetic(x, y, z)
        npt.assert_allclose(llh, (lat, lon, h), atol=1e-9)
