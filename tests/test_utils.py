"""Unittest for the s1eats.utils module."""

import pytest

import s1etad.utils
from s1etad import ECorrectionType


@pytest.mark.parametrize(
    ("input_", "output"),
    [
        pytest.param(
            None,
            [
                (ECorrectionType.BISTATIC, "y"),
                (ECorrectionType.DOPPLER, "x"),
                (ECorrectionType.FMRATE, "y"),
                (ECorrectionType.GEODETIC, "x"),
                (ECorrectionType.GEODETIC, "y"),
                (ECorrectionType.IONOSPHERIC, "x"),
                (ECorrectionType.SUM, "x"),
                (ECorrectionType.SUM, "y"),
                (ECorrectionType.TROPOSPHERIC, "x"),
            ],
            id="default",
        ),
        pytest.param(
            [
                ECorrectionType.GEODETIC,
                ECorrectionType.IONOSPHERIC,
                ECorrectionType.TROPOSPHERIC,
            ],
            [
                (ECorrectionType.GEODETIC, "x"),
                (ECorrectionType.GEODETIC, "y"),
                (ECorrectionType.IONOSPHERIC, "x"),
                (ECorrectionType.TROPOSPHERIC, "x"),
            ],
            id="enums",
        ),
        pytest.param(
            ["geodetic", "ionospheric", "tropospheric"],
            [
                (ECorrectionType.GEODETIC, "x"),
                (ECorrectionType.GEODETIC, "y"),
                (ECorrectionType.IONOSPHERIC, "x"),
                (ECorrectionType.TROPOSPHERIC, "x"),
            ],
            id="strings",
        ),
    ],
)
def test_iter_corrections(input_, output):
    out = sorted(s1etad.utils.iter_corrections(input_), key=str)
    assert out == output
