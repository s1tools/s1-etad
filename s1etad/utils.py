# -*- coding: utf-8 -*-

from .product import ECorrectionType


def iter_corrections(corrections=None):
    if corrections is None:
        corrections = ECorrectionType

    if isinstance(corrections, str) or corrections in ECorrectionType:
        corrections = [corrections]

    for correction in corrections:
        correction = ECorrectionType(correction)
        if correction in {ECorrectionType.TROPOSPHERIC,
                          ECorrectionType.IONOSPHERIC,
                          ECorrectionType.DOPPLER}:
            yield correction, 'x'
        elif correction in {ECorrectionType.BISTATIC, ECorrectionType.FMRATE}:
            yield correction, 'y'
        elif correction in {ECorrectionType.SUM, ECorrectionType.GEODETIC}:
            yield correction, 'x'
            yield correction, 'y'
