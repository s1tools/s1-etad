"""Tools for easy access to Sentinel-1 Extended Timing Annotation Datasets.

This package provides a set of classes to open and access all elements,
data and meta-data, of the Sentinel-1 Extended Timing Annotation Datasets
(S1-ETAD).

Basic elements are:

* :class:`Sentinel1Etad`
* :class:`Sentinel1EtadSwath`
* :class:`Sentinel1EtadBurst`
"""

from .product import (  # noqa: F401
    Sentinel1Etad,
    ECorrectionType,
    Sentinel1EtadBurst,
    Sentinel1EtadSwath,
)

__version__ = "0.5.6.dev0"


# register display functions for Jupyter
from ._jupyter_support import _register_jupyter_formatters

_register_jupyter_formatters()
del _register_jupyter_formatters
