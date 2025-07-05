s1etad Python package
======================

.. badges

|PyPI Status| |GHA Status| |Documentation Status| |Python Versions| |License|

.. |PyPI Status| image:: https://img.shields.io/pypi/v/s1etad.svg
    :target: https://pypi.org/project/s1etad
    :alt: PyPI Status
.. |GHA Status| image:: https://github.com/s1tools/s1-etad/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/s1tools/s1-etad/actions
    :alt: GitHub Actions Status
.. |Documentation Status| image:: https://readthedocs.org/projects/s1etad/badge
    :target: https://s1etad.readthedocs.io
    :alt: Documentation Status
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/s1etad
    :target: https://pypi.org/project/s1etad
    :alt: Supported Python versions
.. |License| image:: https://img.shields.io/pypi/l/s1etad
    :target: https://pypi.org/project/s1etad
    :alt: License

.. contents

About
-----

The ``s1etad`` Python package provides tools for easy access to
Sentinel-1 Extended Timing Annotation Datasets.

The current version of the package is based on the Product Format
Specification Document (ETAD-DLR-PS-0014) Issue 1.12.

Main features provided are:

* open and navigate all the S1-ETAD elements:

  - product (|Sentinel1Etad| class)
  - swaths (|Sentinel1EtadSwath| class)
  - bursts (|Sentinel1EtadBurst| class)

* inspect metadata
* perform queries on bursts (by time, swath name, product name or any
  combination) using the |Sentinel1Etad.burst_catalogue|
* easy iteration
* read corrections
* perform correction mosaic: de-bursting and swath stitching (a basic
  algorithm is currently implemented)
* get footprints
* generate simple KML files of the product
* integration with Jupyter_ environments


.. _Jupyter: https://jupyter.org


Project links
-------------

:download: https://pypi.org/project/s1etad
:documentation: `latest <https://s1etad.readthedocs.io/en/latest>`_,
                `stable <https://s1etad.readthedocs.io/en/stable>`_
:sources: https://github.com/s1tools/s1-etad
:issues: https://github.com/s1tools/s1-etad/issues
:conda package: https://anaconda.org/avalentino/s1etad


Requirements
------------

* `Python <https://www.python.org>`_ >= 3.9
* `numpy <https://numpy.org>`_
* `scipy <https://scipy.org>`_
* `lxml <https://lxml.de>`_
* `netCDF4 <https://github.com/Unidata/netcdf4-python>`_
* `pandas <https://pandas.pydata.org>`_
* `shapely <https://github.com/shapely/shapely>`_
* `pymap3d <https://github.com/geospace-code/pymap3d>`_
* `simplekml <https://pypi.org/project/simplekml>`_
* `GDAL <https://gdal.org>`_
* `matplotlib <https://matplotlib.org>`_


Installation
------------

To install the ``s1etad`` package simply run the following command::

  $ python3 -m pip install s1etad[kmz,cli]

In conda environments::

  $ conda install -c avalentino -c conda-forge s1etad


License
-------

:copyright: 2020-2025 Nuno Mirada, Antonio Valentino

The s1etad package is distributed under the terms of the MIT License.

See ``LICENSE.txt`` for mare details.


.. substitutions
.. |Sentinel1Etad| replace:: ``Sentinel1Etad``
.. |Sentinel1EtadSwath| replace:: ``Sentinel1EtadSwath``
.. |Sentinel1EtadBurst| replace:: ``Sentinel1EtadBurst``
.. |Sentinel1Etad.burst_catalogue| replace:: ``Sentinel1Etad.burst_catalogue``
