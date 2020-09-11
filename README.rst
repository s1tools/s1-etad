s1etad Python package
======================

.. image:: https://img.shields.io/readthedocs/s1etad/latest.svg
    :target: http://s1etad.readthedocs.org/
    :alt: Documentation Status


:copyright: 2020 Nuno Mirada, Antonio Valentino


About
-----

The ``s1etad`` Python package provides tools for easy access to
Sentinel-1 Extended Timing Annotation Datasets.

The current version of the package is based on the Product Format
Specification Document (ETAD-DLR-PS-0014) Issue 1.1.

Main features provided are:

* open and navigate all the S1-ETAD elements:

  - product (:class:`s1etad.Sentinel1Etad`)
  - swaths (:class:`s1etad.Sentinel1EtadSwath`)
  - bursts (:class:`s1etad.Sentinel1EtadBurst`)

* inspect metadata
* perform queries on bursts (by time, swath name, product name or any
  combination) using the :attr:`Sentinel1Etad.burst_catalog`
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
:documentation: https://s1etad.readthedocs.io
:sources: https://gitlab.com/nuno.miranda/s1-etad
:issues: https://gitlab.com/nuno.miranda/s1-etad/-/issues


Requirements
------------

* `Python <https://www.python.org>`_ >= 3.6
* `numpy <https://numpy.org>`_
* `scipy <https://scipy.org>`_
* `lxml <https://lxml.de>`_
* `netCDF4 <http://github.com/Unidata/netcdf4-python>`_
* `pandas <https://pandas.pydata.org>`_
* `python-dateutil <https://dateutil.readthedocs.io>`_
* `shapely <https://github.com/Toblerity/Shapely>`_
* `simplekml <https://pypi.org/project/simplekml>`_


Installation
------------

To install the ``s1etad`` package simpy run the following command::

  $ python3 -m pip install s1etad


License
-------

The s1etad package is distributed unthe the terms of the MIT License.

See ``LICENSE.txt`` for mare details.
