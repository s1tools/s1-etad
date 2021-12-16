Version history
===============

s1etad v0.5.2 (27/07/2021)
--------------------------

* Fix imports from ``typing`` (fall-back typing-extensions when necessary).


s1etad v0.5.1 (27/07/2021)
--------------------------

* Add missing dependency form ``typing-extensions`` for Python < 3.8.


s1etad v0.5.0 (16/07/2021)
--------------------------

* Merger functions now return masked arrays.
* The :class:`s1etad.product.Sentinel1EtadBurst` class now has two new
  properties:

  - :data:`s1etad.product.Sentinel1EtadBurst.vg`, the average zero-Doppler
    ground velocity
  - :data:`s1etad.product.Sentinel1EtadBurst.reference_polarization`,
    a string identifying the reference polarization

  and a new method:

  - :meth:`s1etad.product.Sentinel1EtadBurst.get_polarimetric_channel_offset`
    to get the constant time offset of a polarimetric channel w.r.t. the
    reference one

* Support for ETAD-DLR-PS-0014 "ETAD Product Format Specification" Issue 1.5
  (SETAP v1.6):

  - new method (
    :meth:`s1etad.product.Sentinel1EtadBurst.get_timing_calibration_constants`)
    for retrieving timing calibration constants

* new :mod:`s1etad.ql` module for geocoded quick-look images generation


s1etad v0.4.0 (01/12/2020)
--------------------------

* Implement ETAD-DLR-PS-0014 "Product Format Specification" v1.2.
* The burst catalogue initialization and the
  :meth:`s1etad.product.Sentinel1Etad.s1_product_list` method have been
  re-implemented to exploit NetCDF instead of XML (30% faster product loading).
* All notebooks have been update to use the new demo products
  (with updated format).
* The back-geocoding implementation has been simplified and improved
  (caching of ECEF coordinates during guess computation).
* Fixed :meth:`s1etad.product.Sentinel1Etad.iter_bursts` in case of empty
  ``selection``.
* Fixed KMZ generation in case of missing slices.
* Added min/max range time attributes to :class:`s1etad.product.Sentinel1Etad`
  class.
* New notebook providing a basic step by step guide to perform the
  correction of a single Sentinel-1 SLC burst with the timings provided
  by the S1-ETAD products.
* Now the :meth:`s1etad.product.Sentinel1Etad.s1_product_list` always returns
  a list. Previously a string was returned in case of single swath.
* Fixed the :meth:`s1etad.product.Sentinel1Etad.query` method in the case
  in which the ``product_name`` parameter is used to search for S1 Standard
  ("S") products.
* Always use the `Sphinx RTD Theme <https://sphinx-rtd-theme.readthedocs.io/>`_
  (also for local builds).


s1etad v0.3.0 (27/10/2020)
--------------------------

* Now ``s1etad`` is a package.
* Improved ``get_footprint`` methods:

  - support for extended selection semantics (also accept the result of
    a query as parameter)
  - support for the ``merge`` option: now it is possible to request a
    single "merged" footprint; by default the method returns the set of
    footprints of all bursts

* New :meth:`s1etad.product.Sentinel1Etad.get_statistics` method.
* Added missing attributes to :class:`s1etad.product.Sentinel1Etad`,
  :class:`s1etad.product.Sentinel1EtadSwath` and
  :class:`s1etad.product.Sentinel1EtadBurst` classes.
* Removed :meth:`s1etad.product.Sentinel1Etad.xpath_to_list` method from
  the public API (the private one is still available)
* Strongly improved KMZ export function.

  - new dedicated :mod:`s1etad.kmz` module (providing the
    :func:`s1etad.kmz.s1etad_to_kmz` function)
  - removed the obsolete :meth:`s1etad.product.Sentinel1Etad.to_kml` method

* New methods to find points (and geometries) intersecting
  the burst/swath/product footprint.
* New method and functions for direct and inverse geocoding
  (approximated algorithm)
* New Command Line Interface (CLI) for basic functions
  (only "export-kmz" at the moment).
* Documentation:

  - added instructions to install via conda packages
  - added pointers to "stable" and "development" (latest) version of the
    documentation


s1etad v0.2.0 (12/09/2020)
--------------------------

Improved packaging and docs.


s1etad v0.1.0 (11/09/2020)
--------------------------

Initial release.
