Version history
===============

s1etad v0.3.0 (27/10/2020)
--------------------------

* Now ``s1etad`` is a package.
* Improved ``get_gootprint`` methods:

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
