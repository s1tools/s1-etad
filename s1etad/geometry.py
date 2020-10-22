# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import fsolve
from scipy.interpolate import interp2d, interp1d

try:
    import pyproj as _pyproj

    def geodetic_to_ecef(lat, lon, h, ell='WGS84', deg=True):
        ecef = _pyproj.crs.CRS(proj='geocent', ellps=ell, datum=ell)
        geodetic = _pyproj.crs.CRS(proj='latlong', ellps=ell, datum=ell)
        transformer = _pyproj.Transformer.from_crs(geodetic, ecef)
        x, y, z = transformer.transform(lon, lat, h, radians=bool(not deg))
        return x, y, z

    def ecef_to_geodetic(x, y, z, ell='WGS84', deg=True):
        ecef = _pyproj.crs.CRS(proj='geocent', ellps=ell, datum=ell)
        geodetic = _pyproj.crs.CRS(proj='latlong', ellps=ell, datum=ell)
        transformer = _pyproj.Transformer.from_crs(geodetic, ecef)
        lon, lat, h = transformer.itransform(x, y, z, radians=bool(not deg))
        return lat, lon, h

except ImportError:
    import pymap3d as _pymap3d

    def geodetic_to_ecef(lat, lon, h, ell='WGS84', deg=True):
        if ell and not isinstance(ell, _pymap3d.ellipsoid.Ellipsoid):
            ell = _pymap3d.ellipsoid.Ellipsoid(ell.lower())
        x, y, z = _pymap3d.ecef.geodetic2ecef(lat, lon, h, ell=ell, deg=deg)
        return x, y, z

    def ecef_to_geodetic(x, y, z, ell='WGS84', deg=True):
        if ell and not isinstance(ell, _pymap3d.ellipsoid.Ellipsoid):
            ell = _pymap3d.ellipsoid.Ellipsoid(ell.lower())
        lat, lon, h = _pymap3d.ecef.ecef2geodetic(x, y, z, ell=ell, deg=deg)
        return lat, lon, h


__all__ = ['GridGeocoding', 'geodetic_to_ecef', 'ecef_to_geodetic']


class GridGeocoding:
    """Class to perform backward and forward geocoding using grid data.

    This class implements a simple geocoding method that exploits
    pre-computed grids of latitude, longitude and height values.

    The method is not as accurate as solving the range-Doppler equations.

    Parameters
    ----------
    grid_latitude : ndarray
        regular grid of latitudes (in degrees) of shape [Ny, Nx]
        (y:azimuth and x:range)
    grid_longitude : ndarray
        regular grid of longitudes (in degrees) of shape [Ny, Nx]
         (y:azimuth and x:range)
    grid_height : ndarray
        regular grid of heights of shape [Ny, Nx] (y:azimuth and x:range)
    xaxis : ndarray
        axis in the x dimension (range) for the grids.
        It could be the range time.
        If not provided then default is np.arange(Nx)
    yaxis : ndarray
        axis in the y dimension (azimuth) for the grids.
        It could be the azimuth time (elapsed seconds since a reference).
        If not provided then default is np.arange(Ny)
    ellipsoid_name : str
        the identifier of the standard ellipsoid to be used for coordinate
        conversion. Default: "WGS84"
    interpolation_kind : str
        Default: "cubic"
    """

    def __init__(self, grid_latitude, grid_longitude, grid_height=0,
                 xaxis=None, yaxis=None, ellipsoid_name='WGS84',
                 interpolation_kind='cubic'):
        self._grid_lats = np.asarray(grid_latitude)
        self._grid_lons = np.asarray(grid_longitude)
        self._grid_heights = np.asarray(grid_height)
        self._ellipsoid_name = ellipsoid_name

        if self._grid_heights.size == 1:
            self._grid_heights = np.full_like(self._grid_lons, grid_height)

        shape = self._grid_lats.shape
        assert self._grid_lons.shape == shape
        assert self._grid_heights.shape == shape

        # define the indices of the grid as regular grid: x:range, y:azimuth
        self._yaxis = np.arange(shape[0])
        self._xaxis = np.arange(shape[1])

        self._yaxis_out = yaxis
        if yaxis is not None:
            self.y_xy_to_time = interp1d(self._yaxis, self._yaxis_out)
            self.y_time_to_xy = interp1d(self._yaxis_out, self._yaxis)

        self._xaxis_out = xaxis
        if xaxis is not None:
            self.x_xy_to_time = interp1d(self._xaxis, self._xaxis_out)
            self.x_time_to_xy = interp1d(self._xaxis_out, self._xaxis)

        # the backward geocoding will be performed by interpolating the input
        # regular grid
        # TODO: interp2d is the simplest interpolator.
        #       Other interpolators could be used.
        self._f_lat = interp2d(self._xaxis, self._yaxis, self._grid_lats,
                               kind=interpolation_kind)
        self._f_lon = interp2d(self._xaxis, self._yaxis, self._grid_lons,
                               kind=interpolation_kind)
        self._f_h = interp2d(self._xaxis, self._yaxis, self._grid_heights,
                             kind=interpolation_kind)

    def _latitude(self, x, y):
        return self._f_lat(x, y)

    def _longitude(self, x, y):
        return self._f_lon(x, y)

    def _height(self, x, y):
        return self._f_h(x, y)

    def _xy_to_time(self, x, y):
        if self._xaxis_out is not None:
            x = self.x_xy_to_time(x)
        if self._yaxis_out is not None:
            y = self.y_xy_to_time(y)
        return x, y

    def _time_to_xy(self, x, y):
        if self._xaxis_out is not None:
            assert self._xaxis_out.min() < x < self._xaxis_out.max(), \
                "Interpolation can't be performed in x"
            x = self.x_time_to_xy(x)
        if self._yaxis_out is not None:
            assert self._yaxis_out.min() < y < self._yaxis_out.max(), \
                "Interpolation can't be performed in x"
            y = self.y_time_to_xy(y)
        return x, y

    def latitude(self, x, y):
        """Interpolate the latitude grid at the (x, y) coordinates.

        Parameters
        ----------
        x : ndarray
            array [N] of x coordinates (or range time) for each input
            in the latitude_grid.
            x shall be the same quantity as used for initialisation
            (coordinates or time).
        y : ndarray
            array [N] of y coordinates (or azimuth time) for each input
            in the latitude_grid.
            y shall be the same quantity as used for initialisation
            (coordinates or time).

        Returns
        -------
        ndarray
            interpolated latitude ([deg])
        """
        # perform coordinate conversion if necessary
        x, y = self._time_to_xy(x, y)
        return self._latitude(x, y)

    def longitude(self, x, y):
        """Interpolate the longitude grid at the (x, y) coordinates.

        Parameters
        ----------
        x : ndarray
            array [N] of x coordinates (or range time) for each input
            in the latitude_grid.
            x shall be the same quantity as used for initialisation
            (coordinates or time).
        y : ndarray
            array [N] of y coordinates (or azimuth time) for each input
            in the latitude_grid.
            y shall be the same quantity as used for initialisation
            (coordinates or time).

        Returns
        -------
        ndarray
            interpolated longitude ([deg])
        """
        # perform coordinate conversion if necessary
        x, y = self._time_to_xy(x, y)
        return self._longitude(x, y)

    def height(self, x, y):
        """Interpolate the height grid at the (x, y) coordinates.

        Parameters
        ----------
        x : ndarray
            array [N] of x coordinates (or range time) for each input
            in the latitude_grid.
            x shall be the same quantity as used for initialisation
            (coordinates or time).
        y : ndarray
            array [N] of y coordinates (or azimuth time) for each input
            in the latitude_grid.
            y shall be the same quantity as used for initialisation
            (coordinates or time).

        Returns
        -------
        ndarray
            interpolated height
        """
        # perform coordinate conversion if necessary
        x, y = self._time_to_xy(x, y)
        return self._height(x, y)

    def _back_equation(self, xy, lat0, lon0):
        """Equation to minimise for back geocoding based on the grid.

        Need to find the root of the following system::

            f(x,y) = latitude_grid(x,y) - lat0 = 0
            g(x,y) = longitude_grid(x,y) -lon0 = 0

        Parameters
        ----------
        xy : tuple
            tuple providing the x, y pixel coordinate
        lat0: float
            target latitude in degrees
        lon0: float
            target longitude in degrees

        Returns
        -------
        (float, float)
            realisation of f and g function at x,y
        """
        x, y = xy

        lat = self._latitude(x, y)
        lon = self._longitude(x, y)
        # h = self._height(x, y)

        eq1 = (lat - lat0).squeeze()
        eq2 = (lon - lon0).squeeze()

        return [eq1, eq2]

    def _initial_guess(self, lat, lon, h=0, deg=True):
        """Return the initial tentative solution for the iterative solver.

        Compute the distance between the point defined by its
        lat, lon and the points of the grid.
        Requires to perform geographic to cartesian conversion.
        """
        # TODO: cache this at least for multipoint conversion
        point_list_cart = geodetic_to_ecef(self._grid_lats.flatten(),
                                           self._grid_lons.flatten(),
                                           self._grid_heights.flatten(),
                                           deg=True)
        point_cart = geodetic_to_ecef(lat, lon, h, deg=deg)

        r = np.asarray(point_list_cart).T - np.asarray(point_cart).T
        dist = np.linalg.norm(r, axis=0)
        ixmin = np.argmin(dist)
        y, x = np.unravel_index(ixmin, self._grid_lats.shape)
        return x, y

    def backward_geocode(self, lats, lons, heights=0, deg=True):
        """Perform the back geocoding: (lat, lon) -> (x, y)

        Parameters
        ----------
        lats : list or ndarray
            array [N] of latitude for which the back geocoding is requested
        lons : list or ndarray
            array [N] of longitude for which the back geocoding is requested
        heights : float, list or ndarray
            height for which the back geocoding is requested
        deg : bool
            True if input geodetic coordinates are expressed in degrees,
            False otherwise

        Returns
        -------
        ndarray, ndarray
            :x0: array [N] of x coordinates (or range time) for each input
                 in lats, lons, heights
            :y0: array [N] of y coordinates (or azimuth time) for each input
                 in lats, lons, heights
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        heights = np.asarray(heights)

        if not deg:
            lats = np.rad2deg(lats)
            lons = np.rad2deg(lons)

        if lats.ndim == 0:
            lats = lats.reshape(1)
        if lons.ndim == 0:
            lons = lons.reshape(1)
        if heights.size == 1:
            heights = np.full_like(lons, heights.item())

        assert lats.shape == lons.shape == heights.shape, \
            "latitude_list shall be of the same shape as longitude_list"

        x0 = np.zeros(lats.shape)
        y0 = np.zeros(lats.shape)

        for idx, (lat, lon) in enumerate(zip(lats, lons)):
            x_guess, y_guess = self._initial_guess(lat, lon, deg=deg)

            sol, info, ier, mesg = fsolve(self._back_equation,
                                          np.asarray([x_guess, y_guess]),
                                          args=(lat, lon), full_output=True)
            x0[idx], y0[idx] = sol[0], sol[1]

        # perform coordinate conversion if necessary
        x0, y0 = self._xy_to_time(x0, y0)

        # TODO: output the convergence flag
        return x0, y0

    def forward_geocode(self, x, y, deg=True):
        """Perform forward geocoding (x, y) -> (lat, lon, h).

        This is simply obtained by re-interpolating the latitude,
        longitude and height grids at the desired coordinates.

        Parameters
        ----------
        x : list or ndarray
            x coordinate or range time at which the geocoding is performed
        y : list or ndarray
            y coordinate or azimuth time at which the geocoding is performed
        deg : bool
            if True than the output lat and lon are expressed in degrees,
            otherwise in radians

        Returns
        -------
        ndarray, ndarray, ndarray
            :latitude: array [N] of latitudes for each input coordinate
            :longitude: array [N] of longitudes for each input coordinate
            :height: array [N] of heights for each input coordinate
        """
        lat = self.latitude(x, y)
        lon = self.longitude(x, y)
        h = self.height(x, y)

        if not deg:
            lat, lon, h = np.deg2rad([lat, lon, h])

        return lat, lon, h
