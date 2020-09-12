# -*- coding: utf-8 -*-

"""Tools for easy access to Sentinel-1 Extended Timing Annotation Datasets.

This module provides a set of classes to open and access all elements,
data and meta-data, of the Sentinel-1 Extended Timing Annotation Datasets
(S1-ETAD).

Basic elements are:

* :class:`Sentinel1Etad`
* :class:`Sentinel1EtadSwath`
* :class:`Sentinel1EtadBurst`
"""

import enum
import pathlib
import warnings
import functools
from typing import Union

import numpy as np
from scipy import constants
from lxml import etree
from netCDF4 import Dataset

import pandas as pd
from dateutil import parser

from shapely.geometry import Polygon, MultiPolygon
import simplekml


__version__ = '0.3.0.dev0'
__all__ = [
    'Sentinel1Etad', 'Sentinel1EtadSwath', 'Sentinel1EtadBurst',
    'ECorrectionType',
]


class ECorrectionType(enum.Enum):
    TROPOSPHERIC = 'tropospheric'
    IONOSPHERIC = 'ionospheric'
    GEODETIC = 'geodetic'
    BISTATIC = 'bistatic'
    DOPPLER = 'doppler'
    FMRATE = 'fmrate'
    SUM = 'sum'


CorrectionType = Union[ECorrectionType, str]


class Sentinel1Etad:
    """Sentinel-1 ETAD product.

    Class to decode and access the elements of the Sentinel ETAD product
    which specification is governed by ETAD-DLR-PS-0014

    The index operator [] (implemented with the __getitem__ method) returns
    a Sentinel1EtadSwath instance.

    Attributes
    ----------
    product : pathlib.Path
        path of the S1-ETAD product (it is a directory)
    burst_catalogue : pandas.DataFrame
        dataframe containing main information of all bursts present in
        the product
    """

    def __init__(self, product):
        """
        Parameters
        ----------
        product : str or pathlib.Path
            path of the S1-ETAD product (it is a directory)
        """
        self.product = pathlib.Path(product)
        self.ds = self._init_measurement_dataset()
        self._annot = self._init_annotation_dataset()
        self.burst_catalogue = self._init_burst_catalogue()

    def _init_measurement_dataset(self):
        """Open the nc dataset."""
        list_ = [i for i in self.product.glob("measurement/*.nc")]
        rootgrp = Dataset(list_[0], "r")
        rootgrp.set_auto_mask(False)
        return rootgrp

    def _init_annotation_dataset(self):
        """Open the xml annotation dataset."""
        list_ = [i for i in self.product.glob("annotation/*.xml")]
        xml_file = str(list_[0])
        root = etree.parse(xml_file).getroot()
        return root

    @functools.lru_cache()
    def __getitem__(self, index):
        assert index in self.swath_list, f"{index} is not in {self.swath_list}"
        return Sentinel1EtadSwath(self.ds[index], self.grid_spacing['y'])

    def __iter__(self):
        yield from self.iter_swaths()

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.product}")  # 0x{id(self):x}'

    def __str__(self):
        return f'{self.__class__.__name__}("{self.product.name}")'

    @property
    def number_of_swath(self):
        return len(self.ds.groups)

    @property
    def swath_list(self):
        return list(self.ds.groups.keys())

    def s1_product_list(self):
        """Return the list of S-1 products used to compose the ETAD one."""
        xp = 'productComponents/inputProductList/inputProduct/productID'
        dd = self._xpath_to_list(self._annot, xp)
        return dd

    @property
    def grid_spacing(self):
        """Return the grid spacing in meters."""
        xp_list = {
            'x': './/correctionGridRangeSampling',
            'y': './/correctionGridAzimuthSampling',
        }
        dd = {}
        for tag, xp in xp_list.items():
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=np.float)
        dd['unit'] = 'm'
        return dd

    @property
    def grid_sampling(self):
        """Return the grid spacing in s."""
        xp_list = {
            'x': './/productInformation/gridSampling/range',
            'y': './/productInformation/gridSampling/azimuth',
        }
        dd = {}
        for tag, xp in xp_list.items():
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=np.float)
        dd['unit'] = 's'
        return dd

    def processing_setting(self):
        """Return the corrections performed.

        Read the xml file to identify the corrections performed.
        If a correction is not performed the matrix is filled with zeros.
        """
        correction_list = [
            'troposphericDelayCorrection', 'ionosphericDelayCorrection',
            'solidEarthTideCorrection', 'bistaticAzimuthCorrection',
            'dopplerShiftRangeCorrection', 'FMMismatchAzimuthCorrection',
        ]
        dd = {}
        xp_root = (
            'processingInformation/processor/setapConfigurationFile/'
            'processorSettings/'
        )
        for correction in correction_list:
            xp = xp_root + correction
            ret = self._xpath_to_list(self._annot, xp)
            if ret == 'true':
                ret = True
            else:
                ret = False
            dd[correction] = ret
        return dd

    def _init_burst_catalogue(self):
        """Build the burst catalog.

        Parses the XML annotation dataset to create a panda DataFrame
        containing all the elements allowing to index properly a burst.
        """
        df = None
        for burst_ in self._annot.findall('.//etadBurst'):
            burst_dict = dict(burst_.find('burstData').attrib)
            burst_dict['productID'] = burst_.find('burstData/productID').text
            burst_dict['swathID'] = burst_.find('burstData/swathID').text
            burst_dict['azimuthTimeMin'] = self._xpath_to_list(
                burst_, 'burstCoverage/temporalCoverage/azimuthTimeMin',
                parse_time_func=lambda x: x)
            burst_dict['azimuthTimeMax'] = self._xpath_to_list(
                burst_, 'burstCoverage/temporalCoverage/azimuthTimeMax',
                parse_time_func=lambda x: x)

            burst_dict['pIndex'] = int(burst_dict['pIndex'])
            burst_dict['sIndex'] = int(burst_dict['sIndex'])
            burst_dict['bIndex'] = int(burst_dict['bIndex'])

            if df is None:
                df = pd.DataFrame(burst_dict, index=[0])
            else:
                df = df.append(burst_dict, ignore_index=True)
        return df

    def query_burst(self, first_time=None, product_name=None, last_time=None,
                    swath=None):
        """Query the burst catalogue to retrieve the burst matching by time.

        Parameters
        ----------
        first_time : datetime
            is set to None then set to the first time
        last_time : datetime
            if set to None the last_time = first_time
        product_name : str
            Name of a real S1 product e.g.
            S1B_IW_SLC__1SDV_20190805T162509_20190805T162...SAFE
        swath : str or list
            list of swathID e.g. 'IW1' or ['IW1'] or ['IW1', 'IW2']

        Returns
        -------
        pandas.DataFrame
            Filtered panda dataframe
        """
        # first sort the burst by time
        df = self.burst_catalogue.sort_values(by=['azimuthTimeMin'])
        if first_time is None:
            first_time = df.iloc[0].azimuthTimeMin
        if last_time is None:
            last_time = df.iloc[-1].azimuthTimeMax

        ix0 = ((df.azimuthTimeMin >= first_time) &
               (df.azimuthTimeMax <= last_time))

        if product_name is not None:
            # build a regex based on the name to avoid issues with annotation
            # products and CRC
            product_name = Sentinel1ProductName(product_name)
            product_name.to_annotation(value='[AD]')
            product_name.crc = ''
            filter_ = product_name.recompose(with_suffix=False)
            ix0 = ix0 & self.burst_catalogue.productID.str.contains(filter_,
                                                                    regex=True)

        if swath is not None:
            if isinstance(swath, str):
                swath = [swath]
            ix0 = ix0 & df.swathID.isin(swath)

        return df.loc[ix0]

    def _selection_to_swath_list(self, selection=None):
        if selection is None:
            selection = self.burst_catalogue

        if isinstance(selection, pd.DataFrame):
            selection = selection.swathID.unique()
        elif isinstance(selection, str):
            selection = [selection]

        return selection

    def iter_swaths(self, selection=None):
        """Iterate over swaths according to the specified selection.

        Parameters
        ----------
        selection : list(str) or pd.Dataframe, optional
            the list of selected swath IDs or the result of a
            Sentinel1Etad.query_burst query.
            If the selection is None (default) the iteration is performed
            on all the swaths of the product.
        """
        selection = self._selection_to_swath_list(selection)
        for swath_name in selection:
            yield self[swath_name]

    def xpath_to_list(self, xpath, dtype=None, namespace=None,
                      parse_time_func=None):
        return self._xpath_to_list(self._annot, xpath, dtype=dtype,
                                   namespace=namespace,
                                   parse_time_func=parse_time_func)

    @staticmethod
    def _xpath_to_list(root, xpath, dtype=None, namespace=None,
                       parse_time_func=None):

        ll = [elt.text for elt in root.findall(xpath, namespace)]
        if parse_time_func is not None:
            ll = [parser.parse(t) for t in ll]
            ll = parse_time_func(ll)
        ll = np.asarray(ll, dtype=dtype)

        if ll.size == 1:
            return ll.item(0)
        else:
            return ll

    def get_footprint(self, swath_list=None):  # , merge=False  # TODO
        """Return the footprints of all the bursts as MultiPolygon.

        It calls in the back the get_footprint of the Sentinel1EtadBurst class.
        """
        if swath_list is None:
            swath_list = self.swath_list

        polys = []
        for swath_name in swath_list:
            for poly_ in self[swath_name].get_footprint():
                polys.append(poly_)

        # TODO: check, Sentinel1EtadSwath.get_footprint returns a MultiPolygon
        return polys

    def _swath_merger(self, burst_var, selection=None, set_auto_mask=False,
                      transpose=True, meter=False):
        if selection is None:
            df = self.burst_catalogue
        elif not isinstance(selection, pd.DataFrame):
            df = self.query_burst(swath=selection)
        else:
            assert isinstance(selection, pd.DataFrame)
            df = selection

        # NOTE: assume a specific order of swath IDs
        first_swath = self[df.swathID.min()]
        near_burst = first_swath[first_swath.burst_list[0]]
        last_swath = self[df.swathID.max()]
        far_burst = last_swath[last_swath.burst_list[0]]

        rg_first_time = near_burst.sampling_start['x']
        rg_last_time = (far_burst.sampling_start['x'] +
                        far_burst.sampling['x'] * far_burst.samples)
        az_first_time = df.azimuthTimeMin.min()
        az_last_time = df.azimuthTimeMax.max()
        az_ref_time = parser.parse(self.ds.azimuthTimeMin)
        az_first_time_rel = (az_first_time - az_ref_time).total_seconds()

        sampling = self.grid_sampling
        dx = sampling['x']
        dy = sampling['y']

        num_samples = np.round(
            (rg_last_time - rg_first_time) / dx
        ).astype(np.int) + 1
        num_lines = np.round(
            (az_last_time - az_first_time).total_seconds() / dy
        ).astype(np.int) + 1

        img = np.zeros((num_lines, num_samples))

        for swath in self.iter_swaths(selection):
            # NOTE: use the private "Sentinel1EtadSwath._burst_merger" method
            # to be able to work only on the specified NetCDF variable
            dd_ = swath._burst_merger(burst_var, selection=selection,  # noqa
                                      set_auto_mask=set_auto_mask,
                                      transpose=transpose, meter=meter)
            yoffset = dd_['first_azimuth_time'] - az_first_time_rel
            xoffset = dd_['first_slant_range_time'] - rg_first_time
            line_ofs = np.round(yoffset / dy).astype(np.int)
            sample_ofs = np.round(xoffset / dx).astype(np.int)

            slice_y = slice(line_ofs, line_ofs + dd_[burst_var].shape[0])
            slice_x = slice(sample_ofs, sample_ofs + dd_[burst_var].shape[1])

            img[slice_y, slice_x] = dd_[burst_var]

        return {
            burst_var: img,
            'first_azimuth_time': az_first_time,
            'first_slant_range_time': rg_first_time,
            'sampling': sampling,
        }

    def _core_merge_correction(self, prm_list, selection=None,
                               set_auto_mask=True, transpose=True, meter=False):
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self._swath_merger(field, selection=selection,
                                     set_auto_mask=set_auto_mask,
                                     transpose=transpose, meter=meter)
            dd[dim] = dd_[field]
            dd['sampling'] = dd_['sampling']
            dd['first_azimuth_time'] = dd_['first_azimuth_time']
            dd['first_slant_range_time'] = dd_['first_slant_range_time']

        dd['unit'] = 'm' if meter else 's'

        # To compute lat/lon/h make a new selection with all gaps filled
        swath_list = self._selection_to_swath_list(selection)
        near_swath = min(swath_list)
        far_swath = max(swath_list)
        idx = self.burst_catalogue.swathID >= near_swath
        idx &= self.burst_catalogue.swathID <= far_swath
        swaths = self.burst_catalogue.swathID[idx].unique()

        data = dd['x' if 'x' in prm_list else 'y']
        lines = data.shape[0]
        duration = lines * self.grid_sampling['y']
        duration = np.float64(duration * 1e9).astype('timedelta64[ns]')
        first_time = dd['first_azimuth_time']
        last_time = first_time + duration

        filled_selection = self.query_burst(first_time=first_time,
                                            last_time=last_time, swath=swaths)

        dd['lats'] = self._swath_merger('lats', selection=filled_selection,
                                        set_auto_mask=set_auto_mask,
                                        transpose=transpose, meter=False)
        dd['lons'] = self._swath_merger('lons', selection=filled_selection,
                                        set_auto_mask=set_auto_mask,
                                        transpose=transpose, meter=False)
        dd['height'] = self._swath_merger('height', selection=filled_selection,
                                          set_auto_mask=set_auto_mask,
                                          transpose=transpose, meter=False)
        return dd

    def merge_correction(self, name: CorrectionType = ECorrectionType.SUM,
                         selection=None, set_auto_mask=True, transpose=True,
                         meter=False):
        correction_type = ECorrectionType(name)  # check values
        prm_list = _CORRECTION_NAMES_MAP[correction_type.value]
        correction = self._core_merge_correction(prm_list, selection=selection,
                                                 set_auto_mask=set_auto_mask,
                                                 transpose=transpose,
                                                 meter=meter)
        correction['name'] = correction_type.value
        return correction

    def to_kml(self, kml_file):
        kml = simplekml.Kml()

        # get the footprints
        burst_ftps = self.get_footprint()
        kml_dir = kml.newfolder(name=f"Sentinel1 Timing Correction Grid ")
        for ix, ftp in enumerate(burst_ftps):
            x, y = ftp.exterior.xy
            corner = [(x[i], y[i]) for i in range(len(x))]

            pol = kml_dir.newpolygon(name=str(ix))
            pol.outerboundaryis = corner
            pol.altitudeMode = 'absolute'
            pol.tessellate = 1
            pol.polystyle.fill = 0
            pol.style.linestyle.width = 2

        if kml_file is not None:
            kml.save(kml_file)

        return kml


_CORRECTION_NAMES_MAP = {
    'tropospheric': {'x': 'troposphericCorrectionRg'},
    'ionospheric': {'x': 'ionosphericCorrectionRg'},
    'geodetic': {'x': 'geodeticCorrectionRg', 'y': 'geodeticCorrectionAz'},
    'bistatic': {'y': 'bistaticCorrectionAz'},
    'doppler': {'x': 'dopplerRangeShiftRg'},
    'fmrate': {'y': 'fmMismatchCorrectionAz'},
    'sum': {'x': 'sumOfCorrectionsRg', 'y': 'sumOfCorrectionsAz'},
}


class Sentinel1EtadSwath:
    def __init__(self, nc_group, daz_m):
        self._grp = nc_group
        self._daz_m = daz_m

    @functools.lru_cache()
    def __getitem__(self, burst_index):
        burst_name = f"Burst{burst_index:04d}"
        return Sentinel1EtadBurst(self._grp[burst_name], self._daz_m)

    def __iter__(self):
        yield from self.iter_bursts()

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._grp.path}")  0x{id(self):x}'

    @property
    def burst_list(self):
        return [burst.bIndex for burst in self._grp.groups.values()]

    @property
    def number_of_burst(self):
        return len(self._grp.groups)

    @property
    def swath_id(self):
        return self._grp.swathID

    @property
    def sampling_start(self):
        """Relative sampling start times."""
        first_burst_index = self.burst_list[0]
        first_burst = self[first_burst_index]
        return first_burst.sampling_start

    @property
    def sampling(self):
        first_burst_index = self.burst_list[0]
        first_burst = self[first_burst_index]
        return first_burst.sampling

    def _selection_to_burst_index_list(self, selection=None):
        if selection is None:
            index_list = self.burst_list
        elif isinstance(selection, pd.DataFrame):
            idx = selection.swathID == self.swath_id
            index_list = selection.bIndex[idx].values
        else:
            index_list = selection
        return index_list

    def iter_bursts(self, selection=None):
        """Iterate over bursts according to the specified selection.

        Parameters
        ----------
        selection : list(int) or pd.Dataframe, optional
            the list of selected bursts or result of a
            Sentinel1Etad.query_burst query.
            If the selection is None (default) the iteration is performed
            on all the burst of the swath.
        """
        index_list = self._selection_to_burst_index_list(selection)
        for burst_index in index_list:
            yield self[burst_index]

    def get_footprint(self, burst_index_list=None):
        """Return the footprints of all the bursts as MultiPolygon.

        It calls in the back the get_footprint of the Sentinel1EtadBurst class.
        """
        if burst_index_list is None:
            burst_index_list = self.burst_list

        footprints = [self[bix].get_footprint() for bix in burst_index_list]
        return MultiPolygon(footprints)

    def _burst_merger(self, burst_var, selection=None,
                      az_time_min=None, az_time_max=None,
                      set_auto_mask=False, transpose=True, meter=False):
        """Low level method to de-burst a NetCDF variable.

        The de-burst strategy is simple as the latest line is on top of the
        oldest.

        Parameters
        ----------
        burst_var : str
            one of the burst netcdf variables
        selection : list or pandas.DataFRame
            list of selected bursts (by default all bursts are selected)
        az_time_min : float
            minimum azimuth time of the merged swath
             (relative to the reference annotated in the NetCDF root)
        az_time_max : float
            maximum azimuth tim eof the merged swath
            (relative to the reference annotated in the NetCDF root)
        set_auto_mask : bool
            requested for netCDF4 to avoid retrieving a masked array
        transpose : bool
            requested to retrieve the correction in array following the
            numpy convention for dimensions
        meter : bool
            transform the result in meters

        Returns
        -------
        dict
            a dictionary containing merged data and sampling information:

            :<burst_var_name>: merged data for the selected burst_var
            :first_azimuth_time: the relative azimuth first time
            :first_slant_range_time: the relative (slant) range first time
            :sampling: a dictionary containing the sampling along the
            'x' and 'y' directions and the 'unit'
        """
        burst_index_list = self._selection_to_burst_index_list(selection)

        # Find what is the extent of the acquisition in azimuth
        first_burst = self[burst_index_list[0]]
        last_burst = self[burst_index_list[-1]]

        if az_time_min is None:
            t0 = first_burst.sampling_start['y']
        else:
            t0 = az_time_min

        last_azimuth, _ = last_burst.get_burst_grid()
        if az_time_max is None:
            t1 = last_azimuth[-1]
        else:
            t1 = az_time_max

        # azimuth grid sampling
        dt = first_burst.sampling['y']

        num_lines = np.round((t1 - t0) / dt).astype(np.int) + 1
        num_samples = first_burst.samples

        debursted_var = np.zeros((num_lines, num_samples))

        for burst_ in self.iter_bursts(burst_index_list):
            assert(dt == burst_.sampling['y']), \
                'The azimuth sampling is changing long azimuth'
            assert(first_burst.sampling_start['x'] ==
                   burst_.sampling_start['x']), \
                'The 2-way range gridStartRangeTime0 is changing long azimuth'

            # get the timing of the burst and convert into line index
            az_time_, rg_time_ = burst_.get_burst_grid()
            line_index_ = np.round((az_time_ - t0) / dt).astype(np.int)

            # NOTE: use the private "Sentinel1EtadBurst._get_etad_param" method
            # to be able to work only on the specified NetCDF variable
            var_ = burst_._get_etad_param(burst_var,  # noqa
                                          set_auto_mask=set_auto_mask,
                                          transpose=transpose, meter=meter)

            debursted_var[line_index_, :] = var_

        return {
            burst_var: debursted_var,
            'first_azimuth_time': t0,
            'first_slant_range_time': first_burst.sampling_start['x'],
            'sampling': first_burst.sampling,
        }

    def _core_merge_correction(self, prm_list, selection=None,
                               set_auto_mask=True, transpose=True, meter=False):
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self._burst_merger(field, selection=selection,
                                     set_auto_mask=set_auto_mask,
                                     transpose=transpose, meter=meter)
            dd[dim] = dd_[field]
            dd['sampling'] = dd_['sampling']
            dd['first_azimuth_time'] = dd_['first_azimuth_time']
            dd['first_slant_range_time'] = dd_['first_slant_range_time']

        dd['unit'] = 'm' if meter else 's'
        dd['lats'] = self._burst_merger('lats', transpose=transpose,
                                        meter=False,
                                        set_auto_mask=set_auto_mask)
        dd['lons'] = self._burst_merger('lons', transpose=transpose,
                                        meter=False,
                                        set_auto_mask=set_auto_mask)
        dd['height'] = self._burst_merger('height', transpose=transpose,
                                          meter=False,
                                          set_auto_mask=set_auto_mask)
        return dd

    def merge_correction(self, name: CorrectionType = ECorrectionType.SUM,
                         selection=None, set_auto_mask=True,
                         transpose=True, meter=False):
        correction_type = ECorrectionType(name)  # check values
        prm_list = _CORRECTION_NAMES_MAP[correction_type.value]
        correction = self._core_merge_correction(prm_list,
                                                 selection=selection,
                                                 set_auto_mask=set_auto_mask,
                                                 transpose=transpose,
                                                 meter=meter)
        correction['name'] = correction_type.value
        return correction


class Sentinel1EtadBurst:
    def __init__(self, nc_group, daz_m):
        self._grp = nc_group
        self._daz_m = daz_m

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._grp.path}")  0x{id(self):x}'

    @property
    def swath_id(self):
        return self._grp.swathID

    @property
    def burst_index(self):
        return self._grp.bIndex

    def get_footprint(self):
        """Return the footprint of ghe bursts as shapely.Polygon.

        It gets the lat / lon / height grid and extract the 4 corners.
        """
        lats, lons, heights = self.get_lat_lon_height()
        corner_list = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
        etaf_burst_footprint = []
        for corner in corner_list:
            lat_, lon_, h_ = lats[corner], lons[corner], heights[corner]
            etaf_burst_footprint.append((lon_, lat_, h_))
        etaf_burst_footprint = Polygon(etaf_burst_footprint)
        return etaf_burst_footprint

    def get_burst_grid(self):
        """Return the t, tau grid of the burst."""
        azimuth = self._get_etad_param('azimuth', set_auto_mask=True)
        range_ = self._get_etad_param('range', set_auto_mask=True)
        return azimuth, range_

    @property
    def sampling_start(self):
        """Relative sampling start times."""
        return dict(
            x=self._grp.gridStartRangeTime0,
            y=self._grp.gridStartAzimuthTime0,
            units='s',
        )

    @property
    def sampling(self):
        return dict(
            x=self._grp.gridSamplingRange,
            y=self._grp.gridSamplingAzimuth,
            units='s',
        )

    @property
    def lines(self):
        return self._grp.dimensions['azimuthExtent'].size

    @property
    def samples(self):
        return self._grp.dimensions['rangeExtent'].size

    def _get_etad_param(self, name, set_auto_mask=False, transpose=True,
                        meter=False):
        assert name in self._grp.variables, f'Parameter {name!r} is not allowed'

        self._grp.set_auto_mask(set_auto_mask)

        field = np.asarray(self._grp[name])
        if transpose:
            field = np.transpose(field)

        if meter:
            if name.endswith('Az'):
                k = self._daz_m / self.sampling['y']
            elif name.endswith('Rg'):
                k = constants.c / 2
            else:
                # it is not a correction (azimuth, range, lats, lons, height)
                k = 1
                warnings.warn(
                    f'the {name} is not a correction: '
                    'the "meter" parameter will be ignored')
            field *= k

        return field

    def get_lat_lon_height(self, transpose=True):
        lats = self._get_etad_param(
            'lats', transpose=transpose, meter=False, set_auto_mask=True)
        lons = self._get_etad_param(
            'lons', transpose=transpose, meter=False, set_auto_mask=True)
        h = self._get_etad_param(
            'height', transpose=transpose, meter=False, set_auto_mask=True)
        return lats, lons, h

    def _core_get_correction(self, prm_list, set_auto_mask=False,
                             transpose=True, meter=False):
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self._get_etad_param(
                field, set_auto_mask=set_auto_mask, transpose=transpose,
                meter=meter)

        correction['unit'] = 'm' if meter else 's'

        return correction

    def get_correction(self, name: CorrectionType = ECorrectionType.SUM,
                       set_auto_mask=False, transpose=True, meter=False):
        """Retrieve the correction for the specified correction "name".

        Puts the results in a dict.

        Parameters
        ----------
        name : ECorrectionType or str
            the desired correction
        set_auto_mask : bool
            requested for netCDF4 to avoid retrieving a masked array
        transpose : bool
            requested to retrieve the correction in array following the
            numpy convention for dimensions
        meter : bool
            transform the result in meters

        Returns
        -------
        dict
            a dictionary containing the following items for the
            requested correction:

            :x: correction in range (if applicable)
            :y: correction in azimuth (if applicable)
            :unit: 'm' or 's'
            :name: name of the correction
        """
        correction_type = ECorrectionType(name)  # check values
        name = correction_type.value
        prm_list = _CORRECTION_NAMES_MAP[name]
        correction = self._core_get_correction(prm_list,
                                               set_auto_mask=set_auto_mask,
                                               transpose=transpose, meter=meter)
        correction['name'] = name
        return correction


class Sentinel1ProductName:
    """Class to manipulate the filename of Sentinel 1 products."""

    def __init__(self, product_name):
        self.__product_name = pathlib.Path(product_name)
        self.suffix = self.__product_name.suffix
        self.file_name = self.__product_name.stem
        self._parts = self.file_name.split('_')

        # trick for SLC
        if len(self._parts) == 10 and 'SLC' in self.ptype:
            del self._parts[3]
            self.ptype = 'SLC_'

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}("{self.__product_name}")  # 0x{id(self):x}'

    @property
    def mission(self):
        return self._parts[0]

    @mission.setter
    def mission(self, value):
        self._parts[0] = value

    @property
    def mode(self):
        return self._parts[1]

    @mode.setter
    def mode(self, value):
        self._parts[1] = value

    @property
    def ptype(self):
        return self._parts[2]

    @ptype.setter
    def ptype(self, value):
        self._parts[2] = value

    @property
    def typepol(self):
        return self._parts[3]

    @typepol.setter
    def typepol(self, value):
        self._parts[3] = value

    @property
    def start_time(self):
        return self._parts[4]

    @start_time.setter
    def start_time(self, value):
        self._parts[4] = value

    @property
    def stop_time(self):
        return self._parts[5]

    @stop_time.setter
    def stop_time(self, value):
        self._parts[5] = value

    @property
    def orbit(self):
        return self._parts[6]

    @orbit.setter
    def orbit(self, value):
        self._parts[6] = value

    @property
    def dtid(self):
        return self._parts[7]

    @dtid.setter
    def dtid(self, value):
        self._parts[7] = value

    @property
    def crc(self):
        return self._parts[8]

    @crc.setter
    def crc(self, value):
        self._parts[8] = value

    def is_annotation(self):
        if self.typepol[1] == 'A':
            return True
        else:
            return False

    def to_annotation(self, value='A'):
        ll = list(self.typepol)
        ll[1] = value
        self.typepol = ''.join(ll)

    def to_standard(self):
        ll = list(self.typepol)
        ll[1] = 'S'
        self.typepol = ''.join(ll)

    def recompose(self, with_suffix=True):
        pp = '_'.join(self._parts)
        if with_suffix:
            pp += self.suffix
        return pp


# --- Jupyter integration -----------------------------------------------------
def _sentinel1_etad_repr_pretty_(obj, p, cycle):
    if cycle:
        p.text(repr(obj))
    else:
        p.text(repr(obj))
        p.break_()
        plist = obj.s1_product_list()
        if isinstance(plist, str):
            plist = [plist]
        with p.group(2, 'Sentinel-1 products list:'):
            for name in plist:
                p.break_()
                p.text(name)
        p.break_()
        p.text(f'Number of swaths: {obj.number_of_swath}')
        p.break_()
        p.text('Swath list: {}'.format(', '.join(obj.swath_list)))
        p.break_()
        with p.group(2, 'Grid sampling:'):
            for key, value in obj.grid_sampling.items():
                p.break_()
                p.text(f'{key}: {value}')
        p.break_()
        with p.group(2, 'Grid spacing:'):
            for key, value in obj.grid_spacing.items():
                p.break_()
                p.text(f'{key}: {value}')
        p.break_()
        with p.group(2, 'Processing settings:'):
            for key, value in obj.processing_setting().items():
                p.break_()
                p.text(f'{key}: {value}')


def _sentinel1_etad_swath_repr_pretty_(obj, p, cycle):
    if cycle:
        p.text(repr(obj))
    else:
        p.text(repr(obj))
        p.break_()
        p.text(f'Swaths ID: {obj.swath_id}')
        p.break_()
        p.text(f'Number of bursts: {obj.number_of_burst}')
        p.break_()
        p.text('Burst list: ' + str(obj.burst_list))
        p.break_()
        with p.group(2, 'Sampling start:'):
            for key, value in obj.sampling_start.items():
                p.break_()
                p.text(f'{key}: {value}')
        p.break_()
        with p.group(2, 'Sampling:'):
            for key, value in obj.sampling.items():
                p.break_()
                p.text(f'{key}: {value}')


def _sentinel1_etad_burst_repr_pretty_(obj, p, cycle):
    if cycle:
        p.text(repr(obj))
    else:
        p.text(repr(obj))
        p.break_()
        p.text(f'Swaths ID: {obj.swath_id}')
        p.break_()
        p.text(f'Burst index: {obj.burst_index}')
        p.break_()
        p.text(f'Shape: ({obj.lines}, {obj.samples})')
        p.break_()
        with p.group(2, 'Sampling start:'):
            for key, value in obj.sampling_start.items():
                p.break_()
                p.text(f'{key}: {value}')
        p.break_()
        with p.group(2, 'Sampling:'):
            for key, value in obj.sampling.items():
                p.break_()
                p.text(f'{key}: {value}')


def _register_jupyter_formatters():
    try:
        ipy = get_ipython()  # noqa
    except NameError:
        return False
    else:
        formatter = ipy.display_formatter.formatters['text/plain']
        formatter.for_type(
            Sentinel1Etad, _sentinel1_etad_repr_pretty_)
        formatter.for_type(
            Sentinel1EtadSwath, _sentinel1_etad_swath_repr_pretty_)
        formatter.for_type(
            Sentinel1EtadBurst, _sentinel1_etad_burst_repr_pretty_)
        return True


_register_jupyter_formatters()
