"""S1-ETAD data model core classes."""

import enum
import pathlib
import datetime
import warnings
import functools
import itertools
import collections
from typing import Union

import numpy as np
from scipy import constants
from lxml import etree
from netCDF4 import Dataset

import pandas as pd

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
import shapely.ops

from ._s1utils import Sentinel1ProductName


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


_CORRECTION_NAMES_MAP = {
    'tropospheric': {'x': 'troposphericCorrectionRg'},
    'ionospheric': {'x': 'ionosphericCorrectionRg'},
    'geodetic': {'x': 'geodeticCorrectionRg', 'y': 'geodeticCorrectionAz'},
    'bistatic': {'y': 'bistaticCorrectionAz'},
    'doppler': {'x': 'dopplerRangeShiftRg'},
    'fmrate': {'y': 'fmMismatchCorrectionAz'},
    'sum': {'x': 'sumOfCorrectionsRg', 'y': 'sumOfCorrectionsAz'},
}


_STATS_TAG_MAP = {
    ECorrectionType.TROPOSPHERIC: 'troposphericCorrection',
    ECorrectionType.IONOSPHERIC: 'ionosphericCorrection',
    ECorrectionType.GEODETIC: 'geodeticCorrection',
    ECorrectionType.BISTATIC: 'bistaticCorrection',
    ECorrectionType.DOPPLER: 'dopplerRangeShift',
    ECorrectionType.FMRATE: 'fmMismatchCorrection',
    ECorrectionType.SUM: 'sumOfCorrections',
}


Statistics = collections.namedtuple('Statistics', ['min', 'mean', 'max'])


class Sentinel1Etad:
    """Sentinel-1 ETAD product.

    Class to decode and access the elements of the Sentinel ETAD product
    which specification is governed by ETAD-DLR-PS-0014.

    The index operator [] (implemented with the __getitem__ method) returns
    a Sentinel1EtadSwath instance.

    Parameters
    ----------
    product : str or pathlib.Path
        path of the S1-ETAD product (it is a directory)

    Attributes
    ----------
    product : pathlib.Path
        path of the S1-ETAD product (it is a directory)
    burst_catalogue : pandas.DataFrame
        dataframe containing main information of all bursts present in
        the product
    ds : netCDF.Dataset
        (provisional) the NetCDF.Dataset in which data are stored
    """

    def __init__(self, product):
        # TODO: make this read-only (property)
        self.product = pathlib.Path(product)
        # TODO: ds should not be exposed
        self.ds = self._init_measurement_dataset()
        self._annot = self._init_annotation_dataset()
        self.burst_catalogue = self._init_burst_catalogue()

    def _init_measurement_dataset(self):
        """Open the nc dataset."""
        # @TODO: retrieve form manifest
        netcdf_file = next(self.product.glob("measurement/*.nc"))
        rootgrp = Dataset(netcdf_file, "r")
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
        return Sentinel1EtadSwath(self.ds[index])

    def __iter__(self):
        yield from self.iter_swaths()

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.product}")  # 0x{id(self):x}'

    def __str__(self):
        return f'{self.__class__.__name__}("{self.product.name}")'

    @property
    def number_of_swath(self):
        """The number of swaths in the product."""
        return len(self.ds.groups)

    @property
    def swath_list(self):
        """The list of swath identifiers (str) in the product."""
        return list(self.ds.groups.keys())

    def s1_product_list(self):
        """Return the list of S-1 products used to compose the ETAD one."""
        df = self.burst_catalogue

        # this ensures that each product name is located at the correct pIndex
        product_list = [
            item[1] for item in sorted(set(zip(df['pIndex'], df['productID'])))
        ]

        return product_list

    @property
    def grid_spacing(self):
        """Return the grid spacing in meters."""
        xp_list = {
            'x': './/correctionGridRangeSampling',
            'y': './/correctionGridAzimuthSampling',
        }
        dd = {}
        for tag, xp in xp_list.items():
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=float)
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
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=float)
        dd['unit'] = 's'
        return dd

    @property
    def min_azimuth_time(self):
        """The minimum azimuth time of all bursts in the product."""
        return datetime.datetime.fromisoformat(self.ds.azimuthTimeMin)

    @property
    def max_azimuth_time(self):
        """The maximum azimuth time of all bursts in the product."""
        return datetime.datetime.fromisoformat(self.ds.azimuthTimeMax)

    @property
    def min_range_time(self):
        """The minimum range time of all bursts in the product."""
        return self.ds.rangeTimeMin

    @property
    def max_range_time(self):
        """The maximum range time of all bursts in the product."""
        return self.ds.rangeTimeMax

    @property
    def vg(self):
        """Mean ground velocity [m/s]."""
        try:
            xp = (
                'productInformation/gridGroundSampling/'
                'averageZeroDopplerVelocity'
            )
            vg = float(self._annot.find(xp).taxt)
        except (AttributeError, ValueError):
            vg = self.grid_spacing['y'] / self.grid_sampling['y']
        return vg

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

        Using information stored in the NetCDF file create a
        pandas.DataFrame containing all the elements allowing to index
        properly a burst.
        """
        def _to_tdelta64(t):
            return np.float64(t * 1e9).astype('timedelta64[ns]')

        data = collections.defaultdict(list)
        t0 = np.datetime64(self.ds.azimuthTimeMin, 'ns')
        for swath in self.ds.groups.values():
            for burst in swath.groups.values():
                ax = burst.variables['azimuth']
                tmin = t0 + _to_tdelta64(ax[0])
                tmax = t0 + _to_tdelta64(ax[-1])

                data['bIndex'].append(burst.bIndex)
                data['pIndex'].append(burst.pIndex)
                data['sIndex'].append(burst.sIndex)
                data['productID'].append(burst.productID)
                data['swathID'].append(burst.swathID)
                data['azimuthTimeMin'].append(tmin)
                data['azimuthTimeMax'].append(tmax)

        df = pd.DataFrame(data=data)

        return df

    def query_burst(self, first_time=None, product_name=None, last_time=None,
                    swath=None, geometry=None):
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
        geometry : shapely.geometry.[Point, Polygon, ...]
            A shapely geometry for which interstion will be searched

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
            product_name.to_annotation(value='[AS]')
            product_name.crc = ''
            filter_ = product_name.recompose(with_suffix=False)
            ix0 = ix0 & self.burst_catalogue.productID.str.contains(filter_,
                                                                    regex=True)

        if swath is not None:
            if isinstance(swath, str):
                swath = [swath]
            ix0 = ix0 & df.swathID.isin(swath)

        if geometry is not None:
            bix_list = self.intersects(geometry)
            ix0 = ix0 & df.bIndex.isin(bix_list)

        return df.loc[ix0]

    def _selection_to_swath_list(self, selection=None):
        if selection is None:
            selection = self.burst_catalogue

        if isinstance(selection, pd.DataFrame):
            burst_selection = selection
            swath_list = selection.swathID.unique()
        elif isinstance(selection, str):
            burst_selection = None
            swath_list = [selection]
        else:
            # assume it is a list of swaths already
            import collections.abc
            assert isinstance(selection, collections.abc.Iterable)
            assert all(isinstance(item, str) for item in selection)
            burst_selection = None
            swath_list = selection

        return swath_list, burst_selection

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
        swath_list, _ = self._selection_to_swath_list(selection)
        for swath_name in swath_list:
            yield self[swath_name]

    def iter_bursts(self, selection=None):
        """Iterate over burst according to the specified selection.

        Parameters
        ----------
        selection : list(int) or pd.Dataframe, optional
            the list of selected burst indexes or the result of a
            Sentinel1Etad.query_burst query.
            If the selection is None (default) the iteration is performed
            on all the bursts of the product.
        """
        if selection is None:
            selection = self.burst_catalogue
        elif not isinstance(selection, pd.DataFrame):
            # assume it is a list of burst indexes
            bursts = selection
            if isinstance(bursts, int):
                bursts = [selection]
            # NOTE: preserve the order
            selection = self.burst_catalogue.bIndex.isin(bursts)

        assert isinstance(selection, pd.DataFrame)

        for idx, row in selection.iterrows():
            burst = self[row.swathID][row.bIndex]
            yield burst

    @staticmethod
    def _xpath_to_list(root, xpath, dtype=None, namespace=None,
                       parse_time_func=None):

        ll = [elt.text for elt in root.findall(xpath, namespace)]
        if parse_time_func is not None:
            ll = [datetime.datetime.fromisoformat(t) for t in ll]
            ll = parse_time_func(ll)  # TODO: check
        ll = np.asarray(ll, dtype=dtype)

        if ll.size == 1:
            return ll.item(0)
        else:
            return ll

    def get_statistics(self, correction, meter=False):
        """Return the global statistic value of the specified correction.

        The returned value is the pre-computed one that is stored in the
        XML annotation file of the product.

        Parameters
        ----------
        correction : str or ECorrectionType
            the corrections for which the statistic value is requested
        meter : bool
            if set to True then the returned value is expressed in meters,
            otherwise it is expressed in seconds (default: False)

        Returns
        -------
        dict
            a dictionary containing :class:`Statistics` (min, mean and max)
            for all available components of the specified correction:

            :x:
                a :class:`Statistics` instance relative to the range
                component of the specified correction
            :y:
                a :class:`Statistics` instance relative to the azimuth
                component of the specified correction
            :unit:
                the units of the returned statistics ("m" or "s")
        """
        units = 'm' if meter else 's'

        stat_xp = './qualityAndStatistics'
        target = ECorrectionType(correction)
        target_tag = _STATS_TAG_MAP[target]

        statistics = {'unit': units}

        # NOTE: looping on element and heuristic test on tags is necessary
        #       due to inconsistent naming of range and azimuth element
        # TODO: report the inconsistency to DLR? (TBD)
        correction_elem = self._annot.find(f'{stat_xp}/{target_tag}')
        for elem in correction_elem:
            if 'range' in elem.tag:
                direction = 'x'
            elif 'azimuth' in elem.tag:
                direction = 'y'
            else:
                continue

            statistics[direction] = Statistics(
                float(elem.findtext(f'min[@unit="{units}"]')),
                float(elem.findtext(f'mean[@unit="{units}"]')),
                float(elem.findtext(f'max[@unit="{units}"]')),
            )

        return statistics

    def get_footprint(self, selection=None, merge=False):
        """Return the footprints of all the bursts as MultiPolygon.

        It calls in the back the get_footprint of the Sentinel1EtadBurst class.

        Parameters
        ----------
        selection : list(str) or pd.Dataframe, optional
            the list of selected swath IDs or the result of a
            Sentinel1Etad.query_burst query.
            If the selection is None (default) the iteration is performed
            on all the swaths of the product.
        merge : bool
            if set to True return a single polygon that is the union of the
            footprints of all bursts
        """
        polys = []
        swath_list, burst_selection = self._selection_to_swath_list(selection)
        for swath in self.iter_swaths(swath_list):
            polys.extend(swath.get_footprint(burst_selection))

        if merge:
            polys = shapely.ops.cascaded_union(polys)
        else:
            polys = MultiPolygon(polys)

        return polys

    def intersects(self, geometry: BaseGeometry):
        """Return the list of burst indexes intersecting the input geometry.

        Computes the intersection of the footprint of the swath (all bursts)
        with the input geometry.

        Parameters
        ----------
        geometry : shapely.geometry.[Point, Polygon, MultiPolygon, line]

        Returns
        -------
        list
            list of all the burst intersecting with the input shape geometry
        """
        lists_of_burst_indexes = [
            swath.intersects(geometry) for swath in self.iter_swaths()
        ]
        # return the flattened list
        return list(itertools.chain(*lists_of_burst_indexes))

    def _swath_merger(self, burst_var, selection=None, set_auto_mask=False,
                      meter=False, fill_value=0.):
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
        az_ref_time = self.min_azimuth_time
        az_first_time_rel = (az_first_time - az_ref_time).total_seconds()

        sampling = self.grid_sampling
        dx = sampling['x']
        dy = sampling['y']

        num_samples = np.round(
            (rg_last_time - rg_first_time) / dx
        ).astype(int) + 1
        num_lines = np.round(
            (az_last_time - az_first_time).total_seconds() / dy
        ).astype(int) + 1

        img = np.full((num_lines, num_samples), fill_value=fill_value)
        # TODO: add some control option
        img = np.ma.array(img, mask=True, fill_value=fill_value)

        for swath in self.iter_swaths(df):
            # NOTE: use the private "Sentinel1EtadSwath._burst_merger" method
            # to be able to work only on the specified NetCDF variable
            dd_ = swath._burst_merger(burst_var, selection=df,  # noqa
                                      set_auto_mask=set_auto_mask, meter=meter)
            yoffset = dd_['first_azimuth_time'] - az_first_time_rel
            xoffset = dd_['first_slant_range_time'] - rg_first_time
            line_ofs = np.round(yoffset / dy).astype(int)
            sample_ofs = np.round(xoffset / dx).astype(int)

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
                               set_auto_mask=True, meter=False):
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self._swath_merger(field, selection=selection,
                                     set_auto_mask=set_auto_mask, meter=meter)
            dd[dim] = dd_[field]
            dd['sampling'] = dd_['sampling']
            dd['first_azimuth_time'] = dd_['first_azimuth_time']
            dd['first_slant_range_time'] = dd_['first_slant_range_time']

        dd['unit'] = 'm' if meter else 's'

        # To compute lat/lon/h make a new selection with all gaps filled
        swath_list, _ = self._selection_to_swath_list(selection)
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
                                        meter=False, fill_value=np.nan)['lats']
        dd['lons'] = self._swath_merger('lons', selection=filled_selection,
                                        set_auto_mask=set_auto_mask,
                                        meter=False, fill_value=np.nan)['lons']
        dd['height'] = self._swath_merger('height', selection=filled_selection,
                                          set_auto_mask=set_auto_mask,
                                          meter=False,
                                          fill_value=np.nan)['height']
        return dd

    def merge_correction(self, name: CorrectionType = ECorrectionType.SUM,
                         selection=None, set_auto_mask=True, meter=False,
                         direction=None):
        """Merge multiple swaths of the specified correction variable.

        Data of the selected swaths (typically overlapped) are merged
        together to form a single data matrix with a consistent (range and
        azimuth) time axis.

        Note
        ----

        The current implementation uses a very simple algorithm that
        iterates over selected swaths and bursts and stitches correction
        data together.

        In overlapping regions, new data simpy overwrite the old ones.
        This is an easy algorithm and perfectly correct for atmospheric
        and geodetic correction.

        It is, instead, sub-optimal for system corrections (bi-static,
        Doppler, FM Rate) which have different values in overlapping
        regions. In this case results are *not* correct.

        Parameters
        ----------
        name : str or CorrectionType
            the name of the desired correction
        selection : list or pandas.DataFrame
            list of selected bursts (by default all bursts are selected)
        set_auto_mask : bool
            requested for netCDF4 to avoid retrieving a masked array
        meter : bool
            transform the result in meters
        direction : str or None
            if set to "x" (for range) or "y" (for "azimuth") only extracts
            the specified correction component.
            By default (None) all available components are returned.

        Returns
        -------
        dict
            a dictionary containing merged data and sampling information:

            :<burst_var_name>:
                merged data for the selected burst_var
            :first_azimuth_time:
                the relative azimuth first time
            :first_slant_range_time:
                the relative (slant) range first time
            :sampling:
                a dictionary containing the sampling along the
                'x' and 'y' directions and the 'unit'
            :units:
                of the correction (seconds or meters)
            :lats:
                the matrix of latitude values (in degrees) for each point
            :lons:
                the matrix of longitude values (in degrees) for each point
            :height:
                the matrix of height values (in meters) for each point
        """
        correction_type = ECorrectionType(name)  # check values
        prm_list = _CORRECTION_NAMES_MAP[correction_type.value]
        if direction is not None:
            prm_list = {direction: prm_list[direction]}
        correction = self._core_merge_correction(prm_list, selection=selection,
                                                 set_auto_mask=set_auto_mask,
                                                 meter=meter)
        correction['name'] = correction_type.value
        return correction


class Sentinel1EtadSwath:
    """Object representing a swath in the S1-ETAD product.

    This objects are returned by methods of the :class:`Sentine1Etad` class.
    It is not expected that the user instantiates this objects directly.
    """

    def __init__(self, nc_group):
        self._grp = nc_group

    @functools.lru_cache()
    def __getitem__(self, burst_index):
        burst_name = f"Burst{burst_index:04d}"
        return Sentinel1EtadBurst(self._grp[burst_name])

    def __iter__(self):
        yield from self.iter_bursts()

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._grp.path}")  0x{id(self):x}'

    @property
    def burst_list(self):
        """The list of burst identifiers (str) of all bursts in the swath."""
        return [burst.bIndex for burst in self._grp.groups.values()]

    @property
    def number_of_burst(self):
        """The number of bursts in the swath."""
        return len(self._grp.groups)

    @property
    def swath_id(self):
        """The swath identifier (str)."""
        return self._grp.swathID

    @property
    def swath_index(self):
        """The swath index (int)."""
        return self._grp.sIndex

    @property
    def sampling_start(self):
        """Relative sampling start times."""
        first_burst_index = self.burst_list[0]
        first_burst = self[first_burst_index]
        return first_burst.sampling_start

    @property
    def sampling(self):
        """Sampling in seconds used for all bursts of the swath.

        A dictionary containing the following keys:

        * "x": range spacing,
        * "y": azimuth spacing,
        * "units": the measurement units used for "x' and "y"
        """
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

    def get_footprint(self, selection=None, merge=False):
        """Return the footprints of all the bursts as MultiPolygon.

        It calls in the back the get_footprint of the Sentinel1EtadBurst class.

        Parameters
        ----------
        selection : list(int) or pd.Dataframe, optional
            the list of selected bursts or result of a
            Sentinel1Etad.query_burst query.
            If the selection is None (default) the iteration is performed
            on all the burst of the swath.
        merge : bool
            if set to True return a single polygon that is the union of the
            footprints of all bursts
        """
        polys = [burst.get_footprint() for burst in self.iter_bursts(selection)]
        if merge:
            polys = shapely.ops.cascaded_union(polys)
        else:
            polys = MultiPolygon(polys)

        return polys

    def intersects(self, geometry: BaseGeometry):
        """Return the list of burst indexes intersecting the input geometry.

        Computes the intersection of the footprint of the swath (all bursts)
        with the input Geometry

        Parameters
        ----------
        geometry : shapely.geometry.[Point, Polygon, MultiPolygon, line]

        Returns
        -------
        list
            list of the indexes of all bursts intersecting with the input
            geometry
        """
        assert isinstance(geometry, BaseGeometry), \
            'The input shape is not a shapely BaseGeometry object'
        burst_index_list = []
        swath_footprint = self.get_footprint(merge=True)
        if swath_footprint.intersects(geometry):
            burst_index_list = [
                b.burst_index for b in self.iter_bursts()
                if b.intersects(geometry)
            ]
        return burst_index_list

    def _burst_merger(self, burst_var, selection=None,
                      az_time_min=None, az_time_max=None,
                      set_auto_mask=False, meter=False, fill_value=0.):
        """Low level method to de-burst a NetCDF variable.

        The de-burst strategy is simple as the latest line is on top of the
        oldest.

        Parameters
        ----------
        burst_var : str
            one of the burst netcdf variables
        selection : list or pandas.DataFrame
            list of selected bursts (by default all bursts are selected)
        az_time_min : float
            minimum azimuth time of the merged swath
             (relative to the reference annotated in the NetCDF root)
        az_time_max : float
            maximum azimuth tim eof the merged swath
            (relative to the reference annotated in the NetCDF root)
        set_auto_mask : bool
            requested for netCDF4 to avoid retrieving a masked array
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

        tau0 = min(
            burst.sampling_start['x']
            for burst in self.iter_bursts(burst_index_list)
        )

        # grid sampling
        dt = first_burst.sampling['y']
        dtau = first_burst.sampling['x']

        num_lines = np.round((t1 - t0) / dt).astype(int) + 1
        num_samples = max(
            burst.samples for burst in self.iter_bursts(burst_index_list)
        )

        debursted_var = np.full((num_lines, num_samples),
                                fill_value=fill_value)
        # TODO: add some control option
        debursted_var = np.ma.array(debursted_var,
                                    mask=True, fill_value=fill_value)

        for burst_ in self.iter_bursts(burst_index_list):
            assert(dt == burst_.sampling['y']), \
                'The azimuth sampling is changing long azimuth'
            assert(first_burst.sampling_start['x'] ==
                   burst_.sampling_start['x']), \
                'The 2-way range gridStartRangeTime is changing long azimuth'

            # get the timing of the burst and convert into line index
            az_time_, rg_time_ = burst_.get_burst_grid()
            line_index_ = np.round((az_time_ - t0) / dt).astype(int)
            p0 = np.round((rg_time_[0] - tau0) / dtau).astype(int)

            # NOTE: use the private "Sentinel1EtadBurst._get_etad_param" method
            # to be able to work only on the specified NetCDF variable
            var_ = burst_._get_etad_param(burst_var,  # noqa
                                          set_auto_mask=set_auto_mask,
                                          meter=meter)

            _, burst_samples = var_.shape
            debursted_var[line_index_, p0:p0+burst_samples] = var_

        return {
            burst_var: debursted_var,
            'first_azimuth_time': t0,
            'first_slant_range_time': first_burst.sampling_start['x'],
            'sampling': first_burst.sampling,
        }

    def _core_merge_correction(self, prm_list, selection=None,
                               set_auto_mask=True, meter=False):
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self._burst_merger(field, selection=selection,
                                     set_auto_mask=set_auto_mask, meter=meter)
            dd[dim] = dd_[field]
            dd['sampling'] = dd_['sampling']
            dd['first_azimuth_time'] = dd_['first_azimuth_time']
            dd['first_slant_range_time'] = dd_['first_slant_range_time']

        dd['unit'] = 'm' if meter else 's'
        dd['lats'] = self._burst_merger('lats', set_auto_mask=set_auto_mask,
                                        meter=False)['lats']
        dd['lons'] = self._burst_merger('lons', set_auto_mask=set_auto_mask,
                                        meter=False)['lons']
        dd['height'] = self._burst_merger('height',
                                          set_auto_mask=set_auto_mask,
                                          meter=False)['height']
        return dd

    def merge_correction(self, name: CorrectionType = ECorrectionType.SUM,
                         selection=None, set_auto_mask=True, meter=False,
                         direction=None):
        """Merge multiple bursts of the specified correction variable.

        Data of the selected bursts (typically overlapped) are merged
        together to form a single data matrix with a consistent (azimuth)
        time axis.

        Note
        ----

        The current implementation uses a very simple algorithm that
        iterates over selected bursts and stitches correction data
        together.

        In overlapping regions, new data simpy overwrite the old ones.
        This is an easy algorithm and perfectly correct for atmospheric
        and geodetic correction.

        It is, instead, sub-optimal for system corrections (bi-static,
        Doppler, FM Rate) which have different values in overlapping
        regions. In this case results are *not* correct.

        Parameters
        ----------
        name : str or CorrectionType
            the name of the desired correction
        selection : list or pandas.DataFrame
            list of selected bursts (by default all bursts are selected)
        set_auto_mask : bool
            requested for netCDF4 to avoid retrieving a masked array
        meter : bool
            transform the result in meters
        direction : str or None
            if set to "x" (for range) or "y" (for "azimuth") only extracts
            the specified correction component.
            By default (None) all available components are returned.

        Returns
        -------
        dict
            a dictionary containing merged data and sampling information:

            :<burst_var_name>:
                merged data for the selected burst_var
            :first_azimuth_time:
                the relative azimuth first time
            :first_slant_range_time:
                the relative (slant) range first time
            :sampling:
                a dictionary containing the sampling along the
                'x' and 'y' directions and the 'unit'
            :units:
                of the correction (seconds or meters)
            :lats:
                the matrix of latitude values (in degrees) for each point
            :lons:
                the matrix of longitude values (in degrees) for each point
            :height:
                the matrix of height values (in meters) for each point
        """
        correction_type = ECorrectionType(name)  # check values
        prm_list = _CORRECTION_NAMES_MAP[correction_type.value]
        if direction is not None:
            prm_list = {direction: prm_list[direction]}
        correction = self._core_merge_correction(prm_list, selection=selection,
                                                 set_auto_mask=set_auto_mask,
                                                 meter=meter)
        correction['name'] = correction_type.value
        return correction


class Sentinel1EtadBurst:
    """Object representing a burst in the S1-ETAD product.

    This objects are returned by methods of the :class:`Sentinel1EtadSwath`
    class.
    It is not expected that the user instantiates this objects directly.
    """

    def __init__(self, nc_group):
        self._grp = nc_group
        self._geocoder = None

    def __repr__(self):
        return f'{self.__class__.__name__}("{self._grp.path}")  0x{id(self):x}'

    @property
    def product_id(self):
        """The S1 product (str) to which the burst object is associated."""
        return self._grp.productID

    @property
    def swath_id(self):
        """The swath identifier (str) to which the burst belongs."""
        return self._grp.swathID

    @property
    def burst_id(self):
        """The burst identifier (str)."""
        return self._grp.name

    @property
    def product_index(self):
        """Index (int) of the S1 product to which the burst is associated."""
        return self._grp.pIndex

    @property
    def swath_index(self):
        """The index (int) of the swath to which the burst belongs."""
        return self._grp.sIndex

    @property
    def burst_index(self):
        """The index (int) of the burst."""
        return self._grp.bIndex

    @functools.lru_cache()
    def get_footprint(self):
        """Return the footprint of ghe bursts as shapely.Polygon.

        It gets the lat/lon/height grid and extract the 4 corners.
        """
        lats, lons, heights = self.get_lat_lon_height()
        corner_list = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
        etaf_burst_footprint = []
        for corner in corner_list:
            lat_, lon_, h_ = lats[corner], lons[corner], heights[corner]
            etaf_burst_footprint.append((lon_, lat_, h_))
        etaf_burst_footprint = Polygon(etaf_burst_footprint)
        return etaf_burst_footprint

    def intersects(self, geometry: BaseGeometry):
        """Intersects the footprint of the burst with the provided shape

        Parameters
        ----------
        geometry : shapely.geometry.[Point, Polygon, MultiPolygon, line]

        Returns
        -------
        bool
            True if intersects, False otherwise
        """
        assert isinstance(geometry, BaseGeometry), \
            'Not a shapely BaseGeometry object'
        return self.get_footprint().intersects(geometry)

    def get_burst_grid(self):
        """Return the t, tau grid of the burst."""
        azimuth = self._get_etad_param('azimuth', set_auto_mask=True)
        range_ = self._get_etad_param('range', set_auto_mask=True)
        return azimuth, range_

    @property
    def sampling_start(self):
        """Relative sampling start times.

        Value in seconds relative to the beginning of the product.
        """
        # TODO: put a reference in the docstring to the proper
        #       Sentinel1Etad property.
        return dict(
            x=self._grp.gridStartRangeTime,
            y=self._grp.gridStartAzimuthTime,
            units='s',
        )

    @property
    def sampling(self):
        """Sampling in seconds used for all bursts of the swath.

        A dictionary containing the following keys:

        * "x": range spacing,
        * "y": azimuth spacing,
        * "units": the measurement units used for "x' and "y"
        """
        return dict(
            x=self._grp.gridSamplingRange,
            y=self._grp.gridSamplingAzimuth,
            units='s',
        )

    @property
    def lines(self):
        """The number of lines in  the burst."""
        return self._grp.dimensions['azimuthExtent'].size

    @property
    def samples(self):
        """The number of samples in the burst."""
        return self._grp.dimensions['rangeExtent'].size

    @property
    def vg(self) -> float:
        """Average zero-Doppler ground velocity [m/s]."""
        return self._grp.averageZeroDopplerVelocity

    @property
    def reference_polarization(self) -> str:
        """Reverence polarization (string)."""
        return self._grp.referencePolarization

    def get_polarimetric_channel_offset(self, channel: str) -> dict:
        """Polarimetric channel delay.

        Return the electronic delay of the specified polarimetric channel
        w.r.t. the reference one (see
        :data:`Sentinel1EtadBurst.reference_polarization`).

        channel : str
            the string ID of the requested polarimetric channel:
            * 'VV' or 'VH' for DV products
            * 'HH' or 'HV' for DH products
        """
        if channel not in {'HH', 'HV', 'VV', 'VH'}:
            raise ValueError(f'invalid channel ID: {channel!r}')

        if channel[0] != self._grp.referencePolarization[0]:
            raise ValueError(
                f'polarimetric channel not available: {channel!r}')

        data = dict(units='s')

        if channel == 'HH':
            data['x'] = self._grp.rangeOffsetHH,
            data['y'] = self._grp.rangeOffsetHH,
        elif channel == 'HV':
            data['x'] = self._grp.rangeOffsetHV,
            data['y'] = self._grp.rangeOffsetHV,
        elif channel == 'VH':
            data['x'] = self._grp.rangeOffsetVH,
            data['y'] = self._grp.rangeOffsetVH,
        elif channel == 'VV':
            data['x'] = self._grp.rangeOffsetVV,
            data['y'] = self._grp.rangeOffsetVV,

        return data

    def get_timing_calibration_constants(self) -> dict:
        try:
            return dict(
                x=self._grp.instrumentTimingCalibrationRange,
                y=self._grp.instrumentTimingCalibrationAzimuth,
                units='s',
            )
        except AttributeError:
            # @COMPATIBILITY: with SETAP , v1.6
            warnings.warn(
                'instrument timing calibration constants are not available '
                'in the NetCDF data component this product. '
                'Calibration constants have been added to the NetCDF '
                'component in SETAP v1.6 (ETAD-DLR-PS-0014 - '
                '"ETAD Product Format Specification" Issue 1.5).'
            )
            return dict(x=0, y=0, units='s')

    def _get_etad_param(self, name, set_auto_mask=False, transpose=False,
                        meter=False):
        assert name in self._grp.variables, f'Parameter {name!r} is not allowed'

        self._grp.set_auto_mask(set_auto_mask)

        # TODO: avoid double copies
        # TODO: decimation factor
        field = np.asarray(self._grp[name])
        if transpose:
            field = np.transpose(field)

        if meter:
            if name.endswith('Az'):
                k = self._grp.averageZeroDopplerVelocity
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

    def get_lat_lon_height(self, transpose=False):
        """Return the latitude, longitude and height for each point.

        Data are returned as (3) matrices (lines x samples).
        Latitude and longitude are expressed in degrees, height is
        expressed in meters.
        """
        lats = self._get_etad_param(
            'lats', transpose=transpose, meter=False, set_auto_mask=True)
        lons = self._get_etad_param(
            'lons', transpose=transpose, meter=False, set_auto_mask=True)
        h = self._get_etad_param(
            'height', transpose=transpose, meter=False, set_auto_mask=True)
        return lats, lons, h

    def _core_get_correction(self, prm_list, set_auto_mask=False,
                             transpose=False, meter=False):
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self._get_etad_param(
                field, set_auto_mask=set_auto_mask, transpose=transpose,
                meter=meter)

        correction['unit'] = 'm' if meter else 's'

        return correction

    def get_correction(self, name: CorrectionType = ECorrectionType.SUM,
                       set_auto_mask=False, transpose=False, meter=False,
                       direction=None):
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
            numpy convention for dimensions (default: False)
        meter : bool
            transform the result in meters
        direction : str or None
            if set to "x" (for range) or "y" (for "azimuth") only extracts
            the specified correction component.
            By default (None) all available components are returned.

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
        if direction is not None:
            prm_list = {direction: prm_list[direction]}
        correction = self._core_get_correction(prm_list,
                                               set_auto_mask=set_auto_mask,
                                               transpose=transpose, meter=meter)
        correction['name'] = name
        return correction

    def _get_geocoder(self):
        if self._geocoder is None:
            from .geometry import GridGeocoding
            azimuth, range_ = self.get_burst_grid()
            lats, lons, heights = self.get_lat_lon_height()
            self._geocoder = GridGeocoding(lats, lons, heights,
                                           xaxis=range_, yaxis=azimuth)
        return self._geocoder

    def radar_to_geodetic(self, tau, t, deg=True):
        """Convert RADAR coordinates into geodetic coordinates.

        Compute the geodetic coordinates (lat, lon, h) corresponding to
        RADAR coordinates (tau, t), i.e. fast time (range time) and slow
        time (azimuth time expressed in seconds form the reference
        :data:`Sentinel1Etad.min_azimuth_time`)::

            (tau, t) -> (lat, lon, h)

        If ``deg`` is True the output ``lat`` and ``lon`` are expressed
        in degrees, otherwise in radians.

        The implementation is approximated and exploits pre-computed grids
        of latitude, longitude and height values.

        The method is not as accurate as solving the range-Doppler equations.

        .. seealso:: :class:`s1etad.geometry.GridGeocoding`.
        """
        return self._get_geocoder().forward_geocode(tau, t, deg=deg)

    def geodetic_to_radar(self, lat, lon, h=0, deg=True):
        """Convert geodetic coordinates into RADAR coordinates.

        Compute the RADAR coordinates (tau, t), i.e. fast time (range time)
        and slow time (azimuth time expressed in seconds form the reference
        :data:`Sentinel1Etad.min_azimuth_time`) corresponding to
        geodetic coordinates (lat, lon, h)::

            (lat, lon, h) -> (tau, t)

        If ``deg`` is True it is assumed that input ``lat`` and ``lon``
        are expressed in degrees, otherwise it is assumed that angles
        are expressed in radians.

        The implementation is approximated and exploits pre-computed grids
        of latitude, longitude and height values.

        The method is not as accurate as solving the range-Doppler equations.

        .. seealso:: :class:`s1etad.geometry.GridGeocoding`.
        """
        return self._get_geocoder().backward_geocode(lat, lon, h, deg=deg)

    def radar_to_image(self, t, tau):
        """Convert RADAR coordinates into image coordinates.

        Compute the image coordinates (line, sample) corresponding
        to RADAR coordinates (tau, t), i.e. fast time (range time) and
        slow time (azimuth time expressed in seconds form the reference
        :data:`Sentinel1Etad.min_azimuth_time`)::

            (tau, t) -> (line, sample)
        """
        line = (t - self.sampling_start['y']) / self.sampling['y']
        sample = (tau - self.sampling_start['x']) / self.sampling['x']
        return line, sample

    def image_to_radar(self, line, sample):
        """Convert image coordinates into RADAR coordinates.

        Compute the RADAR coordinates (tau, t), i.e. fast time (range time)
        and slow time (azimuth time expressed in seconds form the reference
        :data:`Sentinel1Etad.min_azimuth_time`) corresponding to
        image coordinates (line, sample)::

            (line, sample) -> (t, tau)
        """
        t = self.sampling_start['y'] + line * self.sampling['y']
        tau = self.sampling_start['x'] + sample * self.sampling['x']
        return t, tau
