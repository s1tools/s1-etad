import pathlib
import numpy as np

from scipy import constants
from lxml import etree
from netCDF4 import Dataset


import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil import parser


from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point
from shapely.ops import cascaded_union
import simplekml


class Sentinel1Etad:
    """
        Class to decode and access the elements of the Sentinel ETAD product
        which specification is governed by ETAD-DLR-PS-0014

        The index operator [] implemented with the __getitem__ method returns a
        Sentinel1EtadSwath class

    """
    def __init__(self, etadProduct):
        self.product = pathlib.Path(etadProduct)
        self.ds = self.__measurement_dataset
        self._annot = self.__annotation_dataset
        self.burst_catalogue = self._burst_catalogue()

    @property
    def __measurement_dataset(self):
        """ open the nc dataset  """
        list_ = [ i for i in self.product.glob("measurement/*.nc")]
        rootgrp = Dataset(list_[0], "r", set_auto_mask=False)
        return(rootgrp)

    @property
    def __annotation_dataset(self):
        """ open the xml annotation dataset  """
        list_ = [i for i in self.product.glob("annotation/*.xml")]
        xml_file = str(list_[0])
        root = etree.parse(xml_file).getroot()
        return(root)

    def __getitem__(self, index):
        assert index in self.swath_list, f"{index} is not in { self.swath_list}"

        return Sentinel1EtadSwath(self.ds[index])

    @property
    def number_of_swath(self):
        nn = len(self.ds.groups)
        return(nn)

    @property
    def swath_list(self):
        return self.ds.groups.keys()

    def s1_product_list(self):
        """
        Read the list of S-1 products that have been used to compose the ETAD one

        """
        xp = 'productComponents/inputProductList/inputProduct/productID'
        dd = self._xpath_to_list(self._annot, xp)
        return dd


    @property
    def grid_spacing(self):
        """ Return the grid spacing in meters """

        xp_list = { 'x':'.//correctionGridRangeSampling', 'y':'.//correctionGridAzimuthSampling'}
        dd = {}
        for tag, xp in xp_list.items():
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=np.float)
        dd['unit'] = 'm'
        return dd

    @property
    def grid_sampling(self):
        """ Return the grid spacing in s """

        xp_list = { 'x':'.//productInformation/gridSampling/range', 'y':'.//productInformation/gridSampling/azimuth'}
        dd = {}
        for tag, xp in xp_list.items():
            dd[tag] = self._xpath_to_list(self._annot, xp, dtype=np.float)
        dd['unit'] = 's'
        return dd

    def processing_setting(self):
        """Read the xml file to  identify the corrections performed
        If a correction is not performed the matrix is filled with zeros

        """

        correction_list = ['troposphericDelayCorrection', 'ionosphericDelayCorrection', \
                            'solidEarthTideCorrection', 'bistaticAzimuthCorrection', \
                            'dopplerShiftRangeCorrection', 'FMMismatchAzimuthCorrection']
        dd = {}
        xp_root = 'processingInformation/processor/setapConfigurationFile/processorSettings/'
        for correction in correction_list:
            xp = xp_root + correction
            ret = self._xpath_to_list(self._annot, xp)
            if ret == 'true':
                ret = True
            else: ret = False
            dd[correction] = ret
        return dd

    def _burst_catalogue(self):
        """ Parses the XML annotation dataset to create a panda DataFrame conntaining
            all the elements allowing to index properly a burst
        """
        df = None
        for burst_ in self._annot.findall('.//etadBurst'):
            burst_dict = dict(burst_.find('burstData').attrib)
            burst_dict['productID']  =  burst_.find('burstData/productID').text
            burst_dict['swathID'] = burst_.find('burstData/swathID').text
            burst_dict['azimuthTimeMin'] = self._xpath_to_list(burst_,'burstCoverage/temporalCoverage/azimuthTimeMin', parse_time_func=lambda x: x)
            burst_dict['azimuthTimeMax'] = self._xpath_to_list(burst_,'burstCoverage/temporalCoverage/azimuthTimeMax', parse_time_func=lambda x: x)

            if df is None:
                df = pd.DataFrame(burst_dict, index=[0])
            else:
                df = df.append(burst_dict, ignore_index=True)
        return df


    def query_burst(self, first_time=None, product_name=None, last_time=None, swath=None):
        """Implements a query to the burst catalogue to retrieve the burst matching
        the query by time

        Parameters:
        ------------
            first_time : datetime
                is set to None then set to the first time

            last_time : datetime
                if set to None the last_time = first_time

            product_name : str
                Name of a real S1 product e.g. S1B_IW_SLC__1SDV_20190805T162509_20190805T162...SAFE

            swath : list
                list of swathID e.g. 'IW1' or ['IW1'] or ['IW1', 'IW2']

        Returns:
        --------
            Filtered panda dataframe
        """
        #first sort the burst by time
        df = self.burst_catalogue.sort_values(by=['azimuthTimeMin'])
        if first_time is None: first_time = df.iloc[0].azimuthTimeMin
        if last_time is None: last_time = df.iloc[-1].azimuthTimeMax

        ix0 = (df.azimuthTimeMin >= first_time) & (df.azimuthTimeMax <= last_time)

        if product_name is not None:
            #build a reges based on the name to avoid issues with annotation products and CRC
            product_name  = Sentinel1ProductName(product_name)
            product_name.to_annotation(value='[AD]')
            product_name.crc=''
            filter = product_name.recompose(with_suffix=False)
            ix0 = ix0 & (self.burst_catalogue.productID.str.contains(filter,regex=True))


        if swath is not None:
            if not isinstance(swath, list):
                #hugly methhod to transfrom string into a list_
                swath = swath.split(' ')
            ix0 = ix0 & (df.swathID.isin(swath) )

        return df.loc[ix0]


    def xpath_to_list(self, xpath, dtype=None, namespace={}, parse_time_func=None):
        return self._xpath_to_list(self._annot, xpath, dtype=dtype, namespace=namespace, parse_time_func=parse_time_func)

    @staticmethod
    def _xpath_to_list(root, xpath, dtype=None, namespace={}, parse_time_func=None):

        ll = [elt.text for elt in root.findall(xpath,namespace) ]
        if parse_time_func is not None:
            ll = [parser.parse(t) for t in ll ]
            ll = parse_time_func(ll)
        ll = np.asarray(ll, dtype=dtype)

        if ll.size == 1:
            return ll.item(0)
        else :
            return ll


    def get_footprint(self, swath_list=None, merge=False):
        """method to get the footprints of all the bursts as MultiPolygon
            It calls in the back the get_footprint of the Sentinel1EtadBurst class
        """
        if swath_list is None:
            swath_list = self.swath_list
            polys=[]
            for swath_ in swath_list:
                for poly_ in self.__getitem__(swath_).get_footprint() :
                    polys.append(poly_)

        return polys


    def __swath_merger(self, burst_variable, swath_list=None, burst_index_list=None):
        if swath_list is None:
            swath_list = self.swath_list

        sampling = self.grid_sampling

        first_slant_range_time = self.ds.rangeTimeMin
        last_slant_range_time = self.ds.rangeTimeMax
        first_azimuth_time = parser.parse(self.ds.azimuthTimeMin)
        last_azimuth_time = parser.parse(self.ds.azimuthTimeMax)

        num_samples = np.round((last_slant_range_time-first_slant_range_time) / sampling['x']).astype(np.int)+1
        num_lines =  np.round((last_azimuth_time - first_azimuth_time).total_seconds() / sampling['y']).astype(np.int)+1

        img = np.zeros((num_lines, num_samples))

        for swath_ in  swath_list:
            dd_ = self[swath_].merge_sum_correction(burst_index_list=burst_index_list)
            line_ofs = np.round(dd_['first_azimuth_time'] / sampling['y']).astype(np.int)
            sample_ofs = np.round((dd_['first_slant_range_time']) /  sampling['x']).astype(np.int)

            slice_y = slice(line_ofs, line_ofs + dd_['x'].shape[0])
            slice_x = slice(sample_ofs, sample_ofs + dd_['x'].shape[1])

            img[slice_y, slice_x ] = dd_['x']
        return(img)

        # np.round((last_slant_range_time-first_slant_range_time) / sampling['x']).astype(np.int)+1

    _swath_merger = __swath_merger

    def to_kml(self, kml_file):
        kml = simplekml.Kml()

        #get the footprints
        burst_ftps = self.get_footprint()
        kml_dir = kml.newfolder(name=f"Sentinel1 Timing Correction Grid ")
        for ix, ftp in enumerate(burst_ftps):
            x, y = ftp.exterior.xy
            corner= [ (x[i], y[i]) for i in range(len(x)) ]

            pol = kml_dir.newpolygon(name=str(ix))
            pol.outerboundaryis = corner
            pol.altitudeMode = 'absolute'
            pol.tessellate=1
            pol.polystyle.fill = 0
            pol.style.linestyle.width = 2

        kml.save(kml_file)



class Sentinel1EtadSwath():
    def __init__(self, nc_group):
        self._grp = nc_group

    def __getitem__(self, burst_index):
        burst_index = str(burst_index).rjust(4,'0')
        burst_name = f"Burst{burst_index}"
        return Sentinel1EtadBurst(self._grp[burst_name])

    @property
    def burst_list(self):
        burst_list = [int(burst_str[5:]) for burst_str in list(self._grp.groups.keys())]
        return burst_list


    @property
    def number_of_burst(self):
        return len(self._grp.groups)

    def get_footprint(self, burst_index_list=None):
        """method to get the footprints of all the bursts as MultiPolygon
            It calls in the back the get_footprint of the Sentinel1EtadBurst class
        """
        if burst_index_list is None:
            burst_index_list = self.burst_list

        footprints = [self.__getitem__(bix).get_footprint() for bix in burst_index_list]
        return MultiPolygon(footprints)


    def merge_sum_correction(self, burst_index_list=None, set_auto_mask=True, transpose=True, meter=False):

        prm_list = {'x' : 'sumOfCorrectionsRg', 'y':'sumOfCorrectionsAz'}
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self.__burst_merger(field, burst_index_list=burst_index_list, \
                                                                set_auto_mask=set_auto_mask, \
                                                                transpose=transpose, \
                                                                meter=meter)
            dd[dim] = dd_[field]

        unit = 's'
        if meter: unit = 'm'
        dd['unit'] = unit
        dd['lats'] = self.__burst_merger('lats', transpose=transpose, meter=False, set_auto_mask=set_auto_mask)
        dd['lons'] = self.__burst_merger('lons', transpose=transpose, meter=False, set_auto_mask=set_auto_mask)
        dd['sampling'] =  dd_['sampling']
        dd['first_azimuth_time'] =  dd_['first_azimuth_time']
        dd['first_slant_range_time'] = dd_['first_slant_range_time']
        #heights = self.__burst_merger__burst_merger('height', transpose=transpose, meter=False, set_auto_mask=set_auto_mask)
        return dd


    def merge_troposhere_correction(self, burst_index_list=None, set_auto_mask=True, transpose=True, meter=False):

        prm_list = {'x':'troposphericCorrectionRg'}
        dd = {}
        for dim, field in prm_list.items():
            dd_ = self.__burst_merger(field, burst_index_list=burst_index_list, \
                                                                set_auto_mask=set_auto_mask, \
                                                                transpose=transpose, \
                                                                meter=meter)
            dd[dim] = dd_[field]

        unit = 's'
        if meter: unit = 'm'
        dd['unit'] = unit
        dd['lats'] = self.__burst_merger('lats', transpose=transpose, meter=False, set_auto_mask=set_auto_mask)
        dd['lons'] = self.__burst_merger('lons', transpose=transpose, meter=False, set_auto_mask=set_auto_mask)
        dd['sampling'] =  dd_['sampling']
        dd['first_azimuth_time'] =  dd_['first_azimuth_time']
        dd['first_slant_range_time'] = dd_['first_slant_range_time']


    def __burst_merger(self, burst_var, burst_index_list=None, \
                            azimuthTimeMin=None, azimuthTimeMax=None, \
                            set_auto_mask=False, transpose=True, meter=False):
        """Template method to deburst a variables
            The deburst strategu is  soimple as the latest line is on top of the oldest

            Parameter:
                burst_var : one of the burst netcdf variables
        """
        if burst_index_list is None:
            burst_index_list = self.burst_list

        #Find what is the extent of the acquistion in azimuth
        first_burst = self.__getitem__(burst_index_list[0])
        last_burst = self.__getitem__(burst_index_list[-1])

        if azimuthTimeMin == None:
            t0 = first_burst._grp['azimuth'][0]
        else: t0 = azimuthTimeMin

        if azimuthTimeMax == None:
            t1 = last_burst._grp['azimuth'][-1]
        else:
            t1 = azimuthTimeMax

        #azimuth grid sampling
        dt = first_burst.sampling['y']

        num_lines = np.round((t1-t0) / dt).astype(np.int)+1
        num_samples = first_burst._grp.dimensions['rangeExtent'].size

        debursted_var = np.zeros((num_lines, num_samples))

        for b, burst_index in enumerate(burst_index_list):
            burst_ = self.__getitem__(burst_index)
            assert(dt == burst_.sampling['y']), 'The azimuth sampling is changing long azimuth'
            assert(first_burst._grp.gridStartRangeTime0 == burst_._grp.gridStartRangeTime0), 'The 2-way range gridStartRangeTime0 is changing long azimuth'

            #get the timing of the burst and convert into line index
            az_time_, rg_time_ = burst_.get_burst_grid()
            line_index_ =  np.round( (az_time_ - t0) / dt).astype(np.int)

            var_ = burst_._get_etad_param(burst_var, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

            debursted_var [line_index_,:] = var_

        dd={burst_var : debursted_var, \
            'first_azimuth_time': t0, \
            'first_slant_range_time' : first_burst._grp.gridStartRangeTime0, \
            'sampling' : first_burst.sampling}

        return  dd


class Sentinel1EtadBurst():
    def __init__(self, nc_group):
        self._grp =  nc_group
    def get_footprint(self):
        """method to get the footprint of ghe bursts as shapely.Polygon. It gets the
        lat / lon / height grid and extract the 4 corners
        """

        lats, lons = self.__get_etad_param('lats', set_auto_mask=True)
        lons = self.__get_etad_param('lons', set_auto_mask=True)
        heights = self.__get_etad_param('height', set_auto_mask=True)


        corner_list = [ (0, 0), (0, -1), (-1, -1), (-1, 0)]
        etaf_burst_footprint = []
        for corner in corner_list:
            lat_, lon_, h_ = lats[corner], lons[corner], heights[corner]
            etaf_burst_footprint.append((lon_, lat_, h_))
        etaf_burst_footprint = Polygon(etaf_burst_footprint)
        return etaf_burst_footprint



    def get_burst_grid(self, burst_index_list=None):
        """method to get the t, tau grid of  the burst as

        """

        azimuth = self.__get_etad_param('azimuth', set_auto_mask=True)
        range = self.__get_etad_param('range', set_auto_mask=True)
        return azimuth, range

    @property
    def sampling(self):
        dd={ 'x':self._grp.gridSamplingRange, 'y':self._grp.gridSamplingAzimuth}
        dd['unit'] = 's'
        return dd

    def __get_etad_param(self, correction, set_auto_mask=False, transpose=True, meter=False):

        correction_list = list(self._grp.variables.keys())
        assert(correction in correction_list), 'Parameter is not allowed list'

        self._grp.set_auto_mask(set_auto_mask)

        field = np.asarray(self._grp[correction])
        if transpose:
            field = np.transpose(field)

        if meter:
            field *= constants.c/2

        return field

    _get_etad_param = __get_etad_param

    def get_lat_lon_heigth(self, transpose=True):
        lats = self.__get_etad_param('lats', transpose=transpose, meter=False,set_auto_mask=True)
        lons = self.__get_etad_param('lons', transpose=transpose, meter=False,set_auto_mask=True)
        h = self.__get_etad_param('height', transpose=transpose, meter=False,set_auto_mask=True)
        return  lats, lons, h

    def get_tropospheric_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        correction = {}
        prm_list = {'x':'troposphericCorrectionRg'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

        unit = 's'
        if meter: unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'tropospheric'


        return correction


    def get_ionospheric_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        correction = {}
        prm_list = {'x':'ionosphericCorrectionRg'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

        unit = 's'
        if meter: unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'ionospheric'

        return correction

    def get_geodetic_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        prm_list = {'x':'geodeticCorrectionRg', 'y':'geodeticCorrectionAz'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

        unit = 's'
        if meter:
            unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'geodetic'
        return correction

    def get_bistatic_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
            '''
        prm_list = {'y':'bistaticCorrectionAz'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

        unit = 's'
        if meter:
            unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'bistatic'
        return correction

    def get_doppler_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        prm_list = {'x':'dopplerRangeShiftRg'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)
        unit = 's'
        if meter:
            unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'Doppler'
        return correction

    def get_fmrate_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        prm_list = {'y':'fmMismatchCorrectionAz'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)

        unit = 's'
        if meter:
            unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'FM rate'
        return correction

    def get_sum_correction(self, set_auto_mask=False, transpose=True, meter=False):
        ''' Retrieve the requested correction in range (and azimuth if applicable).
            Puts the results in a dict.

            Parameters:
                set_auto_mask : bool
                    requested for netCDF4 to avoid retrieving a masked array
                transpose : bool
                    requested to retrieve the correction in array following the numpy convention for dimensions
                meter : bool
                    transform the result in meters

            Returns:
                correction : dict
                    x : correction in range
                    y : correction in azimuth (if applicable)
                    unit : 'm' or 's'
                    name : name of the correction
        '''
        prm_list = {'x' : 'sumOfCorrectionsRg', 'y':'sumOfCorrectionsAz'}
        correction = {}
        for dim, field in prm_list.items():
            correction[dim] = self.__get_etad_param(field, set_auto_mask=set_auto_mask, transpose=transpose, meter=meter)
        unit = 's'
        if meter:
            unit = 'm'
        correction['unit'] = unit
        correction['name'] = 'Sum'
        return correction



class Sentinel1ProductName():
    """Class to manipulate the filename of Sentinel 1 products"""
    def __init__(self, product_name):

        self.__product_name = pathlib.Path(product_name)
        self.suffix = self.__product_name.suffix
        self.file_name = self.__product_name.stem
        self._parts = self.file_name.split('_')

        #trick for SLC
        if len(self._parts) == 10 and 'SLC' in self.ptype :
            del self._parts[3]
            self.ptype = 'SLC_'


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
        print (self._parts[1])

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
        else: return False

    def to_annotation(self, value='A'):
        ll = list(self.typepol)
        ll[1]=value
        self.typepol = ''.join(ll)

    def to_standard(self):
        ll = list(self.typepol)
        ll[1]='S'
        self.typepol = ''.join(ll)

    def recompose(self, with_suffix=True):
        pp = '_'.join(self._parts)
        if with_suffix:
            pp += self.suffix
        return(pp)
