# -*- coding: utf-8 -*-

import pathlib


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
        path = '_'.join(self._parts)
        if with_suffix:
            path += self.suffix
        return path
