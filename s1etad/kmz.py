"""Support for Google KMZ preview generation."""

import shutil
import pathlib
import functools

import numpy as np

from datetime import timedelta
from dateutil import parser

from simplekml import Kml, OverlayXY, ScreenXY, Units, RotationXY

from osgeo import gdal
from osgeo import osr

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot

from .product import Sentinel1Etad


__all__ = ['etad_to_kmz', 'Sentinel1EtadKmlWriter']


class Sentinel1EtadKmlWriter:
    def __init__(self, etad, corrections=None, decimation_factor=1):
        self.etad = etad
        self.etad_file = self.etad.product

        # TODO: should depend on the output filename
        self._kmzdir = pathlib.Path(self.etad_file.stem)
        if self._kmzdir.exists():
            raise FileExistsError(
                f'output path already exists "{self._kmzdir}"')

        self.kml = Kml()
        self.kml_root = self.kml.newfolder(name=self.etad_file.stem)
        self._set_timespan()

        self.write_overall_footprint()
        if corrections is None:
            corrections = ['sumOfCorrections', 'troposphericCorrection']
        self.write_corrections(corrections, decimation_factor=decimation_factor,
                               colorizing=True)
        self.write_burst_footprint()

    def _set_timespan(self, duration=30, range_=1500000):
        swath = self.etad[list(self.etad.swath_list)[0]]
        burst = swath[list(swath.burst_list)[0]]
        lon0, lat0, lon1, lat1 = burst.get_footprint().bounds
        lat = np.mean([lat0, lat1])
        lon = np.mean([lon0, lon1])
        t0 = parser.parse(self.etad.ds.azimuthTimeMin)
        t1 = t0 + timedelta(seconds=duration)  # configure duration

        self.kml_root.lookat.latitude = lat
        self.kml_root.lookat.longitude = lon
        self.kml_root.lookat.range = range_
        self.kml_root.lookat.gxtimespan.begin = t0.isoformat()
        self.kml_root.lookat.gxtimespan.end = t1.isoformat()

    def save(self, outpath='preview.kmz'):
        outpath = pathlib.Path(outpath)
        assert outpath.suffix.lower() in {'.kml', '.kmz', ''}
        self._kmzdir.mkdir(exist_ok=True)
        self.kml.save(self._kmzdir / 'doc.kml')

        if outpath.name.lower().endswith('.kmz'):
            shutil.make_archive(str(outpath.with_suffix('')),
                                format='zip', root_dir=str(self._kmzdir))
            shutil.move(outpath.with_suffix('.zip'), outpath)
            shutil.rmtree(self._kmzdir)
        else:
            shutil.move(self._kmzdir, outpath)

    def write_overall_footprint(self):
        data_footprint = self.etad.get_footprint(merge=True)
        corners = Sentinel1EtadKmlWriter._get_footprint_corners(data_footprint)

        pol = self.kml_root.newpolygon(name='footprint')
        pol.outerboundaryis = corners
        pol.altitudeMode = 'absolute'
        pol.tessellate = 1
        pol.polystyle.fill = 0
        pol.style.linestyle.width = 2

    @staticmethod
    def _get_footprint_corners(footprint):
        x, y = footprint.exterior.xy
        corners = list(zip(x, y))
        return corners

    @staticmethod
    def _get_burst_time_span(burst, t_ref):
        azimuth_time, _ = burst.get_burst_grid()
        t0 = t_ref + timedelta(seconds=azimuth_time[0])
        t1 = t_ref + timedelta(seconds=azimuth_time[-1])
        if t1 < t0:
            t0, t1 = t1, t0
        return t0, t1

    @staticmethod
    def _write_burst_footprint(burst, kml_dir, t_ref):
        corners = Sentinel1EtadKmlWriter._get_footprint_corners(
            burst.get_footprint())
        t0, t1 = Sentinel1EtadKmlWriter._get_burst_time_span(burst, t_ref)

        pol = kml_dir.newpolygon(name=str(burst.burst_index))
        pol.outerboundaryis = corners
        pol.altitudeMode = 'absolute'
        pol.tessellate = 1
        pol.polystyle.fill = 0
        pol.style.linestyle.width = 2
        pol.timespan.begin = t0.isoformat()
        pol.timespan.end = t1.isoformat()

    def write_burst_footprint(self):
        first_azimuth_time = parser.parse(self.etad.ds.azimuthTimeMin)

        kml_fp_dir = self.kml_root.newfolder(name='burst_footprint')
        selection = None
        for swath in self.etad.iter_swaths(selection):
            kml_swath_dir = kml_fp_dir.newfolder(name=swath.swath_id)
            for burst in swath.iter_bursts(selection):
                self._write_burst_footprint(burst, kml_swath_dir,
                                            first_azimuth_time)

    def write_corrections(self, correction_list, selection=None,
                          decimation_factor=1, colorizing=False):
        first_azimuth_time = parser.parse(self.etad.ds.azimuthTimeMin)

        for correction in correction_list:
            # get the parameter list
            prm_list = {}
            xp_ = f".//qualityAndStatistics/{correction}"
            for child in self.etad._annot.find(xp_).getchildren():
                tag = child.tag
                if 'range' in tag:
                    prm_list['x'] = tag
                elif 'azimuth' in tag:
                    prm_list['y'] = tag

            for dim, correction_name in prm_list.items():
                # only enable sum of corrections in range
                # TODO: make configurable
                if correction == 'sumOfCorrections' and dim == 'x':
                    visibility = True
                else:
                    visibility = False

                kml_cor_dir = self.kml_root.newfolder(
                    name=f"{correction}_{prm_list[dim]}")

                cor_max = np.max(
                    self.etad._xpath_to_list(
                        self.etad._annot,
                        f"{xp_}/{prm_list[dim]}/max[@unit='m']", dtype=np.float)
                )
                cor_min = np.min(
                    self.etad._xpath_to_list(
                        self.etad._annot,
                        f"{xp_}/{prm_list[dim]}/min[@unit='m']", dtype=np.float)
                )

                color_table = None
                gdal_palette = None
                if colorizing:
                    color_table = Colorizer(cor_min, cor_max)
                    gdal_palette = color_table.gdal_palette()

                color_table.build_colorbar(
                    self._kmzdir / f'{correction}_{dim}_cb.png')

                screen = kml_cor_dir.newscreenoverlay(name='ScreenOverlay')
                screen.icon.href = f"{correction}_{dim}_cb.png"
                screen.overlayxy = OverlayXY(x=0, y=0,
                                             xunits=Units.fraction,
                                             yunits=Units.fraction)
                screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                           xunits=Units.fraction,
                                           yunits=Units.fraction)
                screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                               xunits=Units.fraction,
                                               yunits=Units.fraction)

                for swath in self.etad.iter_swaths(selection):
                    kml_swath_dir = kml_cor_dir.newfolder(name=swath.swath_id)

                    for burst in swath.iter_bursts(selection):
                        corners = self._get_footprint_corners(burst.get_footprint())
                        t0, t1 = self._get_burst_time_span(burst, first_azimuth_time)

                        ground = kml_swath_dir.newgroundoverlay(
                            name='GroundOverlay')
                        ground.visibility = visibility
                        ground.name = (
                            f'{burst.product_index}_{burst.swath_index}_{burst.burst_index}'
                        )

                        ground.timespan.begin = t0.isoformat()
                        ground.timespan.end = t1.isoformat()

                        ground.gxlatlonquad.coords = corners

                        # ground.altitudeMode = 'absolute'
                        # ground.polystyle.fill = 0
                        # ground.tessellate=1
                        # ground.style.linestyle.width = 2

                        burst_img = f'burst_{swath.swath_id}_{burst.burst_index}_{correction}_{dim}'
                        ground.icon.href = burst_img + '.tiff'

                        # data
                        if correction == 'sumOfCorrections':
                            func_ = functools.partial(
                                burst.get_correction, name='sum')
                        elif correction == 'troposphericCorrection':
                            func_ = functools.partial(
                                burst.get_correction, name='tropospheric')
                        else:
                            raise RuntimeError(
                                f'unexpected correction: {correction!r}')

                        etad_correction = func_(meter=True)
                        cor = etad_correction[dim]
                        if colorizing is not None:
                            cor = (cor-cor_min) / np.abs(cor_max-cor_min) * 255
                            pixel_depth = gdal.GDT_Byte
                        else:
                            pixel_depth = gdal.GDT_Float32

                        cor = np.flipud(cor)

                        self.array2raster(self._kmzdir / burst_img, cor,
                                          color_table=gdal_palette,
                                          pixel_depth=pixel_depth,
                                          driver='GTiff',
                                          decimation_factor=decimation_factor,
                                          gcp_list=None)

    @staticmethod
    def array2raster(outfile, array, gcp_list=None, color_table=None,
                     pixel_depth=gdal.GDT_Float32, driver='GTiff',
                     decimation_factor=None):
        # http://osgeo-org.1560.x6.nabble.com/Transparent-PNG-with-color-table-palette-td3748906.html
        if decimation_factor is not None:
            array = array[::decimation_factor, ::decimation_factor]

        cols = array.shape[1]
        rows = array.shape[0]

        if driver == 'GTiff':
            outfile = outfile.with_suffix('.tiff')
        elif driver == 'PNG':
            outfile = outfile.with_suffix('.png')
        else:
            raise RuntimeError(f'unexpected driver: {driver}')

        driver = gdal.GetDriverByName(driver)
        outraster = driver.Create(str(outfile), cols, rows, 1, pixel_depth)

        # outRaster.SetGeoTransform(
        #     (originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outraster.GetRasterBand(1)
        if color_table is not None:
            assert(isinstance(color_table, gdal.ColorTable))
            outband.SetRasterColorTable(color_table)
        outband.WriteArray(array)

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(4326)
        outraster.SetProjection(out_srs.ExportToWkt())

        if gcp_list is not None:
            wkt = outraster.GetProjection()
            outraster.SetGCPs(gcp_list, wkt)

        outband.FlushCache()


# http://osgeo-org.1560.x6.nabble.com/Transparent-PNG-with-color-table-palette-td3748906.html
class Colorizer:
    def __init__(self, vmin, vmax, color_table=cm.viridis):
        # normalize item number values to colormap
        delta = np.abs(vmax - vmin)
        self.vmin = vmin - 0.05*delta
        self.vmax = vmax + 0.05*delta
        self.norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.color_table = color_table

    def rgba_color(self, value):
        # colormap possible values = viridis, jet, spectral
        if self.color_table is None:
            return int(value), int(value), int(value), int(value)
        else:
            return self.color_table(self.norm(value), bytes=True)

    def gdal_palette(self):
        value_list = np.linspace(self.vmin, self.vmax, 255)
        palette = gdal.ColorTable()
        for v in value_list:
            v_ = int((v-self.vmin) / np.abs(self.vmax-self.vmin) * 255)
            palette.SetColorEntry(v_, self.rgba_color(v)[0:3])
        return palette

    def build_colorbar(self, cb_filename):
        # https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
        fig = pyplot.figure(figsize=(0.8, 3))
        ax1 = fig.add_axes([0.1, 0.075, 0.25, 0.85])

        pyplot.tick_params(axis='y', which='major', labelsize=8)

        norm = self.norm

        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=self.color_table, norm=norm,
                                        orientation='vertical')
        cb1.set_label('[meters]', rotation=90, color='k')
        # This is called from plotpages, in <plotdir>.
        pathlib.Path(cb_filename).parent.mkdir(exist_ok=True)
        pyplot.savefig(str(cb_filename), transparent=False)


def etad_to_kmz(etad, outpath=None, *args, **kargs):
    if not isinstance(etad, Sentinel1Etad):
        etad = Sentinel1Etad(etad)

    if outpath is None:
        outpath = etad.product.stem + '.kmz'

    writer = Sentinel1EtadKmlWriter(etad, *args, **kargs)
    writer.save(outpath)
