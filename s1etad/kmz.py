"""Support for Google KMZ preview generation."""

import shutil
import pathlib
import datetime
import functools
from typing import Literal
from collections.abc import Sequence

import numpy as np
import matplotlib as mpl
from osgeo import gdal, osr
from simplekml import Kml, OverlayXY, RotationXY, ScreenXY, Units
from matplotlib import cm
from matplotlib import pyplot as plt

from .utils import iter_corrections
from .product import ECorrectionType, Sentinel1Etad

__all__ = ["etad_to_kmz", "Sentinel1EtadKmlWriter"]


class Sentinel1EtadKmlWriter:
    """Writer for ETAD KML/KMZ files."""

    # TODO: only SUM by default
    DEFAULT_CORRECTIONS: Sequence[ECorrectionType] = (
        ECorrectionType.SUM,
        ECorrectionType.TROPOSPHERIC,
    )
    DEFAULT_TIMESPAN = 30  # [s]
    DEFAULT_LOOKAT_RANGE = 1500000
    DEFAULT_OPEN_FOLDER: bool = False

    def __init__(
        self,
        etad,
        corrections=None,
        timespan=DEFAULT_TIMESPAN,
        selection=None,
        decimation_factor: int = 1,
        colorizing: bool = True,
        open_folders: bool = DEFAULT_OPEN_FOLDER,
    ):
        """Initialize a `Sentinel1EtadKmlWriter` object."""
        assert isinstance(etad, Sentinel1Etad)
        self.etad = etad

        if corrections is None:
            corrections = list(self.DEFAULT_CORRECTIONS)  # make a copy

        if selection is None:
            selection = self.etad.burst_catalogue

        if len(selection) < 1:
            raise ValueError(f"no burst found in selection: {selection}")

        self._corrections = corrections
        self._selection = selection
        self._decimation_factor = decimation_factor
        self._colorizing = colorizing
        self._open_folders = open_folders

        self.kml = Kml(open=self._open_folders)
        self.kml_root = self.kml.newfolder(
            name=self.etad.product.stem, open=self._open_folders
        )

        if timespan is not None:
            timespan = self.DEFAULT_TIMESPAN if timespan is True else timespan
            self.set_timespan(duration=timespan)

        self.add_overall_footprint()
        self.add_burst_footprints()

    def set_timespan(
        self, duration=DEFAULT_TIMESPAN, range_=DEFAULT_LOOKAT_RANGE
    ):
        """Set the time span in the KML file."""
        record = self._selection.iloc[0]
        swath = self.etad[record.swathID]
        burst = swath[record.bIndex]

        lon0, lat0, lon1, lat1 = burst.get_footprint().bounds
        lat = np.mean([lat0, lat1])
        lon = np.mean([lon0, lon1])
        t0 = self.etad.min_azimuth_time
        t1 = t0 + datetime.timedelta(seconds=duration)  # configure duration

        self.kml_root.lookat.latitude = lat
        self.kml_root.lookat.longitude = lon
        self.kml_root.lookat.range = range_
        self.kml_root.lookat.gxtimespan.begin = t0.isoformat()
        self.kml_root.lookat.gxtimespan.end = t1.isoformat()

        return self.kml_root.lookat

    @staticmethod
    def _get_footprint_corners(footprint):
        x, y = footprint.exterior.xy
        corners = list(zip(x, y, strict=True))
        return corners

    def add_overall_footprint(self):
        """Add the overall footprint to the KML file."""
        data_footprint = self.etad.get_footprint(self._selection, merge=True)
        if not hasattr(data_footprint, "exterior"):
            # it is a multipolygon

            # option 1
            # data_footprint = data_footprint.convex_hull

            # option 2
            # import shapely.ops
            # data_footprint = shapely.ops.unary_union(
            #     [data_footprint, data_footprint.convex_hull])

            # option 3: write multiple footprints
            data_footprints = data_footprint
            for data_footprint in data_footprints:
                corners = Sentinel1EtadKmlWriter._get_footprint_corners(
                    data_footprint
                )

                polygon = self.kml_root.newpolygon(name="footprint")
                polygon.outerboundaryis = corners
                polygon.altitudeMode = "absolute"
                polygon.tessellate = 1
                polygon.polystyle.fill = 0
                polygon.style.linestyle.width = 2

            return None
            # TODO: find a better solution

        corners = Sentinel1EtadKmlWriter._get_footprint_corners(data_footprint)

        polygon = self.kml_root.newpolygon(name="footprint")
        polygon.outerboundaryis = corners
        polygon.altitudeMode = "absolute"
        polygon.tessellate = 1
        polygon.polystyle.fill = 0
        polygon.style.linestyle.width = 2

        return polygon

    @staticmethod
    def _get_burst_time_span(burst, t_ref):
        azimuth_time, _ = burst.get_burst_grid()
        t0 = t_ref + datetime.timedelta(seconds=azimuth_time[0])
        t1 = t_ref + datetime.timedelta(seconds=azimuth_time[-1])
        if t1 < t0:
            t0, t1 = t1, t0
        return t0, t1

    @staticmethod
    def _add_burst_footprint(burst, kml_dir, t_ref):
        corners = Sentinel1EtadKmlWriter._get_footprint_corners(
            burst.get_footprint()
        )
        t0, t1 = Sentinel1EtadKmlWriter._get_burst_time_span(burst, t_ref)

        polygon = kml_dir.newpolygon(name=burst.burst_id)
        polygon.description = f"""\
<![CDATA[
<p>{burst.product_id}</p>
<p>swath = {burst.swath_id},</p>
<p>first_time = '{t0.isoformat()}',</p>
<p>last_time = '{t1.isoformat()}',</p>
<p>Index (product, swath, burst) =
 ({burst.product_index}, {burst.swath_index}, {burst.burst_index})
</p>
]]>"""
        polygon.outerboundaryis = corners
        polygon.altitudeMode = "absolute"
        polygon.tessellate = 1
        polygon.polystyle.fill = 0
        polygon.style.linestyle.width = 2
        polygon.timespan.begin = t0.isoformat()
        polygon.timespan.end = t1.isoformat()

    def add_burst_footprints(self):
        """Add the footprint of a burst to the KML file."""
        t_ref = self.etad.min_azimuth_time
        kml_footprint_dir = self.kml_root.newfolder(
            name="burst_footprint", open=self._open_folders
        )
        for swath in self.etad.iter_swaths(self._selection):
            kml_swath_dir = kml_footprint_dir.newfolder(name=swath.swath_id)
            for burst in swath.iter_bursts(self._selection):
                self._add_burst_footprint(burst, kml_swath_dir, t_ref)

    def _colorbar_overlay(
        self,
        correction,
        dim: Literal["x", "y"],
        kml_cor_dir,
        color_table,
        outpath,
    ):
        assert isinstance(correction, ECorrectionType)

        filename = f"{correction.value}_{dim}_cb.png"
        color_table.build_colorbar(outpath / filename)

        screen = kml_cor_dir.newscreenoverlay(name="ColorBar")
        screen.icon.href = filename
        screen.overlayxy = OverlayXY(
            x=0, y=0, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.screenxy = ScreenXY(
            x=0.015, y=0.075, xunits=Units.fraction, yunits=Units.fraction
        )
        screen.rotationXY = RotationXY(
            x=0.5, y=0.5, xunits=Units.fraction, yunits=Units.fraction
        )
        return screen

    def _ground_overlay_node(self, kml_dir, burst):
        corners = self._get_footprint_corners(burst.get_footprint())
        t0, t1 = self._get_burst_time_span(burst, self.etad.min_azimuth_time)

        ground = kml_dir.newgroundoverlay(name="GroundOverlay")
        ground.name = burst.burst_id

        ground.timespan.begin = t0.isoformat()
        ground.timespan.end = t1.isoformat()

        ground.gxlatlonquad.coords = corners

        # ground.altitudeMode = 'absolute'
        # ground.polystyle.fill = 0
        # ground.tessellate=1
        # ground.style.linestyle.width = 2

        return ground

    @staticmethod
    def _ground_overlay_data(
        correction,
        burst,
        dim: Literal["x", "y"],
        vmin: float,
        vmax: float,
        colorizing: bool,
    ):
        func = functools.partial(
            burst.get_correction, name=correction, direction=dim
        )

        etad_correction = func(meter=True)
        data = etad_correction[dim]
        if colorizing:
            data = 255 / np.abs(vmax - vmin) * (data - vmin)

        data = np.flipud(data)  # TODO: check ascending/descending

        return data

    @staticmethod
    def _get_visibility(correction, dim: Literal["x", "y"]) -> bool:
        if correction == ECorrectionType.SUM and dim == "x":
            return True
        else:
            return False

    def add_ground_overlays(self, outpath):
        """Add the ground overlay to the KML file."""
        outpath = pathlib.Path(outpath)
        outpath.mkdir(exist_ok=True)

        pixel_depth = gdal.GDT_Byte if self._colorizing else gdal.GDT_Float32

        for correction, dim in iter_corrections(self._corrections):
            # only enable sum of corrections in range
            # TODO: make configurable
            # TODO: support cases in which SUM is not in the corrections list
            visibility = self._get_visibility(correction, dim)

            kml_correction_dir = self.kml_root.newfolder(
                name=f"{correction.value}_{dim}", open=self._open_folders
            )
            statistics = self.etad.get_statistics(correction, meter=True)
            vmin = statistics[dim].min
            vmax = statistics[dim].max

            gdal_palette = None
            if self._colorizing:
                color_table = Colorizer(vmin, vmax)
                gdal_palette = color_table.gdal_palette()
                colorbar_overlay = self._colorbar_overlay(
                    correction, dim, kml_correction_dir, color_table, outpath
                )
                colorbar_overlay.visibility = visibility

            for swath in self.etad.iter_swaths(self._selection):
                kml_swath_dir = kml_correction_dir.newfolder(
                    name=swath.swath_id
                )

                for burst in swath.iter_bursts(self._selection):
                    ground = self._ground_overlay_node(kml_swath_dir, burst)
                    ground.visibility = visibility

                    data = self._ground_overlay_data(
                        correction, burst, dim, vmin, vmax, self._colorizing
                    )

                    b = burst
                    burst_img = (
                        f"burst_{b.swath_id}_{b.burst_index:03d}_"
                        f"{correction.value}_{dim}"
                    )

                    ds = array2raster(
                        outpath / burst_img,
                        data,
                        color_table=gdal_palette,
                        pixel_depth=pixel_depth,
                        driver="GTiff",
                        decimation_factor=self._decimation_factor,
                        gcp_list=None,
                    )

                    ground.icon.href = pathlib.Path(ds.GetDescription()).name

    def save(self, outpath="preview.kmz"):
        """Save the KMZ file for the ETAD product."""
        outpath = pathlib.Path(outpath)

        assert outpath.suffix.lower() in {".kml", ".kmz", ""}
        if outpath.exists():
            raise FileExistsError(f'output path already exists "{outpath}"')

        kmzdir = outpath.with_suffix("")
        if kmzdir.exists():
            raise FileExistsError(f'output path already exists "{kmzdir}"')
        kmzdir.mkdir()

        self.add_ground_overlays(kmzdir)
        self.kml.save(kmzdir / "doc.kml")

        if outpath.name.lower().endswith(".kmz"):
            shutil.make_archive(
                str(outpath.with_suffix("")),
                format="zip",
                root_dir=str(kmzdir),
            )
            shutil.move(outpath.with_suffix(".zip"), outpath)
            shutil.rmtree(kmzdir)
        else:
            shutil.move(kmzdir, outpath)


def array2raster(
    outfile,
    array,
    gcp_list=None,
    color_table=None,
    pixel_depth=gdal.GDT_Float32,
    driver: str = "GTiff",
    decimation_factor: int | None = None,
):
    # http://osgeo-org.1560.x6.nabble.com/Transparent-PNG-with-color-table-palette-td3748906.html
    if decimation_factor is not None:
        array = array[::decimation_factor, ::decimation_factor]

    cols = array.shape[1]
    rows = array.shape[0]

    if driver == "GTiff":
        outfile = outfile.with_suffix(".tiff")
    elif driver == "PNG":
        outfile = outfile.with_suffix(".png")
    else:
        raise RuntimeError(f"unexpected driver: {driver}")

    drv = gdal.GetDriverByName(driver)
    outraster = drv.Create(str(outfile), cols, rows, 1, pixel_depth)
    assert outraster is not None

    # outRaster.SetGeoTransform(
    #     (originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outraster.GetRasterBand(1)
    if color_table is not None:
        assert isinstance(color_table, gdal.ColorTable)
        outband.SetRasterColorTable(color_table)
    outband.WriteArray(array)

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    outraster.SetProjection(out_srs.ExportToWkt())

    if gcp_list is not None:
        wkt = outraster.GetProjection()
        outraster.SetGCPs(gcp_list, wkt)

    outband.FlushCache()

    return outraster


# http://osgeo-org.1560.x6.nabble.com/Transparent-PNG-with-color-table-palette-td3748906.html
class Colorizer:
    def __init__(self, vmin: float, vmax: float, color_table=cm.viridis):
        # normalize item number values to colormap
        delta = np.abs(vmax - vmin)
        self.vmin = vmin - 0.05 * delta
        self.vmax = vmax + 0.05 * delta
        self.norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.color_table = color_table

    def rgba_color(self, value):
        # colormap possible values = viridis, jet, spectral
        if self.color_table is None:
            return int(value), int(value), int(value), int(value)
        else:
            return self.color_table(self.norm(value), bytes=True)

    def gdal_palette(self):
        values = np.linspace(self.vmin, self.vmax, 255)
        norm_values = (
            255 / np.abs(self.vmax - self.vmin) * (values - self.vmin)
        )
        norm_values = norm_values.astype(np.uint8)
        palette = gdal.ColorTable()
        for v, vn in zip(values, norm_values, strict=True):
            palette.SetColorEntry(int(vn), self.rgba_color(v)[0:3])
        return palette

    def build_colorbar(self, cb_filename):
        # https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
        fig = plt.figure(figsize=(0.8, 3))
        ax1 = fig.add_axes(rect=(0.1, 0.075, 0.25, 0.85))

        plt.tick_params(axis="y", which="major", labelsize=8)

        norm = self.norm

        cb1 = mpl.colorbar.ColorbarBase(
            ax1, cmap=self.color_table, norm=norm, orientation="vertical"
        )
        cb1.set_label("[meters]", rotation=90, color="k")
        # This is called from plotpages, in <plotdir>.
        pathlib.Path(cb_filename).parent.mkdir(exist_ok=True)
        plt.savefig(str(cb_filename), transparent=False)


def etad_to_kmz(etad, outpath=None, *args, **kargs):
    """Generate a KMZ file from an ETAD product."""
    if not isinstance(etad, Sentinel1Etad):
        etad = Sentinel1Etad(etad)

    if outpath is None:
        outpath = etad.product.stem + ".kmz"

    writer = Sentinel1EtadKmlWriter(etad, *args, **kargs)
    writer.save(outpath)
