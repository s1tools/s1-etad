"""Geo-coded QuickLook image generation for ETAD."""

import functools
import os
from typing import List, Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from osgeo import gdal, osr

from . import Sentinel1Etad, ECorrectionType
from .product import CorrectionType                                     # noqa


MAX_GCP_NUM = 10000                 # empirical threshold
DEFAULT_LATLON_SPACING_DEG = 0.005  # deg --> 550m @ equator, 50 @ lat=85deg
DEFAULT_COLOR_TABLE_NAME = 'jet'    # form matplotlib


def _write_band_data(band, data, nodata: float = -9999.):
    if hasattr(data, 'filled'):
        data = data.filled(nodata)
    band.WriteArray(data)
    band.SetNoDataValue(nodata)


def create_gcps(lat, lon, h=None, gcp_step=(10, 10)) -> List[gdal.GCP]:
    """Generate a sub-sampled grid of GCPs form input coordinate matrices."""
    assert lat.shape == lon.shape
    ysize, xsize = lat.shape
    ystep, xstep = gcp_step

    masks = [
        data.mask if hasattr(data, 'mask') else None for data in (lat, lon, h)
    ]

    mask: Optional[np.array] = None
    if masks:
        mask = functools.reduce(np.logical_or, masks)

    gcps = []
    for line in range(0, ysize, ystep):
        for pix in range(0, xsize, xstep):
            if mask is None or mask[line, pix]:
                continue
            height = h[line, pix] if h is not None else 0.
            gcp_info = ''
            gcp_id = f'len(gcps)'
            gcp = gdal.GCP(lon[line, pix], lat[line, pix], height,
                           pix, line, gcp_info, gcp_id)
            gcps.append(gcp)

    assert 0 < len(gcps) <= MAX_GCP_NUM

    return gcps


def save_with_gcps(outfile: str, data, lat, lon, h=None,
                   *, drv_name: str = 'GTIFF', nodata: float = -9999.,
                   gcp_step=(10, 10), srs='wgs84', creation_options=None):
    """Save data into a GDAL dataset and GCPs for coordinates matrices."""
    drv = gdal.GetDriverByName(drv_name)
    assert drv is not None

    ysize, xsize = data.shape
    if creation_options is None:
        creation_options = []
    ds = drv.Create(str(outfile), xsize=xsize, ysize=ysize,
                    bands=1, eType=gdal.GDT_Float32,
                    options=creation_options)

    if isinstance(srs, str):
        srs_str = srs
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS(srs_str)
    gcps = create_gcps(lat, lon, h, gcp_step)
    ds.SetGCPs(gcps, srs)

    _write_band_data(ds.GetRasterBand(1), data, nodata)

    return ds


def _clip_bbox(bbox, q, margin=0):
    return (
        np.floor(bbox[0] / q) * q - margin * q,
        np.floor(bbox[1] / q) * q - margin * q,
        np.ceil(bbox[2] / q) * q + margin * q,
        np.ceil(bbox[3] / q) * q + margin * q,
    )


def _compute_gcp_spacing(xsize, ysize, max_gcp_num: int = MAX_GCP_NUM):
    # assume 200 x 200 m ground spacing for ETAD products
    gcp_step = (25, 25)  # 5 x 5 km
    # gcp_step = (50, 50)  # 10 x 10 km
    # gcp_step = (100, 100)  # 20 x 20 km
    while (ysize // gcp_step[0]) * (xsize // gcp_step[1]) > max_gcp_num:
        # increase the step only in the azimuth direction
        gcp_step = (gcp_step[0] * 2, gcp_step[1])
    return gcp_step


@functools.lru_cache()  # COMPATIBILITY with Python < 3.8
def _get_color_table(name=DEFAULT_COLOR_TABLE_NAME):
    from matplotlib import cm
    from .kmz import Colorizer  # noqa

    cmap = getattr(cm, name)
    # return Colorizer(1, 255, color_table=cmap).gdal_palette()

    table = gdal.ColorTable()
    table.SetColorEntry(0, (0, 0, 0, 0))  # zero is transparent
    for i, v in enumerate(np.linspace(0., 1., 255), start=1):
        table.SetColorEntry(i, cmap(v, bytes=True))
    return table


def save_geocoded_data(outfile, data, lat, lon, h=None, *,
                       gcp_step: Optional[Tuple[int, int]] = None,
                       srs='wgs84', out_spacing=DEFAULT_LATLON_SPACING_DEG,
                       drv_name='GTIFF', creation_options=None,
                       palette=DEFAULT_COLOR_TABLE_NAME, margin=100):
    """Save a geo-coded version of input data into a GDAL dataset."""
    ysize, xsize = data.shape
    if gcp_step is None:
        gcp_step = _compute_gcp_spacing(xsize, ysize)

    # dataset with GCP grid
    ds_sr_with_gcps = save_with_gcps('', data, lat=lat, lon=lon, h=h,
                                     gcp_step=gcp_step, drv_name='MEM')

    # geocode the floating point image
    bbox = (lon.min(), lat.min(), lon.max(), lat.max())
    bbox = _clip_bbox(bbox, out_spacing, margin=margin)

    ds_geocoded_float = gdal.Warp('', ds_sr_with_gcps,
                                  format='MEM',
                                  dstSRS=srs,
                                  xRes=out_spacing,
                                  yRes=out_spacing,
                                  targetAlignedPixels=True,
                                  outputBounds=bbox,
                                  outputBoundsSRS=srs)

    # scale the geocoded image to bytes
    scale_params = [[data.min(), data.max(), 0, 255]]  # NOTE: list of lists
    ds_geocoded_bytes = gdal.Translate('', ds_geocoded_float,
                                       format='MEM',
                                       outputType=gdal.GDT_Byte,
                                       noData=0,
                                       scaleParams=scale_params)
    # attache the color palette
    if isinstance(palette, str):
        palette = _get_color_table()

    band = ds_geocoded_bytes.GetRasterBand(1)
    band.SetRasterColorTable(palette)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    del band

    # Save to disk
    if creation_options is None:
        creation_options = []
    ds_out = gdal.Translate(os.fspath(outfile), ds_geocoded_bytes,
                            format=drv_name, creationOptions=creation_options)
    # , rgbExpand='rgba')

    return ds_out


def etad2ql(etad, outpath, *,
            correction_type: CorrectionType = ECorrectionType.SUM,
            direction: Literal['x', 'y'] = 'x', meter: bool = True,
            drv_name: str = 'PNG', creation_options=['WORLDFILE=YES']):
    """Generate a geo-coded quick-look image starting from an ETAD product."""
    if not isinstance(etad, Sentinel1Etad):
        etad = Sentinel1Etad(etad)
    else:
        etad = etad

    if outpath is None:
        outpath = etad.product.with_suffix('.png').name

    merged_correction = etad.merge_correction(correction_type, meter=meter)

    # masked arrays
    data = merged_correction[direction]
    lat = merged_correction['lats']
    lon = merged_correction['lons']
    height = merged_correction['height']

    return save_geocoded_data(outpath, data, lat, lon, height,
                              drv_name=drv_name,
                              creation_options=creation_options)
