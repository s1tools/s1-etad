# -*- coding: utf-8 -*-

from.product import Sentinel1Etad, Sentinel1EtadSwath, Sentinel1EtadBurst


def _sentinel1_etad_repr_pretty_(obj, p, cycle):
    if cycle:
        p.text(repr(obj))
    else:
        p.text(repr(obj))
        p.break_()
        plist = obj.s1_product_list()
        if isinstance(plist, str):
            plist = [plist]
        p.text(f'Number of Sentinel-1 slices: {len(plist)}')
        p.break_()
        with p.group(2, 'Sentinel-1 products list:'):
            for name in plist:
                p.break_()
                p.text(name)
        p.break_()
        p.text(f'Number of swaths: {obj.number_of_swath}')
        p.break_()
        p.text('Swath list: {}'.format(', '.join(obj.swath_list)))
        p.break_()
        with p.group(2, 'Azimuth time:'):
            p.break_()
            p.text(f'min: {obj.min_azimuth_time}')
            p.break_()
            p.text(f'max: {obj.max_azimuth_time}')
        p.break_()
        with p.group(2, 'Range time:'):
            p.break_()
            p.text(f'min: {obj.min_range_time}')
            p.break_()
            p.text(f'max: {obj.max_range_time}')
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
