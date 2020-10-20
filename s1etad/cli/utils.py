# -*- coding: utf-8 -*-

import logging
import importlib

from .. import __version__

try:
    import argcomplete
except ImportError:
    argcomplete = None


def set_logging_control_args(parser, default_loglevel='WARNING'):
    """Setup command line options for logging control."""

    loglevels = [logging.getLevelName(level) for level in range(10, 60, 10)]

    parser.add_argument(
        '--loglevel', default=default_loglevel, choices=loglevels,
        help='logging level (default: %(default)s)')
    parser.add_argument(
        '-q', '--quiet', dest='loglevel', action='store_const',
        const='ERROR',
        help='suppress standard output messages, '
             'only errors are printed to screen')
    parser.add_argument(
        '-v', '--verbose', dest='loglevel', action='store_const',
        const='INFO', help='print verbose output messages')
    parser.add_argument(
        '-d', '--debug', dest='loglevel', action='store_const',
        const='DEBUG', help='print debug messages')

    return parser


def finalize_parser(parser):
    parser = set_logging_control_args(parser)
    parser.add_argument(
        '--version', action='version', version='%(prog)s v' + __version__)

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser


def get_function(func):
    if callable(func):
        return func

    fullname = func
    if '.' not in fullname:
        raise ValueError(f'unable to retrieve the module name from: {fullname}')

    modulename, funcname = fullname.rsplit('.', maxsplit=1)
    module = importlib.import_module(modulename)
    return getattr(module, funcname)


def get_kwargs(args):
    kwargs = vars(args).copy()
    kwargs.pop('func', None)
    kwargs.pop('loglevel', None)
    return kwargs
