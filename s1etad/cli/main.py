# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK

"""Simple CLI tool to access S1-ETAD products.

The tool provides a set of sub-commands to perform basic tasks involving
S1-ETAD products.
"""

import logging
import argparse

from . import utils as cliutils
from . import exportkmz, ql

try:
    from os import EX_OK
except ImportError:
    EX_OK = 0
EX_FAILURE = 1
EX_INTERRUPT = 130

PROG = __package__.split('.')[0]
# LOGFMT = '%(asctime)s %(levelname)-8s -- %(message)s'
LOGFMT = '%(asctime)s %(name)s %(levelname)s -- %(message)s'


def get_parser():
    """Instantiate the command line argument parser."""
    description = __doc__
    parser = argparse.ArgumentParser(description=description, prog=PROG)

    # Sub-command management
    subparsers = parser.add_subparsers(title='sub-commands')  # dest='func'
    exportkmz.get_parser(subparsers=subparsers)
    ql.get_parser(subparsers=subparsers)

    parser = cliutils.finalize_parser(parser)

    return parser


def parse_args(args=None, namespace=None, parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = get_parser()

    args = parser.parse_args(args, namespace)

    # Common pre-processing of parsed arguments and consistency checks
    # ...

    if getattr(args, 'func', None) is None:
        parser.error('no sub-commnd specified.')

    return args


def main(*argv):
    """Main CLI interface."""
    # setup logging
    logging.basicConfig(format=LOGFMT, level=logging.INFO)  # stream=sys.stdout
    logging.captureWarnings(True)
    log = logging.getLogger(PROG)

    # parse cmd line arguments
    args = parse_args(argv if argv else None)

    # execute main tasks
    exit_code = EX_OK
    try:
        log.setLevel(args.loglevel)
        log.debug('args: %s', args)

        func = cliutils.get_function(args.func)
        kwargs = cliutils.get_kwargs(args)
        func(**kwargs)
    except Exception as exc:
        log.critical(
            'unexpected exception caught: {!r} {}'.format(
                type(exc).__name__, exc))
        log.debug('stacktrace:', exc_info=True)
        exit_code = EX_FAILURE
    except KeyboardInterrupt:
        log.warning('Keyboard interrupt received: exit the program')
        exit_code = EX_INTERRUPT

    return exit_code
