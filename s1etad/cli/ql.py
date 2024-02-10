"""Generate a geo-coded quick-look image from ETAD correction."""

import argparse

from . import utils as cliutils


def get_parser(subparsers=None):
    """Instantiate the command line argument parser."""
    name = "ql"
    synopsis = __doc__.splitlines()[0]
    doc = __doc__

    if subparsers is None:
        parser = argparse.ArgumentParser(prog=name, description=doc)
    else:
        parser = subparsers.add_parser(name, description=doc, help=synopsis)

    parser.set_defaults(func="s1etad.ql.etad2ql")

    # command line options
    # ...

    # positional arguments
    parser.add_argument(
        "etad",
        metavar="etad-path",
        help="path to the S1-ETAD products (a directory)",
    )
    parser.add_argument(
        "outpath",
        nargs="?",
        help="the pathname of the generated KMZ file. "
        "If not provided then a PNG file having the same input product "
        "basename is generated in the current working directory",
    )

    if not subparsers:
        parser = cliutils.finalize_parser(parser)

    return parser
