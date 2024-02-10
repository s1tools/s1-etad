"""Utility functions for the management of command line arguments."""

import logging
import importlib

from .. import __version__

try:
    import argcomplete
except ImportError:
    argcomplete = None


def set_logging_control_args(parser, default_loglevel="WARNING"):
    """Set up command line options for logging control."""
    loglevels = [logging.getLevelName(level) for level in range(10, 60, 10)]

    parser.add_argument(
        "--loglevel",
        default=default_loglevel,
        choices=loglevels,
        help="logging level (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="loglevel",
        action="store_const",
        const="ERROR",
        help=(
            "suppress standard output messages, "
            "only errors are printed to screen"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const="INFO",
        help="print verbose output messages",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        help="print debug messages",
    )

    return parser


def finalize_parser(parser):
    """Finalize the argument parser provided in input.

    Arguments for logging control and version printing are added.
    If the "argcomplete" module is available the automatic completion for
    the parser is also se-up.

    The updated parser is returned.
    """
    parser = set_logging_control_args(parser)
    parser.add_argument(
        "--version", action="version", version="%(prog)s v" + __version__
    )

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser


def get_function(func):
    """Return the function corresponding to the input.

    If the input `func` parameter is already a callable then it is
    immediately returned.

    If the input is a string in the form "modname.[...].funcname" then the
    required Python package/module is imported and the specified function
    ("funcname" in the example) is returned.
    """
    if callable(func):
        return func

    fullname = func
    if "." not in fullname:
        raise ValueError(
            f"unable to retrieve the module name from: {fullname}"
        )

    modulename, funcname = fullname.rsplit(".", maxsplit=1)
    module = importlib.import_module(modulename)
    return getattr(module, funcname)


def get_kwargs(args):
    """Convert an argparse.Namespace into a dictionary.

    The "loglevel" and "func" arguments are never included in the output
    dictionary.
    """
    kwargs = vars(args).copy()
    kwargs.pop("func", None)
    kwargs.pop("loglevel", None)
    return kwargs
