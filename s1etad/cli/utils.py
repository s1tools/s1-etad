"""Utility functions for the management of command line arguments."""

import logging
import argparse
import importlib
from typing import Callable

from .. import __version__

DEFAULT_LOGLEVEL = "WARNING"


def _autocomplete(parser: argparse.ArgumentParser) -> None:
    try:
        import argcomplete
    except ImportError:
        pass
    else:
        argcomplete.autocomplete(parser)


def _add_logging_control_args(
    parser: argparse.ArgumentParser, default_loglevel: str = DEFAULT_LOGLEVEL
) -> argparse.ArgumentParser:
    """Add command line options for logging control."""
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
            "only errors are printed to screen (set 'loglevel' to 'ERROR')"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const="INFO",
        help="print verbose output messages (set 'loglevel' to 'INFO')",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        help="print debug messages (set 'loglevel' to 'DEBUG')",
    )

    return parser


def finalize_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Finalize the argument parser provided in input.

    Arguments for logging control and version printing are added.
    If the "argcomplete" module is available the automatic completion for
    the parser is also set-up.

    The updated parser is returned.
    """
    parser = _add_logging_control_args(parser)
    parser.add_argument(
        "--version", action="version", version="%(prog)s v" + __version__
    )

    _autocomplete(parser)

    return parser


def get_function(func: str | Callable) -> Callable:
    """Return the function corresponding to the input.

    If the input `func` parameter is already a callable then it is
    immediately returned.

    If the input is a string in the form "modname.[...].funcname" then the
    required Python package/module is imported and the specified function
    ("funcname" in the example) is returned.
    """
    if callable(func):
        return func

    fullname: str = func
    if "." not in fullname:
        raise ValueError(
            f"unable to retrieve the module name from: {fullname}"
        )

    modulename, funcname = fullname.rsplit(".", maxsplit=1)
    module = importlib.import_module(modulename)
    return getattr(module, funcname)
