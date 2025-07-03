"""Simple CLI tool to access S1-ETAD products.

The tool provides a set of sub-commands to perform basic tasks involving
S1-ETAD products.
"""

# PYTHON_ARGCOMPLETE_OK

import logging
import argparse
from collections.abc import Sequence

from . import exportkmz, ql
from . import utils as cliutils

EX_OK = 0
EX_FAILURE = 1
EX_INTERRUPT = 130

PROG = __package__.split(".")[0]
LOGFMT = "%(asctime)s %(name)s %(levelname)s -- %(message)s"
DEFAULT_LOGLEVEL = "INFO"


def get_parser() -> argparse.ArgumentParser:
    """Instantiate the command line argument parser."""
    description = __doc__
    parser = argparse.ArgumentParser(prog=PROG, description=description)

    # Sub-command management
    subparsers = parser.add_subparsers(title="sub-commands")  # dest='func'
    exportkmz.get_parser(subparsers=subparsers)
    ql.get_parser(subparsers=subparsers)

    parser = cliutils.finalize_parser(parser)

    return parser


def parse_args(
    args: Sequence[str] | None = None,
    namespace: argparse.Namespace | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command line arguments."""
    if parser is None:
        parser = get_parser()

    out_args = parser.parse_args(args, namespace)

    # Common pre-processing of parsed arguments and consistency checks
    # ...

    if getattr(out_args, "func", None) is None:
        parser.error("no sub-command specified.")

    return out_args


def _get_kwargs(args) -> dict:
    kwargs = vars(args).copy()
    kwargs.pop("func", None)
    kwargs.pop("loglevel", None)
    return kwargs


def main(*argv: str) -> int:
    """Implement the main CLI interface."""
    # setup logging
    logging.basicConfig(format=LOGFMT, level=DEFAULT_LOGLEVEL)
    logging.captureWarnings(True)
    log = logging.getLogger(PROG)

    # parse cmd line arguments
    args = parse_args(argv if argv else None)

    exit_code = EX_OK
    try:
        # NOTE: use the root logger to set the logging level
        logging.getLogger().setLevel(args.loglevel)

        log.debug("args: %s", args)

        func = cliutils.get_function(args.func)
        kwargs = _get_kwargs(args)
        func(**kwargs)
    except Exception as exc:  # noqa: B902, BLE001
        log.critical(
            "unexpected exception caught: %r %s", type(exc).__name__, exc
        )
        log.debug("stacktrace:", exc_info=True)
        exit_code = EX_FAILURE
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received: exit the program")
        exit_code = EX_INTERRUPT

    return exit_code
