"""Main entripoint for s1etad CLI."""

# PYTHON_ARGCOMPLETE_OK

import sys

from .cli.main import main

sys.exit(main())
