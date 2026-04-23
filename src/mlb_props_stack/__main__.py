"""Module entrypoint for ``python -m mlb_props_stack``."""

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
