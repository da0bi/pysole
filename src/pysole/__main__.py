"""Enable ``python -m pysole <config.json> [options]`` (equivalent to the ``pysole`` console script)."""

from .config import main_cli

if __name__ == "__main__":
    main_cli()
