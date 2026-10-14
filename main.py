"""Run one of the projects in this repository.

    python main.py air-travel      [options]   # DSA2101: air travel vs. holidays
    python main.py har             [options]   # CS3244: activity recognition
    python main.py firmographics   [options]   # SDS datathon: company segmentation

Options are forwarded to the project; use ``python main.py <project> --help``.
"""

import importlib
import sys

PROJECTS = {
    "air-travel": "src.air_travel.__main__",
    "har": "src.har_classification.__main__",
    "firmographics": "src.firmographics.__main__",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in PROJECTS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    importlib.import_module(PROJECTS[sys.argv[1]]).main(sys.argv[2:])


if __name__ == "__main__":
    main()
