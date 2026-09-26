"""Command-line interface for the TDMA planner (created in Division 1).

The CLI layer only does I/O: it parses the arguments, calls the pure functions in io_utils
and topology, prints, and turns InputError into a one-line message and exit code 2.
Exit codes: 0 = OK, 1 = verification failed (added in Division 2), 2 = bad input or usage.
"""
import argparse
import sys

from .io_utils import InputError, load_input
from .topology import (
    DEFAULT_RADIO_RANGE_M,
    build_conflict_graph,
    build_radio_graph,
    format_topology_summary,
    topology_summary,
    validate_radio_range,
)

EXIT_OK = 0
EXIT_BAD_INPUT = 2

_EXAMPLES = """\
examples:
  python tdma_schedule.py --file examples/grid16_300m.json --show-topology
  python tdma_schedule.py '{"Node_01":[0.0,0.0],"Node_02":[300.0,0.0]}' --show-topology

Windows PowerShell 5.1 removes the double quotes inside a JSON argument; use --file there.
"""


def build_parser():
    """The argparse parser (kept separate from main() so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="tdma_schedule.py",
        description="TDMA schedule planner: coordinates of radios in, radio graph and 2-hop "
        "conflict graph out (Division 1; the schedule itself arrives in Division 2).",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "coords_json",
        nargs="?",
        metavar="COORDS_JSON",
        help='node coordinates as a JSON string, e.g. \'{"Node_01":[0.0,0.0],"Node_02":[300.0,0.0]}\'',
    )
    parser.add_argument("--file", metavar="PATH", help="read the coordinates JSON from this file instead of the argument")
    parser.add_argument(
        "--range",
        dest="radio_range",
        type=float,
        default=DEFAULT_RADIO_RANGE_M,
        metavar="METRES",
        help="radio range in metres; two radios are linked when their distance is at most this (default: %(default)s)",
    )
    parser.add_argument(
        "--show-topology",
        action="store_true",
        help="print the topology summary: links, conflict pairs and hidden-terminal pairs",
    )
    return parser


def main(argv=None):
    """Run the CLI and return the process exit code (the entry points pass it to sys.exit)."""
    _make_output_ascii_safe()
    args = build_parser().parse_args(argv)
    try:
        validate_radio_range(args.radio_range)  # fail fast: no "continuing anyway" warning before an error
        coords = load_input(args)
        G = build_radio_graph(coords, args.radio_range)
        H = build_conflict_graph(G)
    except InputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    if args.show_topology:
        print(format_topology_summary(topology_summary(G, H)))
    else:
        print(
            "Note: scheduling arrives in Division 2. Use --show-topology to inspect the graphs.",
            file=sys.stderr,
        )
    return EXIT_OK


def _make_output_ascii_safe():
    """Never crash on a console or pipe that cannot encode a character.

    A non-ASCII node name on a Windows cp1252 console would otherwise raise
    UnicodeEncodeError; with this it is printed as an escape sequence instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
