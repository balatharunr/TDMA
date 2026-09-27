"""Command-line interface for the TDMA planner (created in Division 1, extended in Division 2).

The CLI layer only does I/O: it parses the arguments, calls the pure functions of the other
modules, prints, and turns InputError into a one-line message and exit code 2.
Exit codes: 0 = schedule produced and verified, 1 = verification failed, 2 = bad input or usage.
"""
import argparse
import sys
from pathlib import Path

from .heuristics import ALGORITHM_KEYS, run_algorithm
from .io_utils import InputError, load_input
from .report import format_report
from .schedule import make_schedule, to_json
from .topology import (
    DEFAULT_RADIO_RANGE_M,
    build_conflict_graph,
    build_radio_graph,
    format_topology_summary,
    topology_summary,
    validate_radio_range,
)
from .verify import verify_schedule

EXIT_OK = 0
EXIT_VERIFICATION_FAILED = 1
EXIT_BAD_INPUT = 2

_EXAMPLES = """\
examples:
  python tdma_schedule.py --file examples/grid16_300m.json
  python tdma_schedule.py '{"Node_01":[0.0,0.0],"Node_02":[300.0,0.0],"Node_03":[600.0,0.0]}'
  python tdma_schedule.py --file examples/grid16_300m.json --algorithm naive-d1    # why distance-2 matters
  python tdma_schedule.py --file examples/grid16_300m.json --export-json results/grid16_300m.schedule.json

exit codes: 0 = schedule verified conflict-free, 1 = verification failed, 2 = bad input or usage

Windows PowerShell 5.1 removes the double quotes inside a JSON argument; use --file there.
"""


def build_parser():
    """The argparse parser (kept separate from main() so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="tdma_schedule.py",
        description="TDMA schedule planner: turns the coordinates of radios into a conflict-free TDMA "
        "schedule (distance-2 graph colouring), verifies it independently, and prints the schedule "
        "with its Slot x Node Boolean matrix.",
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
        "--algorithm",
        choices=ALGORITHM_KEYS,
        default="dsatur",
        help="dsatur: DSATUR on the 2-hop conflict graph; greedy: first-fit in natural order on the same "
        "graph; naive-d1: first-fit on the radio graph only, UNSAFE, a demo of why distance-2 matters "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--export-json",
        metavar="PATH",
        help='also write the schedule as schedule.json (tdma-schedule/v1) to PATH; it is written even '
        'when verification fails, with "verified": false',
    )
    parser.add_argument(
        "--show-topology",
        action="store_true",
        help="print the topology summary (links, conflict pairs, hidden-terminal pairs) before the report",
    )
    return parser


def compute_schedule(G, H, algorithm):
    """Colour the graphs with the heuristic named by `algorithm` and build the canonical Schedule."""
    name, colouring = run_algorithm(algorithm, G, H)
    return make_schedule(colouring, list(G.nodes), name)


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

    schedule = compute_schedule(G, H, args.algorithm)
    verification = verify_schedule(G, schedule)

    if args.show_topology:
        print(format_topology_summary(topology_summary(G, H)))
        print()
    print(format_report(schedule, verification, G.graph["radio_range"]))

    if args.export_json:
        try:
            _write_text_file(args.export_json, to_json(schedule, G, verified=verification.ok))
        except OSError as exc:
            print(f"Error: cannot write {args.export_json}: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        print(f"Schedule exported to {args.export_json}", file=sys.stderr)

    # CONCEPT: CLI design (argparse, exit codes) - 0 = verified schedule, 1 = verification failed, 2 = bad input: a script or grader can rely on the code without reading the text
    return EXIT_OK if verification.ok else EXIT_VERIFICATION_FAILED


def _write_text_file(path, text):
    """Write UTF-8 text with LF line endings, creating missing parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")


def _make_output_ascii_safe():
    """Never crash on a console or pipe that cannot encode a character.

    A non-ASCII node name on a Windows cp1252 console would otherwise raise
    UnicodeEncodeError; with this it is printed as an escape sequence instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
