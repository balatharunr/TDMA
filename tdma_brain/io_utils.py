"""Input handling for the TDMA planner (created in Division 1).

Turns the user's JSON (a command-line string or a file) into a plain {name: (x, y)} dict,
or raises InputError with a message that names the bad node. Also provides natural sorting,
so 'Node_2' comes before 'Node_10' everywhere in the project.
Later divisions add the export helpers (D2, D4) to this module.
"""
import json
import math
import re
import sys
from pathlib import Path

EXPECTED_NODE_COUNT = 16  # the brief's size; any other size only triggers a warning


class InputError(Exception):
    """The user's input is missing or invalid. The message is written to be shown as-is."""


_DIGIT_RUNS = re.compile(r"([0-9]+)")


def natural_key(name):
    """Sort key for natural order: 'Node_2' < 'Node_10' (digit runs compare as numbers).

    str(name) comes first because D3's solver tests use networkx graphs whose nodes are
    integers. Splitting on digit runs always gives text, number, text, number, ..., so two
    keys never compare a str with an int. The name itself is the last tie-breaker, which
    keeps the order independent of the input order even for 'Node_01' versus 'Node_1'.
    """
    text = str(name)
    parts = _DIGIT_RUNS.split(text)
    return ([int(part) if i % 2 else part for i, part in enumerate(parts)], text)


def parse_coordinates(text: str) -> dict[str, tuple[float, float]]:
    """Parse '{"Node_01": [x, y], ...}' into {name: (x, y)}, keeping the JSON order.

    Each value must be a list of exactly two numbers (int or float, negative is fine).
    Booleans, strings, null, NaN and Infinity are rejected, and so are an empty object and
    duplicate node names. Any problem raises InputError naming the bad node.
    """
    try:
        # CONCEPT: JSON parsing & input validation - object_pairs_hook sees every key/value pair, so a repeated node name is caught (plain json.loads silently keeps the last one)
        data = json.loads(text, object_pairs_hook=_reject_duplicate_names)
    except InputError:
        raise
    except json.JSONDecodeError as exc:
        raise InputError(
            f"input is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}). "
            "If you typed it on a command line, the shell may have removed the double "
            "quotes: escape them or use --file PATH instead."
        ) from exc
    except RecursionError:
        raise InputError("input is nested too deeply to be a list of coordinates") from None
    except ValueError as exc:  # for example an integer with thousands of digits
        raise InputError(f"input could not be parsed: {exc}") from exc

    if not isinstance(data, dict):
        raise InputError(
            'input must be a JSON object such as {"Node_01": [0.0, 0.0], ...}, '
            f"got {_describe(data)}"
        )
    if not data:
        raise InputError("input has no nodes (the JSON object is empty)")
    return {name: _parse_point(name, value) for name, value in data.items()}


def load_input(args) -> dict[str, tuple[float, float]]:
    """Read the coordinates from the positional JSON string or from --file (exactly one).

    `args` is the argparse namespace: `args.coords_json` (str or None) and `args.file`
    (path or None). Warns on stderr when the node count is not 16, then continues.
    """
    coords_json = getattr(args, "coords_json", None)
    file_path = getattr(args, "file", None)
    if coords_json is not None and file_path is not None:
        raise InputError("give the coordinates either as a JSON string or with --file PATH, not both")
    if coords_json is None and file_path is None:
        raise InputError("no input: pass the coordinates as a JSON string or with --file PATH (see --help)")

    if file_path is not None:
        try:
            coords = parse_coordinates(_read_text_file(file_path))
        except InputError as exc:
            raise InputError(f"{file_path}: {exc}") from exc
    else:
        coords = parse_coordinates(coords_json)

    if len(coords) != EXPECTED_NODE_COUNT:
        print(
            f"Warning: {len(coords)} node(s) given, the brief uses {EXPECTED_NODE_COUNT}; continuing anyway.",
            file=sys.stderr,
        )
    return coords


def _read_text_file(path) -> str:
    """Read a UTF-8 text file (a BOM, which Windows tools like to add, is tolerated)."""
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise InputError(f"file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc


def _reject_duplicate_names(pairs):
    """object_pairs_hook for json.loads: build the dict, but refuse a repeated key."""
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise InputError(f"duplicate node name {key!r}: every node must appear exactly once")
        seen.add(key)
    return dict(pairs)


def _parse_point(name: str, value) -> tuple[float, float]:
    """Validate one node's value: a list of exactly two finite numbers."""
    if not isinstance(value, list) or len(value) != 2:
        raise InputError(f"{name}: expected [x, y] (a list of exactly 2 numbers), got {_describe(value)}")
    return (_parse_number(name, "x", value[0]), _parse_number(name, "y", value[1]))


def _parse_number(name: str, axis: str, value) -> float:
    """Validate one coordinate and return it as a float."""
    # CONCEPT: JSON parsing & input validation - in Python bool is a kind of int, so JSON true/false must be rejected by hand
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name}: {axis} coordinate must be a number, got {_describe(value)}")
    try:
        number = float(value)
    except OverflowError:  # an integer far too large for a float
        raise InputError(f"{name}: {axis} coordinate is too large") from None
    # CONCEPT: JSON parsing & input validation - Python's json accepts NaN and Infinity (and 1e999 becomes inf), so finiteness is checked here
    if not math.isfinite(number):
        raise InputError(f"{name}: {axis} coordinate must be a finite number, got {_describe(value)}")
    return number


def _describe(value) -> str:
    """Short description of a parsed JSON value, for error messages (ASCII only)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"boolean {str(value).lower()}"
    if isinstance(value, str):
        return f"string {_clip(json.dumps(value))}"
    if isinstance(value, (int, float)):
        return f"number {_clip(repr(value))}"
    if isinstance(value, list):
        return f"a list of {len(value)} item(s)"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _clip(text: str, limit: int = 40) -> str:
    """Shorten long text so one bad value cannot flood the terminal."""
    return text if len(text) <= limit else text[: limit - 3] + "..."
