"""Division 1 tests: input parsing and validation (tdma_brain.io_utils)."""
import argparse
import itertools

import networkx as nx
import pytest

from tdma_brain.io_utils import InputError, load_input, natural_key, parse_coordinates
from tests.golden import GOLDEN, example_path


def namespace(coords_json=None, file=None):
    """Stand-in for the argparse namespace that load_input() receives."""
    return argparse.Namespace(coords_json=coords_json, file=file)


# --------------------------------------------------------------------------- parse_coordinates
def test_accepts_ints_floats_and_negatives():
    text = '{"Node_01": [0, 0], "Node_02": [300.5, -20], "Node_03": [-1e3, 2.5E2]}'
    coords = parse_coordinates(text)
    assert coords == {"Node_01": (0.0, 0.0), "Node_02": (300.5, -20.0), "Node_03": (-1000.0, 250.0)}
    assert list(coords) == ["Node_01", "Node_02", "Node_03"]  # JSON order is kept
    assert all(isinstance(xy, tuple) for xy in coords.values())
    assert all(isinstance(v, float) for xy in coords.values() for v in xy)  # ints become floats


def test_accepts_non_numeric_node_names_and_a_single_node():
    assert parse_coordinates('{"Node_A": [0.0, 0.0]}') == {"Node_A": (0.0, 0.0)}


BAD_INPUTS = [
    # (id, text, fragments that must all appear in the error message)
    ("empty string", "", ["not valid JSON"]),
    ("not json", "not json", ["not valid JSON"]),
    ("truncated json", '{"Node_01": [0, 0]', ["not valid JSON"]),
    ("single quotes (shell ate the double quotes)", "{'Node_01': [0, 0]}", ["not valid JSON", "--file"]),
    ("json list", "[[0, 0], [1, 1]]", ["JSON object", "list of 2"]),
    ("json number", "5", ["JSON object"]),
    ("json null", "null", ["JSON object", "null"]),
    ("json string", '"Node_01"', ["JSON object"]),
    ("empty object", "{}", ["no nodes"]),
    ("value too short", '{"Node_01": [1]}', ["Node_01", "exactly 2"]),
    ("value too long", '{"Node_01": [1, 2, 3]}', ["Node_01", "exactly 2"]),
    ("value empty list", '{"Node_01": []}', ["Node_01", "exactly 2"]),
    ("value is a number", '{"Node_01": 5}', ["Node_01", "exactly 2"]),
    ("value is a string", '{"Node_01": "0,0"}', ["Node_01", "exactly 2"]),
    ("value is null", '{"Node_01": null}', ["Node_01", "null"]),
    ("value is an object", '{"Node_01": {"x": 1, "y": 2}}', ["Node_01", "object"]),
    ("string coordinate", '{"Node_01": [0, "x"]}', ["Node_01", "y coordinate", "string"]),
    ("numeric string is not a number", '{"Node_01": ["1", 2]}', ["Node_01", "x coordinate", "string"]),
    ("null coordinate", '{"Node_01": [null, 2]}', ["Node_01", "x coordinate", "null"]),
    ("list coordinate", '{"Node_01": [[1], 2]}', ["Node_01", "x coordinate"]),
    ("true is not a number", '{"Node_01": [true, 2]}', ["Node_01", "x coordinate", "boolean"]),
    ("false is not a number", '{"Node_01": [0, false]}', ["Node_01", "y coordinate", "boolean"]),
    ("NaN", '{"Node_01": [NaN, 0]}', ["Node_01", "finite"]),
    ("Infinity", '{"Node_01": [0, Infinity]}', ["Node_01", "finite"]),
    ("-Infinity", '{"Node_01": [-Infinity, 0]}', ["Node_01", "finite"]),
    ("float overflow 1e999", '{"Node_01": [1e999, 0]}', ["Node_01", "finite"]),
    ("integer too large for a float", '{"Node_01": [1' + "0" * 400 + ", 0]}", ["Node_01", "too large"]),
    ("duplicate node name", '{"Node_01": [0, 0], "Node_01": [5, 5]}', ["duplicate", "Node_01"]),
    ("duplicate with identical values", '{"Node_02": [0, 0], "Node_01": [1, 1], "Node_02": [0, 0]}', ["duplicate", "Node_02"]),
    ("absurdly deep nesting", "[" * 100_000, ["nested too deeply"]),
]


@pytest.mark.parametrize("text, fragments", [b[1:] for b in BAD_INPUTS], ids=[b[0] for b in BAD_INPUTS])
def test_invalid_input_raises_input_error_naming_the_problem(text, fragments):
    with pytest.raises(InputError) as info:
        parse_coordinates(text)
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} missing from: {message}"


def test_error_names_only_the_bad_node():
    with pytest.raises(InputError) as info:
        parse_coordinates('{"Node_01": [0, 0], "Node_02": [1], "Node_03": [2, 2]}')
    message = str(info.value)
    assert "Node_02" in message
    assert "Node_01" not in message and "Node_03" not in message


def test_error_message_is_one_readable_line():
    with pytest.raises(InputError) as info:
        parse_coordinates('{"Node_01": [0, "' + "x" * 500 + '"]}')
    assert "\n" not in str(info.value)
    assert len(str(info.value)) < 200  # the long bad value is clipped, not echoed in full


# --------------------------------------------------------------------------- natural_key
def test_natural_order_puts_2_before_10():
    assert sorted(["Node_10", "Node_2", "Node_1"], key=natural_key) == ["Node_1", "Node_2", "Node_10"]
    assert natural_key("Node_2") < natural_key("Node_10")
    assert natural_key("Node_A") < natural_key("Node_B") < natural_key("Node_C")


def test_natural_key_accepts_integer_nodes():
    # D3's solver tests use networkx graphs whose nodes are ints; a str-only key would crash
    assert sorted([10, 2, 33, 1], key=natural_key) == [1, 2, 10, 33]
    graph = nx.mycielski_graph(4)
    assert sorted(graph.nodes, key=natural_key) == sorted(graph.nodes)


def test_natural_key_never_compares_text_with_numbers():
    labels = ["Node_A", 3, "Node_10", "x2y", "7up", "", "-5", "Node_", "N1_2_3"]
    for order in itertools.permutations(labels[:6]):
        sorted(order, key=natural_key)  # would raise TypeError if a str met an int
    assert sorted(labels, key=natural_key) == sorted(reversed(labels), key=natural_key)


def test_natural_order_is_independent_of_input_order_even_for_ties():
    tied = ["Node_1", "Node_01", "Node_001"]  # all "number 1": the name breaks the tie
    expected = sorted(tied, key=natural_key)
    for order in itertools.permutations(tied):
        assert sorted(order, key=natural_key) == expected


# --------------------------------------------------------------------------- load_input
def test_load_input_from_json_string():
    assert load_input(namespace(coords_json='{"Node_01": [1, 2]}')) == {"Node_01": (1.0, 2.0)}


def test_load_input_from_file(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text('{"Node_01": [1, 2], "Node_02": [3, 4]}', encoding="utf-8")
    assert load_input(namespace(file=str(path))) == {"Node_01": (1.0, 2.0), "Node_02": (3.0, 4.0)}


def test_load_input_tolerates_a_utf8_bom(tmp_path):
    # Windows PowerShell 5.1 and Notepad like to write a BOM at the start of UTF-8 files
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"Node_01": [1, 2]}')
    assert load_input(namespace(file=str(path))) == {"Node_01": (1.0, 2.0)}


def test_load_input_rejects_both_sources(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text('{"Node_01": [1, 2]}', encoding="utf-8")
    with pytest.raises(InputError, match="not both"):
        load_input(namespace(coords_json='{"Node_01": [1, 2]}', file=str(path)))


def test_load_input_rejects_no_source():
    with pytest.raises(InputError, match="no input"):
        load_input(namespace())


def test_load_input_empty_string_argument_is_bad_json_not_missing_input():
    with pytest.raises(InputError, match="not valid JSON"):
        load_input(namespace(coords_json=""))


def test_load_input_missing_file(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(InputError, match="file not found") as info:
        load_input(namespace(file=str(missing)))
    assert str(missing) in str(info.value)


def test_load_input_directory_instead_of_file(tmp_path):
    with pytest.raises(InputError, match="cannot read"):
        load_input(namespace(file=str(tmp_path)))


def test_load_input_binary_file(tmp_path):
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    with pytest.raises(InputError, match="cannot read"):
        load_input(namespace(file=str(path)))


def test_load_input_names_the_file_and_the_bad_node(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"Node_01": [0, 0], "Node_02": [1, "x"]}', encoding="utf-8")
    with pytest.raises(InputError) as info:
        load_input(namespace(file=str(path)))
    assert str(path) in str(info.value) and "Node_02" in str(info.value)


def test_load_input_warns_when_node_count_is_not_16(capsys):
    load_input(namespace(coords_json='{"Node_01": [0, 0], "Node_02": [5, 5]}'))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Warning" in captured.err and "2 node" in captured.err and "16" in captured.err


def test_load_input_is_silent_for_16_nodes(capsys):
    load_input(namespace(file=str(example_path("grid16_300m"))))
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- the example files
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_every_example_file_parses(golden):
    coords = load_input(namespace(file=str(example_path(golden.name))))
    assert len(coords) == golden.nodes
    assert list(coords) == sorted(coords, key=natural_key)  # our files are written in natural order


def test_official_16_node_input_is_the_300m_grid():
    coords = parse_coordinates(example_path("grid16_300m").read_text(encoding="utf-8"))
    assert list(coords) == [f"Node_{i:02d}" for i in range(1, 17)]
    assert coords["Node_01"] == (0.0, 0.0)
    assert coords["Node_02"] == (300.0, 0.0)  # the brief's second node
    assert coords["Node_16"] == (900.0, 900.0)  # the brief's last node
