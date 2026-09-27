"""Division 2 tests: the Schedule model, canonical numbering and the schedule.json contract.

Division 4 adds the tests for to_csv() (the Boolean matrix as CSV) near the end of this file.
"""
import csv
import io
import json
import random

import pytest

from tdma_brain import __version__
from tdma_brain.heuristics import dsatur
from tdma_brain.io_utils import InputError, natural_key
from tdma_brain.schedule import SCHEMA, Schedule, canonicalize, make_schedule, to_csv, to_json
from tdma_brain.topology import build_radio_graph
from tests.golden import GOLDEN
from tests.helpers import build_graphs, load_example


def dsatur_schedule(name):
    G, H = build_graphs(name)
    return G, make_schedule(dsatur(H), list(G.nodes), "DSATUR")


# --------------------------------------------------------------------------- canonicalize
def test_canonical_numbering_follows_first_appearance_in_natural_order():
    nodes = ["Node_01", "Node_02", "Node_03", "Node_04", "Node_05"]
    raw = {"Node_01": 7, "Node_02": 3, "Node_03": 7, "Node_04": 9, "Node_05": 3}
    assert canonicalize(raw, nodes) == {"Node_01": 0, "Node_02": 1, "Node_03": 0, "Node_04": 2, "Node_05": 1}


def test_canonicalize_walks_natural_order_whatever_order_the_nodes_come_in():
    raw = {"Node_10": "x", "Node_2": "y", "Node_1": "z"}
    expected = {"Node_1": 0, "Node_2": 1, "Node_10": 2}  # not lexicographic: Node_10 comes after Node_2
    assert canonicalize(raw, ["Node_10", "Node_2", "Node_1"]) == expected
    assert canonicalize(raw, ["Node_1", "Node_2", "Node_10"]) == expected


def test_canonicalize_ignores_how_the_colours_happened_to_be_named():
    rng = random.Random(3)
    nodes = [f"Node_{i:02d}" for i in range(1, 21)]
    for _ in range(50):
        raw = {n: rng.randint(0, 6) for n in nodes}
        renamed_colours = dict(zip(range(7), rng.sample(range(100, 200), 7)))
        renamed = {n: renamed_colours[c] for n, c in raw.items()}
        canonical = canonicalize(raw, nodes)
        assert canonicalize(renamed, nodes) == canonical
        # who shares a slot with whom is untouched ...
        assert all((canonical[a] == canonical[b]) == (raw[a] == raw[b]) for a in nodes for b in nodes)
        # ... the slots are exactly 0..K-1 and the first node is in slot 0
        assert set(canonical.values()) == set(range(len(set(raw.values()))))
        assert canonical["Node_01"] == 0


def test_canonicalize_needs_a_slot_for_every_node():
    with pytest.raises(ValueError, match="Node_02"):
        canonicalize({"Node_01": 0}, ["Node_01", "Node_02"])


# --------------------------------------------------------------------------- make_schedule
def test_make_schedule_builds_every_view_of_the_same_assignment():
    raw = {"Node_03": 5, "Node_01": 2, "Node_02": 9, "Node_04": 2}
    schedule = make_schedule(raw, ["Node_03", "Node_01", "Node_02", "Node_04"], "test")
    assert schedule.nodes == ["Node_01", "Node_02", "Node_03", "Node_04"]  # natural order, whatever came in
    assert schedule.node_to_slot == {"Node_01": 0, "Node_02": 1, "Node_03": 2, "Node_04": 0}
    assert schedule.frame_length == 3
    assert schedule.slot_to_nodes == [["Node_01", "Node_04"], ["Node_02"], ["Node_03"]]
    assert schedule.matrix == [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
    assert schedule.algorithm == "test" and schedule.extra_tx_slots == {}


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_matrix_and_slot_lists_agree_with_node_to_slot(golden):
    _, schedule = dsatur_schedule(golden.name)
    k, n = schedule.frame_length, len(schedule.nodes)
    assert (len(schedule.matrix), {len(row) for row in schedule.matrix}) == (k, {n})
    for j, node in enumerate(schedule.nodes):
        column = [row[j] for row in schedule.matrix]
        assert column.count(1) == 1 and column[schedule.node_to_slot[node]] == 1  # exactly one slot per radio
    for s, group in enumerate(schedule.slot_to_nodes):
        assert group and group == sorted(group, key=natural_key)  # no empty slot; natural order inside a slot
        assert group == [nd for nd in schedule.nodes if schedule.matrix[s][schedule.nodes.index(nd)] == 1]
    assert sum(schedule.occupancy()) == n
    assert schedule.node_to_slot[schedule.nodes[0]] == 0  # Node_01 (or Node_A) is always Slot 0


def test_the_official_grid_schedule_matches_the_guides_metrics():
    _, schedule = dsatur_schedule("grid16_300m")
    assert schedule.frame_length == 9
    assert schedule.occupancy() == [4, 2, 2, 2, 1, 1, 2, 1, 1]  # slot 0 holds the four corners
    assert schedule.reuse_factor == pytest.approx(16 / 9)
    assert f"{schedule.reuse_factor:.2f}" == "1.78"


def test_reuse_factor_extremes():
    _, all_share = dsatur_schedule("isolated16_1000m")  # nobody hears anybody: one slot for all
    assert (all_share.frame_length, all_share.reuse_factor) == (1, 16.0)
    _, none_share = dsatur_schedule("cluster16_30m")  # everyone hears everybody: plain TDMA
    assert (none_share.frame_length, none_share.reuse_factor) == (16, 1.0)
    empty = Schedule(nodes=[], node_to_slot={}, frame_length=0, slot_to_nodes=[], matrix=[], algorithm="x")
    assert empty.reuse_factor == 0.0  # no ZeroDivisionError on a degenerate schedule


def test_tx_slots_include_extra_slots():
    schedule = make_schedule({"Node_1": 0, "Node_2": 1}, ["Node_1", "Node_2"], "test")
    assert schedule.tx_slots("Node_1") == {0}
    schedule.extra_tx_slots = {"Node_1": [1]}
    assert schedule.tx_slots("Node_1") == {0, 1} and schedule.tx_slots("Node_2") == {1}


# --------------------------------------------------------------------------- schedule.json
CONTRACT_KEYS = [  # guide section 4.4, in this order
    "schema", "generator", "radio_range_m", "nodes", "coordinates", "links", "frame_length",
    "node_to_slot", "slot_to_nodes", "extra_tx_slots", "matrix", "lower_bound", "optimality",
    "algorithm", "verified",
]


def test_json_has_exactly_the_contract_fields_in_order():
    G, schedule = dsatur_schedule("grid16_300m")
    text = to_json(schedule, G, verified=True)
    data = json.loads(text)
    assert list(data) == CONTRACT_KEYS
    assert data["schema"] == SCHEMA == "tdma-schedule/v1"
    assert data["generator"] == f"tdma_brain {__version__}"
    assert data["radio_range_m"] == 500.0
    assert data["nodes"] == [f"Node_{i:02d}" for i in range(1, 17)]
    assert data["frame_length"] == 9 and data["algorithm"] == "DSATUR" and data["verified"] is True
    assert data["lower_bound"] is None and data["optimality"] is None and data["extra_tx_slots"] == {}
    assert data["node_to_slot"]["Node_01"] == 0 and data["node_to_slot"]["Node_06"] == 4
    assert data["slot_to_nodes"][0] == ["Node_01", "Node_04", "Node_13", "Node_16"]  # guide, Division 7 expected XML
    assert data["matrix"][0][:4] == [1, 0, 0, 1]
    assert text.endswith("}\n") and "\n    [1, 0, 0, 1," in text  # matrix: one row per line


def test_json_carries_the_coordinates_and_links_part_2_needs():
    G, schedule = dsatur_schedule("grid16_300m")
    data = json.loads(to_json(schedule, G, verified=True))
    coords = load_example("grid16_300m")
    assert data["coordinates"] == {name: list(xy) for name, xy in coords.items()}
    assert len(data["links"]) == 42
    assert data["links"] == sorted(data["links"], key=lambda p: (natural_key(p[0]), natural_key(p[1])))
    assert all(natural_key(u) < natural_key(v) for u, v in data["links"])
    assert data["links"][0] == ["Node_01", "Node_02"]
    assert {frozenset(link) for link in data["links"]} == {frozenset(edge) for edge in G.edges}


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_json_round_trips_and_is_self_contained(golden):
    G, schedule = dsatur_schedule(golden.name)
    text = to_json(schedule, G, verified=True)
    assert Schedule.from_json(text) == schedule
    # the file alone is enough to rebuild the radio graph that the schedule was made for
    data = json.loads(text)
    rebuilt = build_radio_graph({n: tuple(xy) for n, xy in data["coordinates"].items()}, data["radio_range_m"])
    assert {frozenset(e) for e in rebuilt.edges} == {frozenset(link) for link in data["links"]}


def test_json_records_the_verdict_and_the_optional_d3_fields():
    G, schedule = dsatur_schedule("hidden_terminal3")
    failed = json.loads(to_json(schedule, G, verified=False))
    assert failed["verified"] is False
    proven = json.loads(to_json(schedule, G, verified=True, lower_bound=3, optimality="PROVEN_OPTIMAL_BY_BOUND"))
    assert (proven["lower_bound"], proven["optimality"]) == (3, "PROVEN_OPTIMAL_BY_BOUND")


def test_json_is_ascii_even_for_non_ascii_node_names():
    G = build_radio_graph({"Nöde_01": (0.0, 0.0)})
    schedule = make_schedule({"Nöde_01": 0}, ["Nöde_01"], "test")
    text = to_json(schedule, G, verified=True)
    text.encode("ascii")
    assert Schedule.from_json(text) == schedule


# --------------------------------------------------------------------------- Division 4: to_csv
def test_csv_header_is_slot_then_every_node_name_in_order():
    _, schedule = dsatur_schedule("grid16_300m")
    header = to_csv(schedule).splitlines()[0]
    assert header == "Slot," + ",".join(f"Node_{i:02d}" for i in range(1, 17))


def test_csv_rows_match_the_matrix_exactly():
    _, schedule = dsatur_schedule("grid16_300m")
    lines = to_csv(schedule).splitlines()
    assert len(lines) == 1 + schedule.frame_length  # header + one row per slot
    for slot, row in enumerate(schedule.matrix):
        assert lines[1 + slot] == f"Slot {slot:02d}," + ",".join(str(cell) for cell in row)
    assert lines[1] == "Slot 00,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1"  # slot 0: the four corners of the official grid


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_csv_has_one_row_per_slot_for_every_example(golden):
    _, schedule = dsatur_schedule(golden.name)
    rows = list(csv.reader(io.StringIO(to_csv(schedule))))
    assert len(rows) == 1 + schedule.frame_length
    assert rows[0] == ["Slot", *schedule.nodes]
    for slot, row in enumerate(rows[1:]):
        assert row[0] == f"Slot {slot:02d}"
        assert [int(cell) for cell in row[1:]] == schedule.matrix[slot]


def test_csv_uses_lf_line_endings_and_ends_with_one():
    _, schedule = dsatur_schedule("hidden_terminal3")
    text = to_csv(schedule)
    assert "\r" not in text and text.endswith("\n") and not text.endswith("\n\n")


def test_csv_quotes_a_node_name_that_contains_a_comma():
    # the input validator allows any node name; a name with a comma must not corrupt the CSV columns
    schedule = make_schedule({"Node,1": 0, "Node_2": 1}, ["Node,1", "Node_2"], "test")
    rows = list(csv.reader(io.StringIO(to_csv(schedule))))
    assert rows[0] == ["Slot", "Node,1", "Node_2"]  # the comma survived as data, not a new column
    assert len(rows[0]) == 3


def test_csv_round_trips_into_the_same_matrix():
    _, schedule = dsatur_schedule("grid16_400m")
    rows = list(csv.reader(io.StringIO(to_csv(schedule))))
    assert rows[0][1:] == schedule.nodes
    matrix = [[int(cell) for cell in row[1:]] for row in rows[1:]]
    assert matrix == schedule.matrix


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "not a schedule file"),
        ('{"schema": "something-else/v9"}', "not a schedule file"),
        (json.dumps({"schema": SCHEMA}), "missing or has a malformed field"),
        (json.dumps({"schema": SCHEMA, "nodes": ["Node_1"], "node_to_slot": ["oops"], "frame_length": 1,
                     "slot_to_nodes": [["Node_1"]], "matrix": [[1]], "algorithm": "x"}), "malformed"),
    ],
    ids=["not json", "not an object", "wrong schema", "missing fields", "wrong type"],
)
def test_from_json_rejects_bad_files_with_a_clear_error(text, fragment):
    with pytest.raises(InputError, match=fragment):
        Schedule.from_json(text)
