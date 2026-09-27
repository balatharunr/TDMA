"""Division 2 tests: the independent verifier (tdma_brain.verify)."""
import random

import networkx as nx
import pytest

from tdma_brain import topology
from tdma_brain.heuristics import dsatur
from tdma_brain.io_utils import natural_key
from tdma_brain.schedule import make_schedule
from tdma_brain.topology import build_radio_graph
from tdma_brain.verify import DIRECT, HIDDEN_TERMINAL, Violation, verify_schedule
from tests.helpers import (
    build_graphs,
    hops_by_brute_force,
    line_graph,
    neighbours_by_brute_force,
    random_layout,
    raw_schedule,
)

# Six radios on a line, each hearing only its direct neighbours. Slots 0,1,2,0,1,2 are a valid
# distance-2 colouring, and every slot keeps a radio in it after any single change below.
LINE = {"Node_1": 0, "Node_2": 1, "Node_3": 2, "Node_4": 0, "Node_5": 1, "Node_6": 2}


def verdict(schedule, G=None):
    return verify_schedule(G if G is not None else line_graph(), schedule)


# --------------------------------------------------------------------------- conflicts
def test_a_valid_schedule_passes():
    result = verdict(raw_schedule(LINE))
    assert result.ok and result.violations == [] and result.problems == []


def test_an_injected_1_hop_conflict_is_caught():
    result = verdict(raw_schedule({**LINE, "Node_6": 1}))  # Node_6 now shares Slot 1 with its neighbour Node_5
    assert not result.ok and result.problems == []
    assert result.violations == [Violation("Node_5", "Node_6", 1, DIRECT)]
    assert result.violations[0].describe() == "Node_5 & Node_6 share Slot 1 -> DIRECT (1-hop)"


def test_an_injected_2_hop_conflict_is_caught_with_the_correct_via():
    result = verdict(raw_schedule({**LINE, "Node_6": 0}))  # Node_4 and Node_6 cannot hear each other, both hear Node_5
    assert not result.ok and result.problems == []
    assert result.violations == [Violation("Node_4", "Node_6", 0, HIDDEN_TERMINAL, ("Node_5",))]
    assert result.violations[0].describe() == "Node_4 & Node_6 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_5"


def test_the_via_lists_every_common_neighbour_in_natural_order():
    G, H = build_graphs("grid16_300m")
    mapping = make_schedule(dsatur(H), list(G.nodes), "DSATUR").node_to_slot
    result = verify_schedule(G, raw_schedule({**mapping, "Node_03": mapping["Node_01"]}))
    hidden = Violation("Node_01", "Node_03", 0, HIDDEN_TERMINAL, ("Node_02", "Node_06"))
    assert hidden in result.violations and not result.ok
    assert hidden.describe() == "Node_01 & Node_03 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_02, Node_06"


def test_radios_three_hops_apart_may_share_a_slot():
    schedule = raw_schedule({"Node_1": 0, "Node_2": 1, "Node_3": 2, "Node_4": 0})  # Node_1 and Node_4: 3 hops apart
    assert verdict(schedule, line_graph(4)).ok


def test_violations_come_back_in_natural_order():
    result = verdict(raw_schedule({n: 0 for n in LINE}, k=1))  # everybody transmits at once
    pairs = [(v.u, v.v) for v in result.violations]
    assert pairs == sorted(pairs, key=lambda p: (natural_key(p[0]), natural_key(p[1])))
    assert len(pairs) == 5 + 4  # 5 direct links + 4 hidden-terminal pairs on a 6-radio line
    assert [v.kind for v in result.violations].count(DIRECT) == 5


# --------------------------------------------------------------------------- multi-slot radios (Division 3's max-reuse)
def test_radios_that_own_extra_slots_are_checked_slot_by_slot():
    G, H = build_graphs("ring16_r1150m")
    base = make_schedule(dsatur(H), list(G.nodes), "DSATUR").node_to_slot  # 0,1,2,0,1,2,...,3: Node_16 is alone in Slot 3
    assert base["Node_16"] == 3 and list(base.values()).count(3) == 1
    safe = raw_schedule(base, extra={"Node_04": [3], "Node_07": [3], "Node_10": [3], "Node_13": [3]})
    assert verify_schedule(G, safe).ok  # more than 2 hops from Node_16 and from each other: reuse is fine
    unsafe = raw_schedule(base, extra={"Node_02": [3]})  # Node_02 is 2 hops from Node_16, via Node_01
    result = verify_schedule(G, unsafe)
    assert result.violations == [Violation("Node_02", "Node_16", 3, HIDDEN_TERMINAL, ("Node_01",))]


# --------------------------------------------------------------------------- structure
def test_an_empty_slot_is_caught():
    result = verdict(raw_schedule(LINE, k=4))  # nobody transmits in Slot 3
    assert not result.ok and result.violations == []
    assert any("empty slot" in p and "3" in p for p in result.problems)
    gap = verdict(raw_schedule({"Node_1": 0, "Node_2": 2}, k=3), line_graph(2))  # Slot 1 in the middle is empty
    assert any("empty slot" in p and "1" in p for p in gap.problems)


def test_a_radio_without_a_slot_is_caught():
    schedule = raw_schedule(LINE)
    del schedule.node_to_slot["Node_3"]
    result = verdict(schedule)
    assert not result.ok and any("without a slot" in p and "Node_3" in p for p in result.problems)


def test_a_radio_missing_from_the_whole_schedule_is_caught():
    result = verdict(raw_schedule({n: s for n, s in LINE.items() if n != "Node_3"}))
    assert not result.ok
    assert any("missing from the schedule" in p and "Node_3" in p for p in result.problems)
    assert any("without a slot" in p and "Node_3" in p for p in result.problems)


def test_a_node_that_is_not_in_the_radio_graph_is_caught():
    result = verdict(raw_schedule({**LINE, "Node_9": 0}))
    assert not result.ok and any("not in the radio graph" in p and "Node_9" in p for p in result.problems)


def test_a_slot_outside_the_frame_is_caught():
    schedule = raw_schedule(LINE)
    schedule.node_to_slot["Node_6"] = 7
    result = verdict(schedule)
    assert not result.ok and any("outside the frame 0..2" in p and "Node_6 uses Slot 7" in p for p in result.problems)
    schedule.node_to_slot["Node_6"] = -1
    assert any("Slot -1" in p for p in verdict(schedule).problems)


@pytest.mark.parametrize("bad_k", [0, -3, 2.5, True, None])
def test_a_nonsense_frame_length_is_caught(bad_k):
    schedule = raw_schedule(LINE)
    schedule.frame_length = bad_k
    result = verdict(schedule)
    assert not result.ok and any("frame_length" in p for p in result.problems)


def test_duplicate_and_unsorted_node_lists_are_caught():
    schedule = raw_schedule(LINE)
    schedule.nodes.append("Node_1")
    assert any("more than once" in p and "Node_1" in p for p in verdict(schedule).problems)
    reordered = raw_schedule(LINE)
    reordered.nodes.reverse()
    assert any("natural order" in p for p in verdict(reordered).problems)


def test_matrix_problems_are_caught():
    flipped = raw_schedule(LINE)
    flipped.matrix[0][1] = 1  # claims Node_2 transmits in Slot 0 as well
    assert any("matrix[0][1] is 1 but Node_2 does not transmit in Slot 0" in p for p in verdict(flipped).problems)

    cleared = raw_schedule(LINE)
    cleared.matrix[0][0] = 0  # claims Node_1 does not transmit in its own slot
    problems = verdict(cleared).problems
    assert any("matrix[0][0] is 0 but Node_1 transmits in Slot 0" in p for p in problems)

    silent = raw_schedule(LINE)
    for row in silent.matrix:
        row[5] = 0  # Node_6's column is all zeros
    assert any("column(s) with no 1" in p and "Node_6" in p for p in verdict(silent).problems)

    short = raw_schedule(LINE)
    short.matrix.pop()
    assert any("matrix has 2 rows but the frame has 3" in p for p in verdict(short).problems)

    narrow = raw_schedule(LINE)
    narrow.matrix[1] = narrow.matrix[1][:-1]
    assert any("do not have one column per node" in p for p in verdict(narrow).problems)


def test_slot_to_nodes_problems_are_caught():
    swapped = raw_schedule(LINE)
    swapped.slot_to_nodes[0] = ["Node_1"]  # Node_4 is missing from Slot 0's list
    assert any("slot_to_nodes[0]" in p for p in verdict(swapped).problems)
    short = raw_schedule(LINE)
    short.slot_to_nodes.pop()
    assert any("slot_to_nodes has 2 entries" in p for p in verdict(short).problems)


def test_a_flood_of_problems_stays_readable():
    schedule = raw_schedule({f"Node_{i}": 0 for i in range(1, 31)}, k=1)
    G = line_graph(60)  # 30 radios missing from the schedule
    (problem,) = [p for p in verify_schedule(G, schedule).problems if "missing from the schedule" in p]
    assert "and 22 more" in problem and len(problem) < 200


# --------------------------------------------------------------------------- independence and properties
def test_the_verifier_never_touches_the_conflict_graph(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the verifier must not build or use the conflict graph G^2")

    monkeypatch.setattr(nx, "power", forbidden)
    monkeypatch.setattr(topology, "build_conflict_graph", forbidden)
    monkeypatch.setattr(topology, "hidden_terminal_pairs", forbidden)
    assert verdict(raw_schedule(LINE)).ok
    assert not verdict(raw_schedule({**LINE, "Node_6": 0})).ok


def test_the_verifier_agrees_with_brute_force_on_random_slot_assignments():
    """200 seeded layouts with random (mostly bad) slots: same conflicts, same kinds, same via."""
    for seed in range(200):
        coords, radio_range = random_layout(seed)
        G = build_radio_graph(coords, radio_range)
        rng = random.Random(seed)
        slots = rng.randint(1, max(1, len(coords) // 2))
        node_to_slot = {n: rng.randrange(slots) for n in coords}

        heard = neighbours_by_brute_force(coords, radio_range)
        expected = {}
        for pair, hops in hops_by_brute_force(coords, radio_range).items():
            u, v = sorted(pair, key=natural_key)
            if node_to_slot[u] == node_to_slot[v]:
                via = () if hops == 1 else tuple(sorted(heard[u] & heard[v], key=natural_key))
                expected[(u, v, node_to_slot[u])] = (DIRECT if hops == 1 else HIDDEN_TERMINAL, via)

        result = verify_schedule(G, raw_schedule(node_to_slot))
        found = {(x.u, x.v, x.slot): (x.kind, x.via) for x in result.violations}
        assert found == expected, f"seed {seed}"
        assert len(result.violations) == len(found)  # nothing reported twice
        assert all(natural_key(x.u) < natural_key(x.v) for x in result.violations)
