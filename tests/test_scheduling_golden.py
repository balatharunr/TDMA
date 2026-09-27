"""Division 2 tests: scheduling the golden examples end to end (heuristics + schedule + verifier)."""
import pytest

from tdma_brain.heuristics import DSATUR_NAME, dsatur, greedy_first_fit, naive_distance1
from tdma_brain.io_utils import natural_key
from tdma_brain.schedule import make_schedule
from tdma_brain.verify import HIDDEN_TERMINAL, verify_schedule
from tests.golden import GOLDEN
from tests.helpers import build_graphs, conflicts_by_brute_force, load_example, random_layout
from tdma_brain.topology import build_conflict_graph, build_radio_graph

# guide section 5.3: DSATUR on grid16_300m, after canonical numbering
GUIDE_DSATUR_MAPPING = dict(zip(
    [f"Node_{i:02d}" for i in range(1, 17)],
    [0, 1, 2, 0, 3, 4, 5, 3, 6, 7, 8, 6, 0, 1, 2, 0],
))


def schedule_for(name, algorithm="dsatur"):
    G, H = build_graphs(name)
    nodes = list(G.nodes)
    if algorithm == "dsatur":
        return G, H, make_schedule(dsatur(H), nodes, DSATUR_NAME)
    if algorithm == "greedy":
        return G, H, make_schedule(greedy_first_fit(H, sorted(nodes, key=natural_key)), nodes, "greedy")
    return G, H, make_schedule(naive_distance1(G), nodes, "naive-d1")


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_dsatur_schedule_is_verified_and_hits_the_golden_optimum(golden):
    G, _, schedule = schedule_for(golden.name)
    result = verify_schedule(G, schedule)
    assert result.ok and result.violations == [] and result.problems == []
    assert schedule.frame_length == golden.optimal_slots  # and 5.1 says no schedule can be shorter


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_dsatur_is_never_worse_than_greedy_first_fit(golden):
    _, _, dsatur_schedule = schedule_for(golden.name)
    _, _, greedy_schedule = schedule_for(golden.name, "greedy")
    assert dsatur_schedule.frame_length <= greedy_schedule.frame_length
    assert greedy_schedule.frame_length >= golden.optimal_slots


def test_on_grid_400m_greedy_needs_6_slots_and_dsatur_5():
    assert schedule_for("grid16_400m", "greedy")[2].frame_length == 6
    assert schedule_for("grid16_400m")[2].frame_length == 5


def test_dsatur_mapping_on_the_official_grid_equals_the_guide():
    _, _, schedule = schedule_for("grid16_300m")
    assert schedule.node_to_slot == GUIDE_DSATUR_MAPPING
    assert schedule.slot_to_nodes[0] == ["Node_01", "Node_04", "Node_13", "Node_16"]  # the four corners share Slot 0


def test_the_four_centre_radios_of_the_official_grid_are_alone_in_their_slots():
    # each of Node_06/07/10/11 is within 2 hops of all 15 others, so in ANY valid schedule it is alone
    _, _, schedule = schedule_for("grid16_300m")
    for node in ("Node_06", "Node_07", "Node_10", "Node_11"):
        assert schedule.slot_to_nodes[schedule.node_to_slot[node]] == [node]


def test_the_official_dsatur_schedule_is_maximal():
    # no radio can be added to any slot without creating a conflict
    G, H, schedule = schedule_for("grid16_300m")
    for slot, members in enumerate(schedule.slot_to_nodes):
        for node in schedule.nodes:
            if node not in members:
                assert any(H.has_edge(node, member) for member in members), f"{node} could join Slot {slot}"


def test_naive_d1_on_the_official_grid_is_wrong_in_the_way_the_guide_predicts():
    G, _, schedule = schedule_for("grid16_300m", "naive-d1")
    result = verify_schedule(G, schedule)
    assert schedule.frame_length == 4  # a short frame ... (5.3)
    assert not result.ok and len(result.violations) == 24  # ... that collides 24 times
    assert {v.kind for v in result.violations} == {HIDDEN_TERMINAL}  # every one of them a hidden terminal
    assert result.problems == []  # the schedule is well formed, just not safe
    assert result.violations[0].describe() == "Node_01 & Node_03 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_02, Node_06"


def test_naive_d1_on_grid_400m_and_the_examples_where_it_happens_to_be_safe():
    G, _, schedule = schedule_for("grid16_400m", "naive-d1")
    result = verify_schedule(G, schedule)
    assert (schedule.frame_length, len(result.violations)) == (2, 34)
    for name in ("isolated16_1000m", "cluster16_30m", "two_clusters16"):  # no hidden-terminal pairs at all
        G, _, schedule = schedule_for(name, "naive-d1")
        assert verify_schedule(G, schedule).ok, name


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_naive_d1_violations_are_exactly_the_hidden_terminal_pairs_that_share_a_slot(golden):
    G, H, schedule = schedule_for(golden.name, "naive-d1")
    result = verify_schedule(G, schedule)
    shared = {frozenset((u, v)) for u, v, d in H.edges(data=True) if d["kind"] == "2-hop"
              and schedule.node_to_slot[u] == schedule.node_to_slot[v]}
    assert {frozenset((v.u, v.v)) for v in result.violations} == shared
    assert len(result.violations) == len(shared)


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_schedules_are_collision_free_by_a_third_independent_check(golden):
    # no networkx, no BFS, no H: plain distances and set intersections, straight from the definition
    coords = load_example(golden.name)
    for algorithm in ("dsatur", "greedy"):
        _, _, schedule = schedule_for(golden.name, algorithm)
        assert conflicts_by_brute_force(coords, 500.0, schedule.node_to_slot) == {}, algorithm


def test_random_layouts_dsatur_and_greedy_are_always_verified():
    """300 seeded random layouts: both heuristics give schedules that pass two independent checks."""
    for seed in range(300):
        coords, radio_range = random_layout(seed, max_nodes=30)
        G = build_radio_graph(coords, radio_range)
        H = build_conflict_graph(G)
        max_h_degree = max(d for _, d in H.degree())
        for algorithm in ("dsatur", "greedy"):
            _, _, schedule = schedule_for_graphs(G, H, algorithm)
            assert verify_schedule(G, schedule).ok, f"seed {seed} {algorithm}"
            assert conflicts_by_brute_force(coords, radio_range, schedule.node_to_slot) == {}, f"seed {seed} {algorithm}"
            assert schedule.frame_length <= max_h_degree + 1, f"seed {seed} {algorithm}"
            assert schedule.node_to_slot["Node_01"] == 0


def test_random_layouts_naive_d1_only_ever_fails_on_hidden_terminals():
    for seed in range(300):
        coords, radio_range = random_layout(seed, max_nodes=30)
        G = build_radio_graph(coords, radio_range)
        schedule = make_schedule(naive_distance1(G), list(G.nodes), "naive-d1")
        result = verify_schedule(G, schedule)
        assert all(v.kind == HIDDEN_TERMINAL for v in result.violations), f"seed {seed}"  # G is properly coloured
        assert {frozenset(p) for p in conflicts_by_brute_force(coords, radio_range, schedule.node_to_slot)} == {
            frozenset((v.u, v.v)) for v in result.violations
        }, f"seed {seed}"


def schedule_for_graphs(G, H, algorithm):
    nodes = list(G.nodes)
    if algorithm == "dsatur":
        return G, H, make_schedule(dsatur(H), nodes, DSATUR_NAME)
    return G, H, make_schedule(greedy_first_fit(H, sorted(nodes, key=natural_key)), nodes, "greedy")
