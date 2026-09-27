"""Division 2 tests: the colouring heuristics (tdma_brain.heuristics)."""
import itertools
import random

import networkx as nx
import pytest

from tdma_brain.heuristics import (
    ALGORITHM_KEYS,
    DSATUR_NAME,
    GREEDY_NAME,
    NAIVE_D1_NAME,
    dsatur,
    greedy_first_fit,
    naive_distance1,
    run_algorithm,
)
from tdma_brain.io_utils import natural_key
from tests.golden import GOLDEN
from tests.helpers import build_graphs


def is_proper(graph, colouring):
    return set(colouring) == set(graph.nodes) and all(colouring[u] != colouring[v] for u, v in graph.edges)


def slots_used(colouring):
    return len(set(colouring.values()))


def natural_order(graph):
    return sorted(graph.nodes, key=natural_key)


def dsatur_by_the_book(H):
    """The guide's DSATUR spec transcribed literally, recomputing sat and udeg every step.

    Slow but obviously right: the fast dsatur() must agree with it on every graph.
    """
    order = natural_order(H)
    colour = {}
    while len(colour) < len(order):
        best = None
        for v in order:  # natural order, and only a strictly larger key replaces: ties keep the earlier node
            if v in colour:
                continue
            sat = len({colour[u] for u in H[v] if u in colour})
            udeg = sum(1 for u in H[v] if u not in colour)
            if best is None or (sat, udeg) > best[0]:
                best = ((sat, udeg), v)
        v = best[1]
        used = {colour[u] for u in H[v] if u in colour}
        colour[v] = next(c for c in itertools.count() if c not in used)
    return colour


def random_graph(seed):
    rng = random.Random(seed)
    graph = nx.gnp_random_graph(rng.randint(1, 30), rng.choice([0.05, 0.15, 0.3, 0.6]), seed=seed)
    return graph


# --------------------------------------------------------------------------- greedy first-fit
def test_greedy_gives_each_node_the_smallest_free_slot():
    path = nx.path_graph(["Node_1", "Node_2", "Node_3", "Node_4"])
    assert greedy_first_fit(path, ["Node_1", "Node_2", "Node_3", "Node_4"]) == {
        "Node_1": 0, "Node_2": 1, "Node_3": 0, "Node_4": 1,
    }
    triangle = nx.complete_graph(["A", "B", "C"])
    assert greedy_first_fit(triangle, ["C", "A", "B"]) == {"C": 0, "A": 1, "B": 2}  # the order decides who gets slot 0


def test_greedy_slot_is_the_smallest_free_slot_among_earlier_neighbours():
    for seed in range(50):
        graph = random_graph(seed)
        order = natural_order(graph)
        colour = greedy_first_fit(graph, order)
        for i, v in enumerate(order):
            earlier = {colour[u] for u in graph[v] if u in order[:i]}
            assert colour[v] == next(s for s in itertools.count() if s not in earlier), f"seed {seed}"


def test_vertex_order_changes_how_many_slots_greedy_needs():
    # the "crown" graph: a_i is linked to b_j for every j != i. Two slots are enough, but
    # an unlucky order pairs a_1 with b_1 first and ends up needing three.
    crown = nx.Graph((f"a{i}", f"b{j}") for i in (1, 2, 3) for j in (1, 2, 3) if i != j)
    unlucky = ["a1", "b1", "a2", "b2", "a3", "b3"]
    lucky = ["a1", "a2", "a3", "b1", "b2", "b3"]
    assert slots_used(greedy_first_fit(crown, unlucky)) == 3
    assert slots_used(greedy_first_fit(crown, lucky)) == 2


@pytest.mark.parametrize(
    "order",
    [["A", "B"], ["A", "B", "C", "C"], ["A", "B", "C", "D"], []],
    ids=["missing node", "duplicate node", "unknown node", "empty"],
)
def test_greedy_rejects_an_order_that_is_not_a_permutation(order):
    with pytest.raises(ValueError, match="exactly once"):
        greedy_first_fit(nx.path_graph(["A", "B", "C"]), order)


def test_greedy_is_proper_and_within_max_degree_plus_one_on_random_graphs():
    for seed in range(200):
        graph = random_graph(seed)
        colour = greedy_first_fit(graph, natural_order(graph))
        assert is_proper(graph, colour), f"seed {seed}"
        assert slots_used(colour) <= max((d for _, d in graph.degree()), default=0) + 1, f"seed {seed}"


# --------------------------------------------------------------------------- DSATUR
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_dsatur_is_a_proper_colouring_of_every_conflict_graph(golden):
    _, H = build_graphs(golden.name)
    colour = dsatur(H)
    assert is_proper(H, colour)
    assert slots_used(colour) <= max(d for _, d in H.degree()) + 1  # the greedy guarantee


def test_dsatur_matches_the_spec_transcribed_literally():
    for seed in range(300):
        graph = random_graph(seed)
        assert dsatur(graph) == dsatur_by_the_book(graph), f"seed {seed}"
    for golden in GOLDEN:
        _, H = build_graphs(golden.name)
        assert dsatur(H) == dsatur_by_the_book(H), golden.name


def test_dsatur_picks_the_most_uncoloured_neighbours_first_not_the_earliest_node():
    # star: Node_5 in the middle. All saturations start at 0, so udeg decides, not natural order
    star = nx.star_graph(["Node_5", "Node_1", "Node_2", "Node_3"])
    assert dsatur(star) == {"Node_5": 0, "Node_1": 1, "Node_2": 1, "Node_3": 1}


def dsatur_variant(H, rule):
    """DSATUR with a different pick rule, only to show that PICK_RULE_GRAPH tells the rules apart."""
    order = natural_order(H)
    rank = {v: i for i, v in enumerate(order)}
    colour = {}
    while len(colour) < len(order):
        def key(v):
            sat = len({colour[u] for u in H[v] if u in colour})
            udeg = sum(1 for u in H[v] if u not in colour)
            return {"udeg first": (udeg, sat), "saturation only": (sat,), "uncoloured degree only": (udeg,)}[rule] + (-rank[v],)

        v = max((x for x in order if x not in colour), key=key)
        used = {colour[u] for u in H[v] if u in colour}
        colour[v] = next(c for c in itertools.count() if c not in used)
    return colour


PICK_RULE_GRAPH = nx.Graph([
    ("Node_1", "Node_2"), ("Node_1", "Node_6"), ("Node_2", "Node_3"), ("Node_2", "Node_6"),
    ("Node_3", "Node_5"), ("Node_4", "Node_5"), ("Node_4", "Node_6"),
])


def test_dsatur_pick_rule_is_saturation_then_uncoloured_degree_then_natural_order():
    # Traced by hand: Node_2 first (udeg 3 ties with Node_6, earlier wins), then Node_6 (sat 1, udeg 2),
    # Node_1 (sat 2), Node_3 (ties with Node_4, earlier wins), Node_4, Node_5.
    expected = {"Node_2": 0, "Node_6": 1, "Node_1": 2, "Node_3": 1, "Node_4": 0, "Node_5": 2}
    assert dsatur(PICK_RULE_GRAPH) == expected == dsatur_by_the_book(PICK_RULE_GRAPH)
    # the three tempting-but-wrong pick rules colour this very graph differently
    for rule in ("udeg first", "saturation only", "uncoloured degree only"):
        assert dsatur_variant(PICK_RULE_GRAPH, rule) != expected, rule


def test_dsatur_does_not_depend_on_the_order_the_graph_was_built_in():
    _, H = build_graphs("grid16_300m")
    expected = dsatur(H)
    rng = random.Random(7)
    for _ in range(10):
        nodes, edges = list(H.nodes), list(H.edges)
        rng.shuffle(nodes)
        rng.shuffle(edges)
        shuffled = nx.Graph()
        shuffled.add_nodes_from(nodes)
        shuffled.add_edges_from(edges)
        assert dsatur(shuffled) == expected


def test_dsatur_accepts_integer_nodes_and_an_empty_graph():
    assert is_proper(nx.mycielski_graph(4), dsatur(nx.mycielski_graph(4)))
    assert dsatur(nx.Graph()) == {}


KNOWN_CHI = [  # guide section 5.2: (graph, chromatic number)
    (nx.complete_graph(5), 5),
    (nx.cycle_graph(7), 3),
    (nx.cycle_graph(8), 2),
    (nx.petersen_graph(), 3),
    (nx.mycielski_graph(4), 4),
    (nx.mycielski_graph(5), 5),
    (nx.wheel_graph(6), 4),
    (nx.wheel_graph(7), 3),
    (nx.power(nx.cycle_graph(7), 2), 4),
    (nx.power(nx.path_graph(16), 2), 3),
]


@pytest.mark.parametrize("graph, chi", KNOWN_CHI, ids=[f"chi={chi}#{i}" for i, (_, chi) in enumerate(KNOWN_CHI)])
def test_dsatur_on_the_guides_known_graphs(graph, chi):
    colour = dsatur(graph)
    assert is_proper(graph, colour)
    assert slots_used(colour) >= chi  # no proper colouring can beat the chromatic number
    assert slots_used(colour) == chi  # and this heuristic happens to reach it on every one of them


# --------------------------------------------------------------------------- naive distance-1
def test_naive_distance1_is_documented_as_unsafe_in_capitals():
    assert "INTENTIONALLY UNSAFE" in naive_distance1.__doc__
    assert "DEMOS" in naive_distance1.__doc__


@pytest.mark.parametrize("name, slots", [("grid16_300m", 4), ("grid16_400m", 2)])
def test_naive_distance1_colours_g_properly_but_leaves_hidden_terminals(name, slots):
    G, H = build_graphs(name)
    colour = naive_distance1(G)
    assert is_proper(G, colour)  # radios that hear each other never share a slot ...
    assert slots_used(colour) == slots  # ... so it looks great: few slots (guide 5.3) ...
    clashes = [(u, v) for u, v in H.edges if colour[u] == colour[v]]
    assert clashes and all(not G.has_edge(u, v) for u, v in clashes)  # ... but every clash is a hidden terminal


def test_naive_distance1_is_harmless_when_nobody_hears_anybody():
    G, H = build_graphs("isolated16_1000m")
    assert set(naive_distance1(G).values()) == {0}
    assert H.number_of_edges() == 0


# --------------------------------------------------------------------------- registry
def test_run_algorithm_maps_each_key_to_its_heuristic():
    G, H = build_graphs("grid16_400m")
    assert run_algorithm("dsatur", G, H) == (DSATUR_NAME, dsatur(H))
    assert run_algorithm("greedy", G, H) == (GREEDY_NAME, greedy_first_fit(H, natural_order(H)))
    assert run_algorithm("naive-d1", G, H) == (NAIVE_D1_NAME, naive_distance1(G))
    assert set(ALGORITHM_KEYS) == {"dsatur", "greedy", "naive-d1"}


def test_run_algorithm_rejects_an_unknown_key():
    G, H = build_graphs("hidden_terminal3")
    with pytest.raises(ValueError, match="unknown algorithm 'magic'"):
        run_algorithm("magic", G, H)
