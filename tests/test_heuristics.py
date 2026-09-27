"""Division 2 tests: the colouring heuristics (tdma_brain.heuristics)."""
import itertools
import random
import types

import networkx as nx
import pytest

from tdma_brain import heuristics
from tdma_brain.heuristics import (
    ALGORITHM_KEYS,
    DSATUR_NAME,
    GREEDY_NAME,
    LARGEST_FIRST_NAME,
    NAIVE_D1_NAME,
    RLF_NAME,
    SMALLEST_LAST_NAME,
    dsatur,
    greedy_first_fit,
    iterated_greedy,
    iterated_greedy_passes,
    largest_first,
    naive_distance1,
    rlf,
    run_algorithm,
    smallest_last,
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
    assert run_algorithm("rlf", G, H) == (RLF_NAME, rlf(H))
    assert run_algorithm("smallest_last", G, H) == (SMALLEST_LAST_NAME, smallest_last(H))
    assert run_algorithm("largest_first", G, H) == (LARGEST_FIRST_NAME, largest_first(H))
    # the guide's CLI order: the portfolio heuristics first, the unsafe demo last
    assert ALGORITHM_KEYS == ("dsatur", "rlf", "smallest_last", "largest_first", "greedy", "naive-d1")


def test_run_algorithm_rejects_an_unknown_key():
    G, H = build_graphs("hidden_terminal3")
    with pytest.raises(ValueError, match="unknown algorithm 'magic'") as info:
        run_algorithm("magic", G, H)
    assert all(key in str(info.value) for key in ALGORITHM_KEYS)


# --------------------------------------------------------------------------- Division 3: Welsh-Powell, smallest-last, RLF
def largest_first_by_the_book(H):
    """The guide's spec for Welsh-Powell: degree descending, ties in natural order, then first-fit."""
    return greedy_first_fit(H, sorted(natural_order(H), key=lambda v: -H.degree(v)))  # a stable sort keeps natural order for ties


def smallest_last_by_the_book(H):
    """The guide's spec for smallest-last, recomputing the degrees among the remaining nodes every time."""
    order = natural_order(H)
    remaining, removal = list(order), []
    while remaining:
        degree = {v: sum(1 for u in H[v] if u in remaining) for v in remaining}
        v = min(remaining, key=lambda x: (degree[x], order.index(x)))
        removal.append(v)
        remaining.remove(v)
    return greedy_first_fit(H, list(reversed(removal)))


def rlf_by_the_book(H):
    """The guide's RLF spec transcribed literally: X = candidates, Y = blocked, recomputed from scratch."""
    order = natural_order(H)
    U, colour, slot = set(order), {}, 0
    while U:
        X, Y = set(U), set()
        best = None
        for x in order:  # first radio of the slot: the most neighbours inside U, ties in natural order
            if x in X and (best is None or len(set(H[x]) & U) > best[0]):
                best = (len(set(H[x]) & U), x)
        v = best[1]
        colour[v] = slot
        Y |= set(H[v]) & X
        X -= {v} | set(H[v])
        while X:
            best = None
            for x in order:  # then: most neighbours in Y, fewest in X, ties in natural order
                key = (len(set(H[x]) & Y), -len(set(H[x]) & X))
                if x in X and (best is None or key > best[0]):
                    best = (key, x)
            v = best[1]
            colour[v] = slot
            Y |= set(H[v]) & X
            X -= {v} | set(H[v])
        U -= {n for n in colour if colour[n] == slot}
        slot += 1
    return colour


NEW_HEURISTICS = [
    ("largest_first", largest_first, largest_first_by_the_book),
    ("smallest_last", smallest_last, smallest_last_by_the_book),
    ("rlf", rlf, rlf_by_the_book),
]


@pytest.mark.parametrize("name, fast, by_the_book", NEW_HEURISTICS, ids=[n for n, _, _ in NEW_HEURISTICS])
def test_new_heuristics_match_the_spec_transcribed_literally(name, fast, by_the_book):
    for seed in range(300):
        graph = random_graph(seed)
        assert fast(graph) == by_the_book(graph), f"{name} seed {seed}"
    for golden in GOLDEN:
        _, H = build_graphs(golden.name)
        assert fast(H) == by_the_book(H), f"{name} {golden.name}"


@pytest.mark.parametrize("name, fast, by_the_book", NEW_HEURISTICS, ids=[n for n, _, _ in NEW_HEURISTICS])
def test_new_heuristics_are_proper_and_within_max_degree_plus_one(name, fast, by_the_book):
    for golden in GOLDEN:
        _, H = build_graphs(golden.name)
        colour = fast(H)
        assert is_proper(H, colour) and slots_used(colour) <= max(d for _, d in H.degree()) + 1, golden.name
    for seed in range(200):
        graph = random_graph(seed)
        colour = fast(graph)
        assert is_proper(graph, colour), f"{name} seed {seed}"
        assert slots_used(colour) <= max((d for _, d in graph.degree()), default=0) + 1, f"{name} seed {seed}"


def test_smallest_last_never_needs_more_than_the_degeneracy_plus_one():
    # the Matula-Beck guarantee: colouring in reverse peeling order needs at most (degeneracy + 1) slots
    for seed in range(200):
        graph = random_graph(seed)
        degeneracy = max(nx.core_number(graph).values(), default=-1)
        assert slots_used(smallest_last(graph)) <= degeneracy + 1, f"seed {seed}"


def test_welsh_powell_stays_within_its_classical_bound():
    # Welsh and Powell (1967): at most max over i of min(d_i + 1, i), with d_1 >= d_2 >= ... the degrees
    for seed in range(200):
        graph = random_graph(seed)
        degrees = sorted((d for _, d in graph.degree()), reverse=True)
        bound = max((min(d + 1, i) for i, d in enumerate(degrees, start=1)), default=0)
        assert slots_used(largest_first(graph)) <= bound, f"seed {seed}"


def test_rlf_builds_each_slot_as_a_maximal_independent_set_of_what_is_left():
    for seed in range(100):
        graph = random_graph(seed)
        colour = rlf(graph)
        classes = [{v for v in colour if colour[v] == s} for s in range(slots_used(colour))]
        left = set(graph.nodes)
        for members in classes:
            assert not any(graph.has_edge(u, v) for u in members for v in members), f"seed {seed}"  # independent
            assert all(any(graph.has_edge(v, m) for m in members) for v in left - members), f"seed {seed}"  # maximal
            left -= members
        assert not left


def test_rlf_fills_the_first_slot_starting_from_the_most_conflicted_radio():
    star = nx.star_graph(["Node_5", "Node_1", "Node_2", "Node_3"])  # Node_5 is linked to all the others
    colour = rlf(star)
    assert colour["Node_5"] == 0 and {colour[n] for n in ("Node_1", "Node_2", "Node_3")} == {1}


@pytest.mark.parametrize("name, fast, by_the_book", NEW_HEURISTICS, ids=[n for n, _, _ in NEW_HEURISTICS])
def test_new_heuristics_on_the_guides_known_graphs(name, fast, by_the_book):
    for graph, chi in [(g, c) for g, c in KNOWN_CHI]:
        colour = fast(graph)
        assert is_proper(graph, colour) and slots_used(colour) >= chi  # never below the chromatic number


@pytest.mark.parametrize("name, fast, by_the_book", NEW_HEURISTICS, ids=[n for n, _, _ in NEW_HEURISTICS])
def test_new_heuristics_ignore_construction_order_and_accept_integer_nodes(name, fast, by_the_book):
    _, H = build_graphs("grid16_400m")
    expected = fast(H)
    rng = random.Random(11)
    for _ in range(5):
        nodes, edges = list(H.nodes), list(H.edges)
        rng.shuffle(nodes)
        rng.shuffle(edges)
        shuffled = nx.Graph()
        shuffled.add_nodes_from(nodes)
        shuffled.add_edges_from(edges)
        assert fast(shuffled) == expected
    assert is_proper(nx.mycielski_graph(4), fast(nx.mycielski_graph(4)))
    assert fast(nx.Graph()) == {}


def test_grid_400m_heuristic_quality_matches_the_guide():
    # guide 5.1 / 5.3: DSATUR and RLF find the optimum of 5; natural-order greedy and Welsh-Powell need 6
    _, H = build_graphs("grid16_400m")
    assert slots_used(dsatur(H)) == 5 and slots_used(rlf(H)) == 5
    assert slots_used(largest_first(H)) == 6
    assert slots_used(greedy_first_fit(H, natural_order(H))) == 6
    assert slots_used(smallest_last(H)) == 6  # the guide's verbose example


# --------------------------------------------------------------------------- Division 3: iterated greedy
def test_iterated_greedy_never_increases_the_slot_count_at_any_iteration():
    for seed in range(50):
        graph = random_graph(seed)
        start = greedy_first_fit(graph, natural_order(graph))
        previous = slots_used(start)
        for colouring in iterated_greedy_passes(graph, start, iterations=100, seed=seed):
            assert is_proper(graph, colouring), f"seed {seed}"
            assert slots_used(colouring) <= previous, f"seed {seed}"
            previous = slots_used(colouring)
        assert slots_used(iterated_greedy(graph, start, iterations=100, seed=seed)) <= slots_used(start)


def test_iterated_greedy_starts_from_any_proper_colouring_even_a_wasteful_one():
    _, H = build_graphs("grid16_300m")
    wasteful = {v: 10 * i for i, v in enumerate(natural_order(H))}  # 16 slots with gaps in the numbering
    result = iterated_greedy(H, wasteful, iterations=50, seed=1)
    assert is_proper(H, result) and slots_used(result) <= 16


def test_iterated_greedy_repairs_a_wasteful_colouring():
    _, H = build_graphs("grid16_400m")
    assert slots_used(smallest_last(H)) == 6
    assert slots_used(iterated_greedy(H, smallest_last(H), iterations=300, seed=42)) == 5  # 6 -> 5, the optimum
    # on random graphs it usually finds a shorter frame than natural-order first-fit (47 of these 80)
    improved = 0
    for seed in range(80):
        rng = random.Random(seed)
        graph = nx.gnp_random_graph(rng.randint(8, 30), rng.choice([0.15, 0.3, 0.5]), seed=seed)
        start = greedy_first_fit(graph, natural_order(graph))
        improved += slots_used(iterated_greedy(graph, start, iterations=300, seed=42)) < slots_used(start)
    assert improved >= 40


def test_iterated_greedy_cannot_escape_rigid_colour_classes():
    # In the "crown" graph the classes {a_i, b_i} are pairs that first-fit keeps together in EVERY class
    # order, so this 3-slot colouring is a fixed point even though 2 slots suffice. That limit is
    # why the optimizer also runs a portfolio of other heuristics and, if needed, the exact search.
    crown = nx.Graph((f"a{i}", f"b{j}") for i in (1, 2, 3) for j in (1, 2, 3) if i != j)
    unlucky = greedy_first_fit(crown, ["a1", "b1", "a2", "b2", "a3", "b3"])
    assert slots_used(unlucky) == 3
    assert slots_used(iterated_greedy(crown, unlucky, iterations=300, seed=42)) == 3
    assert slots_used(dsatur(crown)) == 2 and slots_used(rlf(crown)) == 2


def test_iterated_greedy_is_deterministic_for_a_seed_and_the_seed_matters():
    _, H = build_graphs("grid16_300m")
    start = dsatur(H)
    same_a = list(iterated_greedy_passes(H, start, 30, seed=7))
    same_b = list(iterated_greedy_passes(H, start, 30, seed=7))
    other = list(iterated_greedy_passes(H, start, 30, seed=8))
    assert same_a == same_b and same_a != other


def test_iterated_greedy_only_replaces_the_start_on_a_strict_improvement():
    # One edge a-b and two isolated radios: {a, c} / {b, d} is already optimal (2 slots). Reordering the
    # classes produces OTHER 2-slot colourings, but none is better, so the start must come back unchanged.
    graph = nx.Graph([("a", "b")])
    graph.add_nodes_from(["c", "d"])
    start = {"a": 0, "b": 1, "c": 0, "d": 1}
    variants = {tuple(sorted(c.items())) for c in iterated_greedy_passes(graph, start, iterations=50, seed=1)}
    assert len(variants) > 1  # the passes really do wander between different equal-slot colourings ...
    assert iterated_greedy(graph, start, iterations=50, seed=1) == start  # ... and none of them replaces the start


def test_iterated_greedy_with_no_iterations_returns_a_copy_of_the_start():
    _, H = build_graphs("hidden_terminal3")
    start = dsatur(H)
    result = iterated_greedy(H, start, iterations=0)
    assert result == start and result is not start


# A complete multipartite graph keeps its colour classes (the parts) intact from pass to pass, so the order
# of every iterated-greedy pass can be predicted. Parts: P0={n1}  P1={n2,n3,n4}  P2={n5,n6}  P3={n7,n8}.
PARTS_GRAPH = nx.relabel_nodes(nx.complete_multipartite_graph(1, 3, 2, 2), {i: f"n{i + 1}" for i in range(8)})
PARTS_START = {"n1": 0, "n2": 1, "n3": 1, "n4": 1, "n5": 2, "n6": 2, "n7": 3, "n8": 3}
LARGEST_FIRST_ORDER = ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n1"]  # classes by size, largest first, ties: lower slot first
REVERSED_ORDER = ["n7", "n8", "n5", "n6", "n2", "n3", "n4", "n1"]  # the classes in reverse slot order


def scripted_random(answers, shuffled):
    """A stand-in for `random` inside heuristics: Random().random() answers the scripted values in turn and
    shuffle() records the list it was given and reverses it in place (a fixed, predictable shuffle)."""

    class ScriptedRandom:
        def __init__(self, seed):
            self.answers = iter(answers)

        def random(self):
            return next(self.answers)

        def shuffle(self, items):
            shuffled.append(list(items))
            items.reverse()

    return types.SimpleNamespace(Random=ScriptedRandom)


@pytest.mark.parametrize(
    "r, branch",
    [(0.0, "largest first"), (0.49, "largest first"), (0.5, "reverse"), (0.79, "reverse"), (0.8, "shuffle"), (0.99, "shuffle")],
)
def test_iterated_greedy_picks_its_reordering_at_the_spec_thresholds(monkeypatch, r, branch):
    seen_orders, shuffled = [], []
    real_greedy = heuristics.greedy_first_fit
    monkeypatch.setattr(heuristics, "greedy_first_fit", lambda H, order: (seen_orders.append(list(order)), real_greedy(H, order))[1])
    monkeypatch.setattr(heuristics, "random", scripted_random([r], shuffled))
    list(iterated_greedy_passes(PARTS_GRAPH, PARTS_START, iterations=1, seed=0))
    if branch == "largest first":  # r < 0.5
        assert seen_orders == [LARGEST_FIRST_ORDER] and not shuffled
    elif branch == "reverse":  # 0.5 <= r < 0.8
        assert seen_orders == [REVERSED_ORDER] and not shuffled
    else:  # r >= 0.8: shuffled (here: reversed by the scripted shuffle), and the classes were handed over in slot order
        assert shuffled == [[["n1"], ["n2", "n3", "n4"], ["n5", "n6"], ["n7", "n8"]]] and seen_orders == [REVERSED_ORDER]


def test_iterated_greedy_complains_if_a_pass_ever_needs_more_slots(monkeypatch):
    # Culberson's theorem says this cannot happen; if it ever did, the code must say so instead of returning a worse frame
    _, H = build_graphs("grid16_300m")
    start = dsatur(H)  # 9 slots
    wasteful = {v: i for i, v in enumerate(natural_order(H))}  # 16 slots
    monkeypatch.setattr(heuristics, "greedy_first_fit", lambda H_, order: dict(wasteful))
    with pytest.raises(RuntimeError, match="Culberson"):
        list(iterated_greedy_passes(H, start, iterations=1, seed=1))


def test_iterated_greedy_reorders_whole_classes_the_way_the_spec_says(monkeypatch):
    graph, start = PARTS_GRAPH, PARTS_START
    assert is_proper(graph, start)
    seen_orders, shuffled = [], []
    real_greedy = heuristics.greedy_first_fit

    def spy(H, order):
        seen_orders.append(list(order))
        return real_greedy(H, order)

    monkeypatch.setattr(heuristics, "greedy_first_fit", spy)
    monkeypatch.setattr(heuristics, "random", scripted_random([0.1, 0.6, 0.9], shuffled))  # largest first, reverse, shuffle
    passes = list(iterated_greedy_passes(graph, start, iterations=3, seed=0))

    # pass 1, r = 0.1: largest class first; the equal-sized P2 and P3 keep the lower slot number first
    assert seen_orders[0] == ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n1"]
    assert [passes[0][n] for n in ("n2", "n5", "n7", "n1")] == [0, 1, 2, 3]  # the classes now sit in slots 0..3 in that order
    # pass 2, r = 0.6: the class order of the new colouring (P1, P2, P3, P0) is reversed
    assert seen_orders[1] == ["n1", "n7", "n8", "n5", "n6", "n2", "n3", "n4"]
    # pass 3, r = 0.9: rng.shuffle() gets the classes in slot order (P0, P3, P2, P1)
    assert shuffled == [[["n1"], ["n7", "n8"], ["n5", "n6"], ["n2", "n3", "n4"]]]
    assert seen_orders[2] == ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n1"]
    assert all(is_proper(graph, colouring) and slots_used(colouring) == 4 for colouring in passes)  # the parts never split
