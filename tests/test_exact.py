"""Division 3 tests: the exact colouring search (tdma_brain.exact)."""
import itertools
import random
import sys
import time

import networkx as nx
import pytest

from tdma_brain import exact
from tdma_brain.bounds import lower_bound
from tdma_brain.exact import (
    BEST_FOUND,
    PROVEN_OPTIMAL_BY_BOUND,
    PROVEN_OPTIMAL_BY_SEARCH,
    SearchTimeout,
    colourable,
    solve_exact,
    validate_time_limit,
)
from tdma_brain.heuristics import dsatur, greedy_first_fit
from tdma_brain.io_utils import InputError, natural_key
from tests.golden import GOLDEN
from tests.helpers import build_graphs


def slots(colouring):
    return len(set(colouring.values()))


def is_proper(graph, colouring):
    return set(colouring) == set(graph.nodes) and all(colouring[u] != colouring[v] for u, v in graph.edges)


def rainbow(graph):
    """The worst incumbent: every node in a slot of its own."""
    return {v: i for i, v in enumerate(sorted(graph.nodes, key=natural_key))}


def brute_force_colourable(graph, k):
    """Is there a proper colouring with slots 0..k-1? Tries every assignment."""
    nodes = list(graph.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in graph.edges]
    return any(all(a[i] != a[j] for i, j in edges) for a in itertools.product(range(k), repeat=len(nodes)))


def brute_force_chromatic_number(graph):
    return next(k for k in range(0, graph.number_of_nodes() + 1) if brute_force_colourable(graph, k))


def small_random_graph(seed):
    rng = random.Random(seed)
    return nx.gnp_random_graph(rng.randint(1, 6), rng.choice([0.2, 0.5, 0.8]), seed=seed)


# --------------------------------------------------------------------------- correctness against brute force
def test_colourable_agrees_with_brute_force_for_every_number_of_slots():
    for seed in range(200):
        graph = small_random_graph(seed)
        for k in range(0, graph.number_of_nodes() + 1):
            found = colourable(graph, k, time.perf_counter() + 30)
            assert (found is not None) == brute_force_colourable(graph, k), f"seed {seed} k={k}"
            if found is not None:
                assert is_proper(graph, found) and slots(found) <= k, f"seed {seed} k={k}"


def test_the_exact_chromatic_number_equals_brute_force():
    for seed in range(200):
        graph = small_random_graph(seed)
        chi = brute_force_chromatic_number(graph)
        best, status = solve_exact(graph, 1, rainbow(graph), 30)  # from the worst possible start, with the weakest bound
        assert is_proper(graph, best) and slots(best) == chi, f"seed {seed}"
        assert status == (PROVEN_OPTIMAL_BY_BOUND if chi == 1 else PROVEN_OPTIMAL_BY_SEARCH), f"seed {seed}"
        clique = len(nx.max_weight_clique(graph, weight=None)[0])  # and with the clique bound as lb
        best, status = solve_exact(graph, clique, rainbow(graph), 30)
        assert slots(best) == chi, f"seed {seed}"
        assert status == (PROVEN_OPTIMAL_BY_BOUND if chi == clique else PROVEN_OPTIMAL_BY_SEARCH), f"seed {seed}"


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
def test_the_exact_solver_returns_the_chromatic_number_of_the_guides_known_graphs(graph, chi):
    for start in (dsatur(graph), greedy_first_fit(graph, sorted(graph.nodes, key=natural_key)), rainbow(graph)):
        best, status = solve_exact(graph, 1, start, 10)
        assert is_proper(graph, best) and slots(best) == chi
        assert status == PROVEN_OPTIMAL_BY_SEARCH


def test_mycielski_5_is_decided_quickly():
    graph = nx.mycielski_graph(5)  # 23 nodes, no triangles: the clique bound is only 2, so the search has real work
    started = time.perf_counter()
    best, status = solve_exact(graph, 2, dsatur(graph), 10)
    assert time.perf_counter() - started < 5
    assert (slots(best), status) == (5, PROVEN_OPTIMAL_BY_SEARCH)


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_the_golden_examples_get_their_proven_status(golden):
    G, H = build_graphs(golden.name)
    bound = lower_bound(G, H)
    best, status = solve_exact(H, bound.value, dsatur(H), 10)
    assert is_proper(H, best) and slots(best) == golden.optimal_slots
    # the ring is the one example whose bound (3) is below the optimum (4): the search has to prove it
    assert status == (PROVEN_OPTIMAL_BY_SEARCH if golden.name == "ring16_r1150m" else PROVEN_OPTIMAL_BY_BOUND)


# --------------------------------------------------------------------------- the time limit
@pytest.mark.parametrize("n", [6, 7])
def test_the_time_limit_is_respected(n):
    graph = nx.mycielski_graph(n)  # DSATUR already finds the optimum, but PROVING it is exponentially hard here
    start = dsatur(graph)
    started = time.perf_counter()
    best, status = solve_exact(graph, 2, start, time_limit_s=1)
    assert time.perf_counter() - started <= 2
    assert status == BEST_FOUND and slots(best) == n and best == start


def test_a_zero_time_limit_gives_up_at_once_unless_no_search_is_needed():
    graph = nx.mycielski_graph(7)
    started = time.perf_counter()
    best, status = solve_exact(graph, 2, dsatur(graph), 0)
    assert time.perf_counter() - started < 0.5 and status == BEST_FOUND
    triangle = nx.complete_graph(3)
    assert solve_exact(triangle, 3, dsatur(triangle), 0) == (dsatur(triangle), PROVEN_OPTIMAL_BY_BOUND)


def test_colourable_raises_search_timeout_once_the_deadline_has_passed():
    with pytest.raises(SearchTimeout):
        colourable(nx.petersen_graph(), 3, time.perf_counter() - 1)
    assert colourable(nx.Graph(), 3, time.perf_counter() - 1) == {}  # nothing to decide, nothing to time out


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.5])
def test_an_invalid_time_limit_is_an_input_error(bad):
    with pytest.raises(InputError, match="time limit"):
        validate_time_limit(bad)
    with pytest.raises(InputError, match="time limit"):
        solve_exact(nx.path_graph(3), 2, dsatur(nx.path_graph(3)), bad)


# --------------------------------------------------------------------------- statuses and the "never below the lower bound" rule
def test_no_search_at_all_when_the_incumbent_already_meets_the_bound(monkeypatch):
    monkeypatch.setattr(exact, "colourable", lambda *args: pytest.fail("no search is needed when K == lb"))
    triangle = nx.complete_graph(3)
    assert solve_exact(triangle, 3, dsatur(triangle), 10) == (dsatur(triangle), PROVEN_OPTIMAL_BY_BOUND)


def test_a_proof_that_one_slot_fewer_is_impossible():
    pentagon = nx.cycle_graph(5)  # clique bound 2, but an odd cycle needs 3
    start = dsatur(pentagon)
    assert slots(start) == 3
    assert solve_exact(pentagon, 2, start, 10) == (start, PROVEN_OPTIMAL_BY_SEARCH)


def test_improving_the_incumbent_down_to_the_bound():
    path = nx.path_graph(6)
    best, status = solve_exact(path, 2, rainbow(path), 10)  # 6 slots -> 2, and 2 is the lower bound
    assert is_proper(path, best) and slots(best) == 2 and status == PROVEN_OPTIMAL_BY_BOUND


def test_improving_the_incumbent_and_then_proving_the_rest():
    pentagon = nx.cycle_graph(5)
    best, status = solve_exact(pentagon, 2, rainbow(pentagon), 10)  # 5 slots -> 3, and 2 is impossible
    assert is_proper(pentagon, best) and slots(best) == 3 and status == PROVEN_OPTIMAL_BY_SEARCH


def test_the_search_never_asks_for_fewer_slots_than_the_lower_bound(monkeypatch):
    asked = []
    real = exact.colourable

    def spy(H, k, deadline):
        asked.append(k)
        return real(H, k, deadline)

    monkeypatch.setattr(exact, "colourable", spy)
    graph = nx.mycielski_graph(4)  # 11 nodes, chi = 4, clique 2
    best, status = solve_exact(graph, 3, rainbow(graph), 10)  # a valid but weak bound of 3
    assert slots(best) == 4 and status == PROVEN_OPTIMAL_BY_SEARCH
    assert asked and min(asked) == 3 and asked == sorted(asked, reverse=True)  # K-1, K-2, ..., down to 3 and no further
    asked.clear()
    best, status = solve_exact(graph, 4, rainbow(graph), 10)  # the tight bound: it stops as soon as it gets there
    assert slots(best) == 4 and status == PROVEN_OPTIMAL_BY_BOUND
    assert asked and min(asked) >= 4


def test_invalid_arguments_are_rejected():
    graph = nx.path_graph(4)
    good = {0: 0, 1: 1, 2: 0, 3: 1}
    with pytest.raises(ValueError, match="proper colouring"):
        solve_exact(graph, 2, {0: 0, 1: 0, 2: 1, 3: 1}, 10)  # two neighbours share a slot
    with pytest.raises(ValueError, match="proper colouring"):
        solve_exact(graph, 2, {0: 0, 1: 1, 2: 0}, 10)  # node 3 has no slot
    with pytest.raises(ValueError, match="bound is wrong"):
        solve_exact(graph, 3, good, 10)  # a valid 2-colouring cannot coexist with a lower bound of 3


def test_the_incumbent_is_not_modified():
    graph = nx.cycle_graph(5)
    incumbent = rainbow(graph)
    snapshot = dict(incumbent)
    solve_exact(graph, 2, incumbent, 10)
    assert incumbent == snapshot


# --------------------------------------------------------------------------- the search itself
def test_symmetry_breaking_makes_a_clique_refutable_at_once():
    # K14 cannot use 13 slots. Without "open at most one brand-new slot per step" the search
    # would try 13! = 6.2 billion namings of the same partial colourings.
    started = time.perf_counter()
    assert colourable(nx.complete_graph(14), 13, time.perf_counter() + 30) is None
    assert time.perf_counter() - started < 1


def test_colourable_handles_graphs_deeper_than_the_default_recursion_limit():
    limit = sys.getrecursionlimit()
    graph = nx.path_graph(limit + 100)  # the search recurses once per radio, so this is deeper than Python allows by default
    found = colourable(graph, 2, time.perf_counter() + 30)
    assert found is not None and is_proper(graph, found) and slots(found) == 2
    assert sys.getrecursionlimit() == limit  # and the limit is put back


def test_colourable_returns_slots_numbered_from_zero():
    graph = nx.petersen_graph()
    found = colourable(graph, 5, time.perf_counter() + 30)  # asked for at most 5, so any 3..5 is fine
    assert is_proper(graph, found) and set(found.values()) == set(range(slots(found))) and slots(found) <= 5
