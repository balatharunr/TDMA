"""Division 3 tests: the optimizer pipeline (tdma_brain.optimizer)."""
import time

import pytest

from tdma_brain import optimizer
from tdma_brain.exact import BEST_FOUND, PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH
from tdma_brain.heuristics import (
    DSATUR_NAME,
    GREEDY_NAME,
    LARGEST_FIRST_NAME,
    RLF_NAME,
    SMALLEST_LAST_NAME,
    dsatur,
    greedy_first_fit,
    largest_first,
    rlf,
    smallest_last,
)
from tdma_brain.io_utils import InputError, natural_key
from tdma_brain.optimizer import ITERATED_GREEDY_ITERATIONS, PORTFOLIO_KEYS, HeuristicRun, OptimizationResult, optimize
from tdma_brain.topology import build_conflict_graph, build_radio_graph
from tdma_brain.verify import verify_schedule
from tests.golden import GOLDEN
from tests.helpers import build_graphs, conflicts_by_brute_force, random_layout

PORTFOLIO_NAMES = [DSATUR_NAME, RLF_NAME, SMALLEST_LAST_NAME, LARGEST_FIRST_NAME, GREEDY_NAME]


# --------------------------------------------------------------------------- the golden examples
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_every_example_is_optimal_verified_and_proven(golden):
    G, H = build_graphs(golden.name)
    result = optimize(G, H)
    assert isinstance(result, OptimizationResult)
    assert result.verification.ok and verify_schedule(G, result.schedule).ok
    assert result.schedule.frame_length == golden.optimal_slots
    assert result.lower_bound.value == golden.max_clique
    assert result.lower_bound.maxdeg_plus_1 == golden.maxdeg_plus_1 and result.lower_bound.clique_size == golden.max_clique
    # only the ring has a gap between the bound (3) and the optimum (4), and only there the search has to prove it
    assert result.status == (PROVEN_OPTIMAL_BY_SEARCH if golden.name == "ring16_r1150m" else PROVEN_OPTIMAL_BY_BOUND)
    assert result.exact_start == (4 if golden.name == "ring16_r1150m" else None)  # the search runs only if K > LB


def test_grid_400m_shows_why_a_portfolio_helps():
    G, H = build_graphs("grid16_400m")
    result = optimize(G, H)
    assert [(run.name, run.slots) for run in result.portfolio] == list(zip(PORTFOLIO_NAMES, [5, 5, 6, 6, 6]))
    assert result.best_name == DSATUR_NAME and result.best_slots == 5  # DSATUR and RLF tie at 5: the earlier one wins
    assert result.polished_slots == 5 and result.schedule.algorithm == "DSATUR"  # iterated greedy found nothing better
    assert (result.lower_bound.value, result.status) == (5, PROVEN_OPTIMAL_BY_BOUND)


def test_ring_needs_the_exact_search_to_prove_its_answer():
    G, H = build_graphs("ring16_r1150m")
    result = optimize(G, H)
    assert result.lower_bound.value == 3 and result.polished_slots == 4
    assert result.exact_start == 4 and result.status == PROVEN_OPTIMAL_BY_SEARCH
    assert result.schedule.algorithm == "DSATUR"  # the search did not change the schedule, it proved it


def test_the_official_grid_keeps_the_dsatur_schedule_of_division_2():
    G, H = build_graphs("grid16_300m")
    result = optimize(G, H)
    assert result.schedule.node_to_slot == make_expected_official_mapping()
    assert result.schedule.algorithm == "DSATUR" and result.exact_start is None


def make_expected_official_mapping():
    slots = [0, 1, 2, 0, 3, 4, 5, 3, 6, 7, 8, 6, 0, 1, 2, 0]  # guide 5.3
    return {f"Node_{i:02d}": s for i, s in enumerate(slots, start=1)}


# --------------------------------------------------------------------------- the pipeline's parts
def test_the_portfolio_table_lists_every_heuristic_in_priority_order():
    G, H = build_graphs("grid16_300m")
    result = optimize(G, H)
    assert PORTFOLIO_KEYS == ("dsatur", "rlf", "smallest_last", "largest_first", "greedy")
    assert [run.name for run in result.portfolio] == PORTFOLIO_NAMES
    direct = [dsatur(H), rlf(H), smallest_last(H), largest_first(H), greedy_first_fit(H, sorted(H.nodes, key=natural_key))]
    assert [run.slots for run in result.portfolio] == [len(set(c.values())) for c in direct]
    assert all(isinstance(run, HeuristicRun) and run.ms >= 0 for run in result.portfolio)
    assert (result.node_count, result.conflict_pairs) == (16, 90)
    assert result.total_ms >= max(run.ms for run in result.portfolio) and result.iterations == ITERATED_GREEDY_ITERATIONS == 300


def test_iterated_greedy_polishes_a_wasteful_portfolio_and_is_named_in_the_schedule(monkeypatch):
    G, H = build_graphs("grid16_400m")
    wasteful = smallest_last(H)  # 6 slots; iterated greedy repairs it to the optimum of 5
    monkeypatch.setattr(optimizer, "run_algorithm", lambda key, G_, H_: (key.upper(), dict(wasteful)))
    result = optimize(G, H)
    assert (result.best_slots, result.polished_slots) == (6, 5)
    assert result.best_name == "DSATUR" and result.schedule.algorithm == "DSATUR + iterated greedy"  # first in the list wins the tie
    assert result.schedule.frame_length == 5 and result.status == PROVEN_OPTIMAL_BY_BOUND and result.verification.ok


def test_the_exact_search_is_named_when_it_improves_the_result(monkeypatch):
    G, H = build_graphs("grid16_400m")
    rainbow = {v: i for i, v in enumerate(sorted(H.nodes, key=natural_key))}  # 16 slots: as wasteful as it gets
    monkeypatch.setattr(optimizer, "run_algorithm", lambda key, G_, H_: (key.upper(), dict(rainbow)))
    monkeypatch.setattr(optimizer, "iterated_greedy", lambda H_, start, iterations, seed: dict(start))  # no polishing
    result = optimize(G, H)
    assert (result.best_slots, result.polished_slots, result.exact_start) == (16, 16, 16)
    assert result.schedule.frame_length == 5 and result.schedule.algorithm == "DSATUR + exact search"
    assert result.status == PROVEN_OPTIMAL_BY_BOUND and result.verification.ok


def test_a_tie_goes_to_the_earlier_heuristic_in_the_priority_list(monkeypatch):
    G, H = build_graphs("grid16_400m")
    colourings = {"dsatur": largest_first(H), "rlf": rlf(H), "smallest_last": rlf(H), "largest_first": rlf(H), "greedy": rlf(H)}
    monkeypatch.setattr(optimizer, "run_algorithm", lambda key, G_, H_: (key.upper(), colourings[key]))
    result = optimize(G, H)  # dsatur is given 6 slots, all the others 5: the earliest of the fives (rlf) wins
    assert [run.slots for run in result.portfolio] == [6, 5, 5, 5, 5]
    assert result.best_name == "RLF"


def test_a_zero_time_limit_leaves_the_gap_open_instead_of_searching():
    G, H = build_graphs("ring16_r1150m")
    result = optimize(G, H, time_limit=0)
    assert result.status == BEST_FOUND and result.exact_start == 4
    assert result.schedule.frame_length == 4 and result.lower_bound.value == 3  # gap = 1
    assert result.verification.ok
    G, H = build_graphs("grid16_300m")
    assert optimize(G, H, time_limit=0).status == PROVEN_OPTIMAL_BY_BOUND  # no gap, no search, no time needed


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_an_invalid_time_limit_is_rejected(bad):
    G, H = build_graphs("hidden_terminal3")
    with pytest.raises(InputError, match="time limit"):
        optimize(G, H, time_limit=bad)


def test_the_result_does_not_depend_on_anything_but_the_input_and_the_seed():
    G, H = build_graphs("grid16_250m")
    first, second = optimize(G, H, seed=42), optimize(G, H, seed=42)
    assert first.schedule == second.schedule and first.status == second.status
    assert [(r.name, r.slots) for r in first.portfolio] == [(r.name, r.slots) for r in second.portfolio]
    other_seed = optimize(G, H, seed=7)
    assert other_seed.verification.ok and other_seed.schedule.frame_length == first.schedule.frame_length


def test_a_bound_above_the_schedule_is_reported_as_a_bug(monkeypatch):
    G, H = build_graphs("grid16_300m")
    real = optimizer.lower_bound
    monkeypatch.setattr(optimizer, "lower_bound", lambda G_, H_: real(G_, H_).__class__(10, 10, 10, ()))
    with pytest.raises(RuntimeError, match="below the proven lower bound"):
        optimize(G, H)


# --------------------------------------------------------------------------- fuzz
def test_random_topologies_are_verified_bounded_and_consistently_labelled():
    """300 seeded random layouts (2 to 40 radios): valid schedules, LB <= K <= every heuristic, honest status."""
    started = time.perf_counter()
    for seed in range(300):
        coords, radio_range = random_layout(seed, max_nodes=40)
        G = build_radio_graph(coords, radio_range)
        H = build_conflict_graph(G)
        result = optimize(G, H, time_limit=2.0)
        schedule, bound = result.schedule, result.lower_bound
        assert result.verification.ok and verify_schedule(G, schedule).ok, f"seed {seed}"
        assert conflicts_by_brute_force(coords, radio_range, schedule.node_to_slot) == {}, f"seed {seed}"
        assert bound.value <= schedule.frame_length <= min(run.slots for run in result.portfolio), f"seed {seed}"
        if schedule.frame_length == bound.value:
            assert result.status == PROVEN_OPTIMAL_BY_BOUND, f"seed {seed}"
        else:  # a gap: the exact search ran, and either proved the answer or gave up
            assert result.status in (PROVEN_OPTIMAL_BY_SEARCH, BEST_FOUND), f"seed {seed}"
            assert result.exact_start is not None, f"seed {seed}"
        assert schedule.node_to_slot["Node_01"] == 0 and len(result.portfolio) == 5, f"seed {seed}"
    assert time.perf_counter() - started < 60
