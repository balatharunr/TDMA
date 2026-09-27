"""Division 3 tests: the lower bound (tdma_brain.bounds)."""
import dataclasses
import os
import subprocess
import sys

import networkx as nx
import pytest

from tdma_brain import bounds
from tdma_brain.bounds import MAX_CLIQUE_NODES, LowerBound, lower_bound
from tdma_brain.heuristics import dsatur, greedy_first_fit
from tdma_brain.io_utils import natural_key
from tdma_brain.topology import build_conflict_graph, build_radio_graph
from tests.golden import GOLDEN, REPO_ROOT
from tests.helpers import build_graphs, line_graph, random_layout


def is_clique(H, nodes):
    return all(H.has_edge(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:])


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_lower_bound_components_match_the_golden_table(golden):
    G, H = build_graphs(golden.name)
    bound = lower_bound(G, H)
    assert bound.maxdeg_plus_1 == golden.maxdeg_plus_1
    assert bound.clique_size == golden.max_clique
    assert bound.value == max(golden.maxdeg_plus_1, golden.max_clique) == golden.max_clique
    assert bound.value <= golden.optimal_slots  # a lower bound may never exceed the optimum
    assert (bound.value == golden.optimal_slots) == (golden.name != "ring16_r1150m")  # tight everywhere but the ring


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_the_reported_clique_is_a_real_clique_in_natural_order(golden):
    G, H = build_graphs(golden.name)
    bound = lower_bound(G, H)
    assert len(bound.clique_nodes) == bound.clique_size and isinstance(bound.clique_nodes, tuple)
    assert list(bound.clique_nodes) == sorted(bound.clique_nodes, key=natural_key)
    assert is_clique(H, list(bound.clique_nodes))


def test_grid_250m_clique_bound_beats_maxdeg_plus_one():
    G, H = build_graphs("grid16_250m")
    bound = lower_bound(G, H)
    assert (bound.maxdeg_plus_1, bound.clique_size, bound.value) == (11, 12, 12)  # guide 5.1


def test_grid_300m_proof_of_the_official_answer():
    # Node_06 and its 8 neighbours are pairwise within 2 hops: a clique of 9, so 9 slots are needed
    G, H = build_graphs("grid16_300m")
    bound = lower_bound(G, H)
    assert (bound.value, bound.maxdeg_plus_1, bound.clique_size) == (9, 9, 9)
    assert set(G["Node_06"]) | {"Node_06"} <= set(H.nodes) and is_clique(H, ["Node_06", *G["Node_06"]])


def test_ring_shows_that_the_bound_is_not_always_tight():
    G, H = build_graphs("ring16_r1150m")
    assert lower_bound(G, H).value == 3  # 4 slots are needed (chi = 4): the search of Division 3 has to prove it


def test_the_clique_size_agrees_with_an_independent_clique_enumeration():
    """200 random layouts: nx.max_weight_clique equals the largest maximal clique from Bron-Kerbosch."""
    for seed in range(200):
        coords, radio_range = random_layout(seed, max_nodes=25)
        G = build_radio_graph(coords, radio_range)
        H = build_conflict_graph(G)
        bound = lower_bound(G, H)
        assert bound.clique_size == max(len(c) for c in nx.find_cliques(H)), f"seed {seed}"
        assert bound.clique_size >= bound.maxdeg_plus_1, f"seed {seed}"  # the closed neighbourhood is a clique of H
        assert is_clique(H, list(bound.clique_nodes)), f"seed {seed}"


def test_no_proper_colouring_beats_the_bound():
    for seed in range(200):
        coords, radio_range = random_layout(seed, max_nodes=25)
        G = build_radio_graph(coords, radio_range)
        H = build_conflict_graph(G)
        floor = lower_bound(G, H).value
        for colouring in (dsatur(H), greedy_first_fit(H, sorted(H.nodes, key=natural_key))):
            assert len(set(colouring.values())) >= floor, f"seed {seed}"


def test_degenerate_graphs():
    empty = nx.Graph()
    assert lower_bound(empty, empty) == LowerBound(0, 0, 0, ())
    G = build_radio_graph({"Node_1": (0.0, 0.0)})
    assert lower_bound(G, build_conflict_graph(G)) == LowerBound(1, 1, 1, ("Node_1",))
    G, H = build_graphs("isolated16_1000m")
    bound = lower_bound(G, H)
    assert (bound.value, bound.maxdeg_plus_1, bound.clique_size) == (1, 1, 1)


def test_more_than_300_nodes_falls_back_to_maxdeg_plus_one_without_the_clique_search(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the clique search must be skipped for more than 300 nodes")

    monkeypatch.setattr(nx, "max_weight_clique", forbidden)
    G = line_graph(MAX_CLIQUE_NODES + 5)  # a long line: every radio hears at most 2 others
    bound = lower_bound(G, build_conflict_graph(G))
    assert bound == LowerBound(3, 3, None, ())


def test_exactly_300_nodes_still_gets_the_clique_search():
    G = line_graph(MAX_CLIQUE_NODES)
    bound = lower_bound(G, build_conflict_graph(G))
    assert bound.clique_size == 3 and bound.value == 3 and len(bound.clique_nodes) == 3


def test_a_lower_bound_is_immutable():
    G, H = build_graphs("hidden_terminal3")
    bound = lower_bound(G, H)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.value = 99
    assert bounds.MAX_CLIQUE_NODES == 300


def test_the_reported_clique_does_not_depend_on_python_hash_randomisation():
    code = (
        "from tests.helpers import build_graphs;"
        "from tdma_brain.bounds import lower_bound;"
        "G, H = build_graphs('grid16_400m'); b = lower_bound(G, H); print(b.value, b.clique_nodes)"
    )
    outputs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
                       env=dict(os.environ, PYTHONHASHSEED=seed)).stdout
        for seed in ("0", "1", "2", "999")
    }
    assert len(outputs) == 1 and "5 (" in outputs.pop()
