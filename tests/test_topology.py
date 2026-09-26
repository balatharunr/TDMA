"""Division 1 tests: radio graph G, conflict graph H = G^2, hidden terminals, summary text."""
import itertools
import math
import random
import re

import networkx as nx
import pytest

from tdma_brain.io_utils import InputError, natural_key, parse_coordinates
from tdma_brain.topology import (
    build_conflict_graph,
    build_radio_graph,
    format_topology_summary,
    hidden_terminal_pairs,
    topology_summary,
)
from tests.golden import GOLDEN, example_path


def load(name):
    return parse_coordinates(example_path(name).read_text(encoding="utf-8"))


def build(name, radio_range=500.0):
    G = build_radio_graph(load(name), radio_range)
    return G, build_conflict_graph(G)


def pair_set(edges):
    return {frozenset(edge) for edge in edges}


def hop_pairs_by_bfs(G):
    """Pairs whose hop distance in G is 1 or 2, found by BFS (independent of nx.power)."""
    pairs = set()
    for v in G:
        reachable = nx.single_source_shortest_path_length(G, v, cutoff=2)
        pairs |= {frozenset((v, u)) for u, hops in reachable.items() if hops in (1, 2)}
    return pairs


# --------------------------------------------------------------------------- golden counts
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_golden_counts(golden):
    G, H = build(golden.name)
    summary = topology_summary(G, H)
    assert G.number_of_nodes() == golden.nodes
    assert G.number_of_edges() == golden.links
    assert H.number_of_edges() == golden.conflict_pairs
    assert len(hidden_terminal_pairs(G, H)) == golden.hidden_pairs
    assert summary["max_degree"] + 1 == golden.maxdeg_plus_1
    # the table's own consistency: conflicts = direct links + hidden-terminal pairs
    assert golden.conflict_pairs == golden.links + golden.hidden_pairs


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_conflict_graph_equals_bfs_pairs_within_2_hops(golden):
    G, H = build(golden.name)
    assert pair_set(H.edges) == hop_pairs_by_bfs(G)
    assert set(H.nodes) == set(G.nodes)
    assert nx.number_of_selfloops(H) == 0


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_conflict_edge_annotations(golden):
    G, H = build(golden.name)
    one_hop = two_hop = 0
    for u, v, data in H.edges(data=True):
        if G.has_edge(u, v):
            one_hop += 1
            assert data["kind"] == "1-hop" and "via" not in data
        else:
            two_hop += 1
            expected_via = sorted((w for w in G if G.has_edge(u, w) and G.has_edge(v, w)), key=natural_key)
            assert data["kind"] == "2-hop"
            assert data["via"] == expected_via and expected_via  # correct, natural-sorted, never empty
    assert (one_hop, two_hop) == (golden.links, golden.hidden_pairs)


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_hidden_terminal_pairs_are_normalised_and_sorted(golden):
    G, H = build(golden.name)
    pairs = hidden_terminal_pairs(G, H)
    assert len(pairs) == golden.hidden_pairs
    for u, v, via in pairs:
        assert natural_key(u) < natural_key(v)
        assert not G.has_edge(u, v) and via
    names = [(u, v) for u, v, _ in pairs]
    assert names == sorted(names, key=lambda uv: (natural_key(uv[0]), natural_key(uv[1])))
    assert len(set(names)) == len(names)  # no pair is listed twice


def test_hidden_terminal_pairs_do_not_depend_on_the_annotations():
    G, H = build("grid16_300m")
    assert hidden_terminal_pairs(G, nx.power(G, 2)) == hidden_terminal_pairs(G, H)


def test_hidden_terminal3_only_two_hop_pair_is_a_and_c_via_b():
    G, H = build("hidden_terminal3")
    assert pair_set(G.edges) == {frozenset(("Node_A", "Node_B")), frozenset(("Node_B", "Node_C"))}
    assert hidden_terminal_pairs(G, H) == [("Node_A", "Node_C", ["Node_B"])]
    assert H.edges["Node_A", "Node_C"]["kind"] == "2-hop"
    assert H.edges["Node_A", "Node_B"]["kind"] == "1-hop"


def test_grid_300m_first_hidden_pair_matches_the_guide():
    # guide, Division 1 expected output: "Node_01 <-> Node_03 via Node_02, Node_06"
    G, H = build("grid16_300m")
    assert hidden_terminal_pairs(G, H)[0] == ("Node_01", "Node_03", ["Node_02", "Node_06"])


def test_building_the_conflict_graph_leaves_g_untouched():
    G, _ = build("grid16_300m")
    assert all("kind" not in data and "via" not in data for _, _, data in G.edges(data=True))


# --------------------------------------------------------------------------- the range rule
def two_nodes(x, radio_range=500.0):
    return build_radio_graph({"Node_01": (0.0, 0.0), "Node_02": (x, 0.0)}, radio_range)


def test_range_boundary_is_inclusive():
    assert two_nodes(500.0).number_of_edges() == 1  # exactly at the range: a link
    assert two_nodes(499.999).number_of_edges() == 1
    assert two_nodes(500.001).number_of_edges() == 0  # just beyond: no link


def test_tolerance_of_1e_minus_9():
    assert two_nodes(500.0000000005).number_of_edges() == 1  # inside range + 1e-9
    assert two_nodes(500.0 + 1e-9).number_of_edges() == 1  # exactly on the limit: the guide's rule is '<='
    assert two_nodes(500.000000002).number_of_edges() == 0  # outside it


def test_tolerance_absorbs_float_noise():
    # 500 m at 1 degree by construction; float rounding makes hypot() a hair over 500.0
    x, y = 499.92384757819565, 8.726203218641755
    assert math.hypot(x, y) > 500.0  # precondition: without the tolerance this pair would be lost
    G = build_radio_graph({"Node_01": (0.0, 0.0), "Node_02": (x, y)})
    assert G.number_of_edges() == 1


def test_custom_radio_range_changes_the_links():
    assert build("grid16_300m", 250.0)[0].number_of_edges() == 0  # neighbours are 300 m apart
    assert build("grid16_300m", 300.0)[0].number_of_edges() == 24  # only the horizontal and vertical neighbours
    assert build("grid16_300m", 500.0)[0].number_of_edges() == 42  # plus the 424 m diagonals


def test_grid_250m_links_pairs_at_exactly_500m():
    G, _ = build("grid16_250m")
    assert G.has_edge("Node_01", "Node_03")  # (0, 0) and (500, 0): exactly 500 m
    assert G.edges["Node_01", "Node_03"]["distance"] == 500.0


@pytest.mark.parametrize("bad", [0, 0.0, -1, -500.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_radio_range_is_an_input_error(bad):
    with pytest.raises(InputError, match="radio range"):
        build_radio_graph({"Node_01": (0.0, 0.0)}, bad)


# --------------------------------------------------------------------------- graph shape
def test_nodes_are_added_in_natural_order_whatever_the_input_order():
    coords = {"Node_10": (0.0, 0.0), "Node_2": (1.0, 0.0), "Node_1": (2.0, 0.0), "Node_A": (3.0, 0.0)}
    G = build_radio_graph(coords)
    assert list(G.nodes) == ["Node_1", "Node_2", "Node_10", "Node_A"]
    assert list(build_conflict_graph(G).nodes) == list(G.nodes)


def test_node_and_edge_attributes():
    G, _ = build("grid16_300m")
    assert G.nodes["Node_06"]["pos"] == (300.0, 300.0)
    assert G.edges["Node_01", "Node_02"]["distance"] == 300.0
    assert G.edges["Node_01", "Node_06"]["distance"] == pytest.approx(300 * math.sqrt(2))
    assert G.graph["radio_range"] == 500.0
    assert not G.has_edge("Node_01", "Node_03")  # 600 m apart


def test_same_input_gives_the_same_graph_every_time():
    a, b = build("grid16_300m"), build("grid16_300m")
    assert list(a[0].edges) == list(b[0].edges)
    assert hidden_terminal_pairs(*a) == hidden_terminal_pairs(*b)


# --------------------------------------------------------------------------- summary
def test_summary_of_the_official_grid():
    G, H = build("grid16_300m")
    assert topology_summary(G, H) == {
        "nodes": 16,
        "radio_range": 500.0,
        "links": 42,
        "conflict_pairs": 90,
        "hidden_terminal_pairs": 48,
        "max_degree": 8,
        "max_degree_nodes": ["Node_06", "Node_07", "Node_10", "Node_11"],
        "components": 1,
        "isolated_nodes": [],
        "example_hidden_terminal": ("Node_01", "Node_03", ["Node_02", "Node_06"]),
    }


def test_summary_components_and_isolated_nodes():
    assert topology_summary(*build("two_clusters16"))["components"] == 2
    assert topology_summary(*build("ring16_r1150m"))["components"] == 1
    isolated = topology_summary(*build("isolated16_1000m"))
    assert isolated["components"] == 16
    assert isolated["isolated_nodes"] == [f"Node_{i:02d}" for i in range(1, 17)]
    assert isolated["max_degree"] == 0 and len(isolated["max_degree_nodes"]) == 16
    assert isolated["example_hidden_terminal"] is None
    assert topology_summary(*build("cluster16_30m"))["isolated_nodes"] == []


def test_summary_max_degree_examples():
    assert topology_summary(*build("cluster16_30m"))["max_degree"] == 15
    line = topology_summary(*build("line16_400m"))
    assert line["max_degree"] == 2
    assert line["max_degree_nodes"] == [f"Node_{i:02d}" for i in range(2, 16)]  # every node except the two ends


GRID_300M_TEXT = """\
TOPOLOGY SUMMARY
  Nodes                      : 16
  Radio range                : 500.0 m (link if distance <= range)
  Radio links (1-hop)        : 42
  Conflict pairs (<= 2 hops) : 90
  Hidden-terminal pairs      : 48 (2-hop only)
  Max degree                 : 8 (Node_06, Node_07, Node_10, Node_11)
  Connected components       : 1
  Isolated nodes             : none
  Example hidden-terminal    : Node_01 <-> Node_03 via Node_02, Node_06"""

HIDDEN_TERMINAL3_TEXT = """\
TOPOLOGY SUMMARY
  Nodes                      : 3
  Radio range                : 500.0 m (link if distance <= range)
  Radio links (1-hop)        : 2
  Conflict pairs (<= 2 hops) : 3
  Hidden-terminal pairs      : 1 (2-hop only)
  Max degree                 : 2 (Node_B)
  Connected components       : 1
  Isolated nodes             : none
  Example hidden-terminal    : Node_A <-> Node_C via Node_B"""


def test_formatted_summary_matches_the_guide_layout():
    assert format_topology_summary(topology_summary(*build("grid16_300m"))) == GRID_300M_TEXT
    assert format_topology_summary(topology_summary(*build("hidden_terminal3"))) == HIDDEN_TERMINAL3_TEXT


def test_formatted_summary_shortens_long_name_lists():
    line = format_topology_summary(topology_summary(*build("line16_400m")))
    assert "Max degree                 : 2 (Node_02, Node_03, Node_04, Node_05, Node_06, Node_07, Node_08, Node_09, ... (+6 more))" in line
    isolated = format_topology_summary(topology_summary(*build("isolated16_1000m")))
    assert "Max degree                 : 0 (all 16 nodes)" in isolated
    assert "Isolated nodes             : all 16 nodes" in isolated
    assert "Example hidden-terminal    : none" in isolated


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_formatted_summary_shows_the_golden_numbers_in_plain_ascii(golden):
    text = format_topology_summary(topology_summary(*build(golden.name)))
    text.encode("ascii")  # the Windows console must never meet a non-ASCII character
    assert all(line == line.rstrip() for line in text.splitlines())  # no trailing whitespace

    def value(label):
        return re.search(rf"^  {re.escape(label)}\s*: (\d+)", text, re.MULTILINE).group(1)

    assert int(value("Nodes")) == golden.nodes
    assert int(value("Radio links (1-hop)")) == golden.links
    assert int(value("Conflict pairs (<= 2 hops)")) == golden.conflict_pairs
    assert int(value("Hidden-terminal pairs")) == golden.hidden_pairs
    assert int(value("Max degree")) + 1 == golden.maxdeg_plus_1


# --------------------------------------------------------------------------- random layouts
def test_random_layouts_match_an_independent_brute_force():
    """200 seeded random layouts: links, 2-hop conflicts, kind and via straight from the definitions."""
    for seed in range(200):
        rng = random.Random(seed)
        n = rng.randint(2, 25)
        side = rng.choice([300.0, 800.0, 1500.0, 3000.0])
        radio_range = rng.choice([250.0, 500.0])
        coords = {
            f"Node_{i:02d}": (round(rng.uniform(0, side), 1), round(rng.uniform(0, side), 1))
            for i in range(1, n + 1)
        }
        names = list(coords)
        G = build_radio_graph(coords, radio_range)
        H = build_conflict_graph(G)

        linked = {
            frozenset((a, b))
            for a, b in itertools.combinations(names, 2)
            if math.dist(coords[a], coords[b]) <= radio_range + 1e-9
        }
        heard = {a: {b for b in names if frozenset((a, b)) in linked} for a in names}
        conflicts = {
            frozenset((a, b))
            for a, b in itertools.combinations(names, 2)
            if frozenset((a, b)) in linked or heard[a] & heard[b]
        }
        assert pair_set(G.edges) == linked, f"seed {seed}: links differ"
        assert pair_set(H.edges) == conflicts, f"seed {seed}: conflict pairs differ"
        for a, b, data in H.edges(data=True):
            if frozenset((a, b)) in linked:
                assert data["kind"] == "1-hop", f"seed {seed}"
            else:
                assert data["kind"] == "2-hop", f"seed {seed}"
                assert data["via"] == sorted(heard[a] & heard[b], key=natural_key), f"seed {seed}"
        assert len(hidden_terminal_pairs(G, H)) == len(conflicts) - len(linked), f"seed {seed}"
