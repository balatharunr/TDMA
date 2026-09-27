"""Shared test helpers (created in Division 2)."""
import itertools
import math
import random

from tdma_brain.io_utils import parse_coordinates
from tdma_brain.topology import build_conflict_graph, build_radio_graph
from tests.golden import example_path


def load_example(name):
    """Coordinates of examples/<name>.json."""
    return parse_coordinates(example_path(name).read_text(encoding="utf-8"))


def build_graphs(name, radio_range=500.0):
    """(G, H) for examples/<name>.json."""
    G = build_radio_graph(load_example(name), radio_range)
    return G, build_conflict_graph(G)


def random_layout(seed, max_nodes=25):
    """A seeded random deployment: ({name: (x, y)}, radio_range)."""
    rng = random.Random(seed)
    n = rng.randint(2, max_nodes)
    side = rng.choice([300.0, 800.0, 1500.0, 3000.0])
    radio_range = rng.choice([250.0, 500.0])
    coords = {
        f"Node_{i:02d}": (round(rng.uniform(0, side), 1), round(rng.uniform(0, side), 1))
        for i in range(1, n + 1)
    }
    return coords, radio_range


def hops_by_brute_force(coords, radio_range):
    """{frozenset({a, b}): 1 or 2} for every pair within 2 hops, straight from the definition.

    No networkx and no BFS: a pair is 1 hop when the distance is <= range (+1e-9) and 2 hops
    when it is not linked but some third radio hears both. This is the third, independent
    path the verifier and the schedulers are checked against.
    """
    names = list(coords)
    linked = {
        frozenset(pair)
        for pair in itertools.combinations(names, 2)
        if math.dist(coords[pair[0]], coords[pair[1]]) <= radio_range + 1e-9
    }
    heard = {a: {b for b in names if frozenset((a, b)) in linked} for a in names}
    hops = {}
    for a, b in itertools.combinations(names, 2):
        pair = frozenset((a, b))
        if pair in linked:
            hops[pair] = 1
        elif heard[a] & heard[b]:
            hops[pair] = 2
    return hops


def conflicts_by_brute_force(coords, radio_range, node_to_slot):
    """Pairs within 2 hops that share a slot: {frozenset({a, b}): 1 or 2 (hops)}."""
    return {
        pair: hops
        for pair, hops in hops_by_brute_force(coords, radio_range).items()
        if len({node_to_slot[n] for n in pair}) == 1
    }
