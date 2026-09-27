"""Shared test helpers (created in Division 2)."""
import itertools
import math
import random

from tdma_brain.io_utils import natural_key, parse_coordinates
from tdma_brain.schedule import Schedule
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


def neighbours_by_brute_force(coords, radio_range):
    """{name: set of names within radio range}, straight from the definition (no networkx)."""
    return {
        a: {b for b in coords if b != a and math.dist(coords[a], coords[b]) <= radio_range + 1e-9}
        for a in coords
    }


def hops_by_brute_force(coords, radio_range):
    """{frozenset({a, b}): 1 or 2} for every pair within 2 hops, straight from the definition.

    No networkx and no BFS: a pair is 1 hop when the distance is <= range (+1e-9) and 2 hops
    when it is not linked but some third radio hears both. This is the third, independent
    path the verifier and the schedulers are checked against.
    """
    heard = neighbours_by_brute_force(coords, radio_range)
    hops = {}
    for a, b in itertools.combinations(coords, 2):
        if b in heard[a]:
            hops[frozenset((a, b))] = 1
        elif heard[a] & heard[b]:
            hops[frozenset((a, b))] = 2
    return hops


def line_graph(count=6, spacing=400.0):
    """Radios Node_1 .. Node_<count> on a line: each one hears only its direct neighbours."""
    return build_radio_graph({f"Node_{i}": (spacing * (i - 1), 0.0) for i in range(1, count + 1)})


def raw_schedule(node_to_slot, k=None, extra=None, algorithm="test"):
    """A Schedule built directly from {node: slot}, WITHOUT renumbering, with consistent tables.

    Unlike make_schedule() this keeps the slot numbers as given (empty slots included), so a
    test can inject exactly one fault into an otherwise well-formed schedule. `extra` is
    {node: [extra slots]}; the matrix and slot lists include those slots too.
    """
    nodes = sorted(node_to_slot, key=natural_key)
    extra = {n: list(slots) for n, slots in (extra or {}).items()}
    transmit = {n: {node_to_slot[n], *extra.get(n, ())} for n in nodes}
    k = k if k is not None else max(max(slots) for slots in transmit.values()) + 1
    return Schedule(
        nodes=nodes,
        node_to_slot=dict(node_to_slot),
        frame_length=k,
        slot_to_nodes=[[n for n in nodes if s in transmit[n]] for s in range(k)],
        matrix=[[1 if s in transmit[n] else 0 for n in nodes] for s in range(k)],
        algorithm=algorithm,
        extra_tx_slots=extra,
    )


def conflicts_by_brute_force(coords, radio_range, node_to_slot):
    """Pairs within 2 hops that share a slot: {frozenset({a, b}): 1 or 2 (hops)}."""
    return {
        pair: hops
        for pair, hops in hops_by_brute_force(coords, radio_range).items()
        if len({node_to_slot[n] for n in pair}) == 1
    }
