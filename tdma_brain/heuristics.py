"""Colouring heuristics that turn a conflict graph into time slots (created in Division 2).

A colouring gives every radio a slot number so that radios joined by an edge never share
one. Everything here is deterministic: ties are broken by natural node order (Node_2 before
Node_10) and no Python hash order is used, so the same input gives the same output on every
run. Division 3 adds the other heuristics (RLF, smallest-last, ...) to this module.
"""
from .io_utils import natural_key

DSATUR_NAME = "DSATUR"
GREEDY_NAME = "greedy (natural order)"
NAIVE_D1_NAME = "naive-d1"
ALGORITHM_KEYS = ("dsatur", "greedy", "naive-d1")  # the values of the CLI's --algorithm


def greedy_first_fit(H, order):
    """Baseline colouring: walk the nodes in `order`, give each the smallest free slot.

    A slot is free for a node when no already-coloured H-neighbour uses it. `order` must
    list every node of H exactly once. Returns {node: slot} with slots numbered from 0.
    """
    order = list(order)
    if sorted(order, key=natural_key) != sorted(H.nodes, key=natural_key):
        raise ValueError("order must list every node of the graph exactly once")
    colour = {}
    for v in order:
        colour[v] = _smallest_free({colour[u] for u in H[v] if u in colour})
    return colour


def dsatur(H):
    """DSATUR (Brelaz 1979) on the conflict graph H: always colour the most constrained radio.

    sat(v)  = number of DISTINCT slots already used by v's coloured H-neighbours,
    udeg(v) = number of v's H-neighbours that are still uncoloured.
    Pick the uncoloured v with the largest (sat, udeg), remaining ties go to the earliest
    node in natural order, and give it the smallest slot no coloured neighbour uses.
    Returns {node: slot}; run schedule.canonicalize() to number the slots by first use.
    """
    # CONCEPT: NP-hard problems (why heuristics) - finding the fewest slots is NP-hard, so DSATUR uses a fast greedy rule instead of trying every colouring
    rank = {v: i for i, v in enumerate(sorted(H.nodes, key=natural_key))}
    colour = {}
    neighbour_slots = {v: set() for v in rank}  # the distinct slots around v: its sat(v) is the size
    udeg = {v: H.degree(v) for v in rank}  # nothing is coloured yet, so every neighbour counts
    uncoloured = set(rank)
    while uncoloured:
        # CONCEPT: DSATUR algorithm - take the radio whose neighbours already use the most distinct slots, then the one with most uncoloured neighbours, then the earliest in natural order
        v = max(uncoloured, key=lambda x: (len(neighbour_slots[x]), udeg[x], -rank[x]))
        # CONCEPT: Distance-2 graph colouring - H is G^2, so a slot no H-neighbour uses is free of every 1-hop and 2-hop conflict
        colour[v] = _smallest_free(neighbour_slots[v])
        uncoloured.remove(v)
        for u in H[v]:
            udeg[u] -= 1
            neighbour_slots[u].add(colour[v])
    return colour


def naive_distance1(G):
    """INTENTIONALLY UNSAFE. FOR DEMOS AND THE DIVISION 8 EXPERIMENT ONLY.

    First-fit in natural order on the RADIO graph G instead of the conflict graph G^2. It
    stops radios that hear each other from sharing a slot, but it IGNORES HIDDEN TERMINALS:
    two radios that merely share a neighbour can get the same slot and collide at that
    neighbour. NEVER use it to plan a real schedule; the verifier is meant to reject it.
    """
    # CONCEPT: Collisions: primary (half-duplex) vs secondary (hidden-terminal) - colouring G alone removes the primary conflicts and leaves every secondary (2-hop) conflict in place
    return greedy_first_fit(G, sorted(G.nodes, key=natural_key))


def run_algorithm(key, G, H):
    """Run the heuristic behind a --algorithm key.

    Returns (name to show in the report and schedule.json, {node: slot}). Raises
    ValueError for an unknown key.
    """
    if key == "dsatur":
        return DSATUR_NAME, dsatur(H)
    if key == "greedy":
        return GREEDY_NAME, greedy_first_fit(H, sorted(H.nodes, key=natural_key))
    if key == "naive-d1":
        return NAIVE_D1_NAME, naive_distance1(G)
    raise ValueError(f"unknown algorithm {key!r}; choose from {', '.join(ALGORITHM_KEYS)}")


def _smallest_free(taken):
    """The smallest slot number (0, 1, 2, ...) that is not in `taken`."""
    # CONCEPT: Greedy first-fit colouring & vertex ordering - the smallest slot that no already-coloured neighbour uses
    slot = 0
    while slot in taken:
        slot += 1
    return slot
