"""Colouring heuristics that turn a conflict graph into time slots (Divisions 2 and 3).

A colouring gives every radio a slot number so that radios joined by an edge never share
one. Everything here is deterministic: ties are broken by natural node order (Node_2 before
Node_10), the only randomness is a seeded random.Random, and no Python hash order is used, so
the same input gives the same output on every run.
"""
import random

from .io_utils import natural_key

DSATUR_NAME = "DSATUR"
RLF_NAME = "RLF"
SMALLEST_LAST_NAME = "smallest_last"
LARGEST_FIRST_NAME = "largest_first (Welsh-Powell)"
GREEDY_NAME = "greedy (natural order)"
NAIVE_D1_NAME = "naive-d1"
# the values of the CLI's --algorithm (besides "auto"); the first five are the optimizer's portfolio
ALGORITHM_KEYS = ("dsatur", "rlf", "smallest_last", "largest_first", "greedy", "naive-d1")


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


def largest_first(H):
    """Welsh-Powell (1967): first-fit with the most-conflicted radios (highest H-degree) first.

    Ties in degree go to the earlier node in natural order.
    """
    # CONCEPT: Welsh-Powell (largest-first) / smallest-last orderings - radios with the most conflicts are placed first, while the frame still has plenty of free slots
    order = sorted(H.nodes, key=lambda v: (-H.degree(v), natural_key(v)))
    return greedy_first_fit(H, order)


def smallest_last(H):
    """Matula-Beck (1983) smallest-last (degeneracy) ordering, then first-fit.

    Repeatedly take out the radio with the fewest conflicts among those left (ties: earliest
    in natural order); then colour in the reverse of the removal order.
    """
    rank = {v: i for i, v in enumerate(sorted(H.nodes, key=natural_key))}
    degree = {v: H.degree(v) for v in rank}
    remaining = set(rank)
    removal = []
    while remaining:
        # CONCEPT: Welsh-Powell (largest-first) / smallest-last orderings - peel off the least-connected radio again and again; the radios peeled last are the densest core and are coloured first
        v = min(remaining, key=lambda x: (degree[x], rank[x]))
        removal.append(v)
        remaining.remove(v)
        for u in H[v]:
            if u in remaining:
                degree[u] -= 1
    return greedy_first_fit(H, reversed(removal))


def rlf(H):
    """Recursive Largest First (Leighton 1979): build one slot at a time, packing it full.

    The first radio of a slot is the one with the most uncoloured neighbours. Then, among the
    radios that can still join, take the one with the most neighbours in the blocked set (the
    radios that conflict with the slot and must wait), then the one with the fewest neighbours
    among the remaining candidates; ties go to the earlier node in natural order.
    Returns {node: slot} with slots numbered in the order they were built.
    """
    rank = {v: i for i, v in enumerate(sorted(H.nodes, key=natural_key))}
    neighbours = {v: set(H[v]) for v in rank}
    uncoloured = set(rank)
    colour = {}
    slot = 0
    while uncoloured:
        candidates = set(uncoloured)  # X: may still join this slot
        blocked = set()  # Y: conflict with a radio already in this slot, so they wait for a later slot
        v = max(candidates, key=lambda x: (len(neighbours[x] & uncoloured), -rank[x]))
        members = []
        while True:
            members.append(v)
            # CONCEPT: RLF (Recursive Largest First) - v joins the slot; its neighbours can no longer join it, so they move from the candidates X into the blocked set Y
            blocked |= neighbours[v] & candidates
            candidates -= {v} | neighbours[v]
            if not candidates:
                break
            v = max(candidates, key=lambda x: (len(neighbours[x] & blocked), -len(neighbours[x] & candidates), -rank[x]))
        for member in members:
            colour[member] = slot
        uncoloured -= set(members)
        slot += 1
    return colour


def iterated_greedy_passes(H, start, iterations=300, seed=42):
    """Culberson and Luo's iterated greedy (1996), yielding the colouring of every pass.

    Each pass re-runs first-fit with the colour classes of the previous colouring kept
    together in a new order (largest first, reversed, or shuffled by a seeded random.Random).
    A theorem says a pass never needs more slots than the colouring it started from; this
    raises RuntimeError if that ever fails, which would mean a bug.
    """
    rng = random.Random(seed)
    current = dict(start)
    for _ in range(iterations):
        slots = sorted(set(current.values()))
        classes = [sorted((v for v in current if current[v] == s), key=natural_key) for s in slots]
        # CONCEPT: Iterated greedy (Culberson) - first-fit fed with WHOLE colour classes in a new order can never need more slots, and a lucky order often needs one fewer
        r = rng.random()
        if r < 0.5:
            classes.sort(key=len, reverse=True)  # largest class first; the sort is stable, so ties keep the lower slot first
        elif r < 0.8:
            classes.reverse()
        else:
            rng.shuffle(classes)
        new = greedy_first_fit(H, [v for group in classes for v in group])
        if len(set(new.values())) > len(set(current.values())):
            raise RuntimeError("iterated greedy used more slots than its input: this contradicts Culberson's theorem")
        current = new
        yield new


def iterated_greedy(H, start, iterations=300, seed=42):
    """The best colouring seen during iterated_greedy_passes(); never worse than `start`."""
    best = dict(start)
    best_slots = len(set(best.values()))
    for colouring in iterated_greedy_passes(H, start, iterations, seed):
        if len(set(colouring.values())) < best_slots:
            best, best_slots = colouring, len(set(colouring.values()))
    return best


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
    if key == "rlf":
        return RLF_NAME, rlf(H)
    if key == "smallest_last":
        return SMALLEST_LAST_NAME, smallest_last(H)
    if key == "largest_first":
        return LARGEST_FIRST_NAME, largest_first(H)
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
