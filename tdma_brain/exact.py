"""Exact colouring search (created in Division 3).

A DSATUR-ordered branch and bound that answers "can the conflict graph H be coloured with k
slots?". solve_exact() only ever asks for k between the proven lower bound and one less than the
best frame length the heuristics found, and it gives up when the time limit runs out. Finding
the fewest slots is NP-hard, so for large hard graphs the search may not finish; the answer
is then honestly reported as BEST_FOUND together with the gap to the lower bound.
"""
import math
import sys
import time

from .io_utils import InputError, natural_key

PROVEN_OPTIMAL_BY_BOUND = "PROVEN_OPTIMAL_BY_BOUND"  # the frame length equals the lower bound
PROVEN_OPTIMAL_BY_SEARCH = "PROVEN_OPTIMAL_BY_SEARCH"  # the search showed that one slot fewer is impossible
BEST_FOUND = "BEST_FOUND"  # the time limit ran out first; the gap to the lower bound stays open


class SearchTimeout(Exception):
    """The time limit ran out before the search could decide."""


def validate_time_limit(seconds):
    """Raise InputError unless the time limit is a finite number of seconds, 0 or more."""
    if not (math.isfinite(seconds) and seconds >= 0):
        raise InputError(f"time limit must be a finite number of seconds, 0 or more, got {seconds}")


def solve_exact(H, lb, incumbent, time_limit_s=10.0):
    """Try to beat `incumbent` (a proper colouring of H); returns (colouring, status).

    `lb` is a proven lower bound on the number of slots. The search asks for K-1, K-2, ... slots
    (K = the incumbent's slot count) and never for fewer than `lb`. Status:
      PROVEN_OPTIMAL_BY_BOUND   the result has exactly `lb` slots (no search was needed, or it got there);
      PROVEN_OPTIMAL_BY_SEARCH  the search proved that one slot fewer than the result is impossible;
      BEST_FOUND                the time limit ran out first; the result is the best colouring found.
    """
    validate_time_limit(time_limit_s)
    deadline = time.perf_counter() + time_limit_s
    best = dict(incumbent)
    if set(best) != set(H.nodes) or any(best[u] == best[v] for u, v in H.edges):
        raise ValueError("the incumbent must be a proper colouring of every node of the graph")
    slots = _slot_count(best)
    if slots < lb:
        raise ValueError(f"the lower bound {lb} is above the {slots} slots of a valid colouring: the bound is wrong")
    try:
        # CONCEPT: Branch and bound + symmetry breaking - the loop only asks for K-1, K-2, ... lb slots and NEVER for fewer than the lower bound: below it no colouring can exist, so searching there would only burn time
        while slots > lb:
            found = colourable(H, slots - 1, deadline)
            if found is None:
                return best, PROVEN_OPTIMAL_BY_SEARCH  # slots-1 is impossible, so `slots` is optimal
            best, slots = found, _slot_count(found)
    except SearchTimeout:
        return best, BEST_FOUND
    return best, PROVEN_OPTIMAL_BY_BOUND  # we got down to the lower bound


def colourable(H, k, deadline):
    """A proper colouring of H with at most k slots, or None if there is none.

    `deadline` is a time.perf_counter() value; SearchTimeout is raised once it has passed.
    Backtracking: at each step colour the radio picked by the DSATUR rule (largest saturation,
    then most uncoloured neighbours, then natural order) with each slot no neighbour uses.
    """
    nodes = sorted(H.nodes, key=natural_key)
    rank = {v: i for i, v in enumerate(nodes)}
    adjacent = {v: list(H[v]) for v in nodes}
    around = {v: [0] * k for v in nodes}  # around[v][c]: how many neighbours of v already sit in slot c
    saturation = {v: 0 for v in nodes}  # how many distinct slots are around v
    uncoloured_degree = {v: len(adjacent[v]) for v in nodes}
    colour, todo = {}, set(nodes)

    def place(v, c):
        colour[v] = c
        todo.remove(v)
        for u in adjacent[v]:
            if around[u][c] == 0:
                saturation[u] += 1
            around[u][c] += 1
            uncoloured_degree[u] -= 1

    def remove(v, c):
        del colour[v]
        todo.add(v)
        for u in adjacent[v]:
            around[u][c] -= 1
            if around[u][c] == 0:
                saturation[u] -= 1
            uncoloured_degree[u] += 1

    def search(used):  # `used` slots (0..used-1) are open so far
        if not todo:
            return True
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        v = max(todo, key=lambda x: (saturation[x], uncoloured_degree[x], -rank[x]))
        # CONCEPT: Branch and bound + symmetry breaking - c <= used opens at most ONE brand-new slot per step, because all unopened slots are interchangeable; that removes almost all of the k! renamings of the same colouring
        for c in range(min(k, used + 1)):
            if around[v][c] == 0:
                place(v, c)
                if search(max(used, c + 1)):
                    return True
                remove(v, c)
        return False

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, len(nodes) + 100))  # the search is one call deeper per radio
    try:
        return dict(colour) if search(0) else None
    finally:
        sys.setrecursionlimit(previous_limit)


def _slot_count(colouring):
    return len(set(colouring.values()))
