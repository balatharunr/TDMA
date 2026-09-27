"""Lower bounds on the number of slots (created in Division 3).

No schedule can use fewer slots than the largest set of radios that all conflict with one
another (a clique of the conflict graph H): each of them needs a slot of its own. That number
is a LOWER bound. When a schedule's frame length equals it, the schedule is provably optimal.
"""
from dataclasses import dataclass

import networkx as nx

from .io_utils import natural_key

MAX_CLIQUE_NODES = 300  # above this the exact clique search is skipped and only maxdeg+1 is used


@dataclass(frozen=True)
class LowerBound:
    """A proven lower bound on the frame length, and where it comes from."""

    value: int  # the bound: max(maxdeg_plus_1, clique_size)
    maxdeg_plus_1: int  # the largest number of neighbours of a radio in G, plus that radio itself
    clique_size: int | None  # size of a largest clique of H; None when the search was skipped (too many nodes)
    clique_nodes: tuple = ()  # one such clique, in natural order


def lower_bound(G, H):
    """Lower bound for colouring H (the conflict graph of the radio graph G).

    maxdeg+1: a radio and its neighbours are pairwise at most 2 hops apart, so they form a
    clique in H. clique: the exact maximum clique of H, which is never smaller. With more than
    300 nodes the clique search is skipped and the bound falls back to maxdeg+1.
    """
    # CONCEPT: Lower bounds: clique number and maxdeg+1 - a radio and all its neighbours are within 2 hops of one another, so they form a clique in H and every one of them needs its own slot
    maxdeg_plus_1 = max((degree for _, degree in G.degree()), default=-1) + 1  # 0 for an empty graph, 1 for isolated radios
    if H.number_of_nodes() > MAX_CLIQUE_NODES:
        return LowerBound(maxdeg_plus_1, maxdeg_plus_1, None, ())
    # CONCEPT: Maximum clique problem - the exact size of the largest group of radios that all conflict with each other; no schedule can use fewer slots than that
    clique, size = nx.max_weight_clique(H, weight=None)
    return LowerBound(max(maxdeg_plus_1, size), maxdeg_plus_1, size, tuple(sorted(clique, key=natural_key)))
