"""Golden reference data: TDMA_IMPLEMENTATION_GUIDE.md section 5.1 (VERIFIED ground truth).

These numbers were computed independently of our code, with exact colouring (exhaustive
branch-and-bound) and exact max-clique search. Radio range is 500 m and a link means
distance <= 500 m (inclusive). If our code disagrees with a number here, the code is wrong:
never edit a value, loosen an assertion or skip a test to make it pass.

Created in Division 1. D1 tests use the first five numeric columns; `max_clique` (the D3
lower bound) and `optimal_slots` (D2/D3 slot counts) are recorded now so later divisions
share one copy of the table.
"""
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


class Golden(NamedTuple):
    name: str             # file stem in examples/
    nodes: int
    links: int            # |E(G)|: pairs within radio range (1 hop)
    conflict_pairs: int   # |E(G^2)|: pairs at most 2 hops apart
    hidden_pairs: int     # pairs exactly 2 hops apart (= conflict_pairs - links)
    maxdeg_plus_1: int    # max degree of G + 1
    max_clique: int       # omega(G^2), the lower bound used from D3
    optimal_slots: int    # chi(G^2), the provably minimal number of slots


GOLDEN = (
    #      name                nodes links conf hidden maxdeg+1 clique optimal
    Golden("grid16_300m",        16,   42,  90,   48,     9,       9,     9),
    Golden("grid16_400m",        16,   24,  58,   34,     5,       5,     5),
    Golden("grid16_250m",        16,   58, 110,   52,    11,      12,    12),
    Golden("line16_400m",        16,   15,  29,   14,     3,       3,     3),
    Golden("cluster16_30m",      16,  120, 120,    0,    16,      16,    16),
    Golden("isolated16_1000m",   16,    0,   0,    0,     1,       1,     1),
    Golden("ring16_r1150m",      16,   16,  32,   16,     3,       3,     4),
    Golden("two_clusters16",     16,   56,  56,    0,     8,       8,     8),
    Golden("hidden_terminal3",    3,    2,   3,    1,     3,       3,     3),
)


def example_path(name: str) -> Path:
    """Path of examples/<name>.json."""
    return EXAMPLES_DIR / f"{name}.json"
