"""The optimizer: portfolio -> polish -> lower bound -> exact search (created in Division 3).

Runs every heuristic in a fixed priority order and keeps the shortest frame, polishes it with
iterated greedy, computes the lower bound, and only if a gap remains lets the time-limited
exact search try to close it. The outcome is a verified Schedule plus the proof status:
provably optimal, or the best found together with the size of the open gap.
"""
import time
from dataclasses import dataclass

from .bounds import LowerBound, lower_bound
from .exact import PROVEN_OPTIMAL_BY_BOUND, solve_exact, validate_time_limit
from .heuristics import iterated_greedy, run_algorithm
from .schedule import Schedule, make_schedule
from .verify import VerificationResult, verify_schedule

PORTFOLIO_KEYS = ("dsatur", "rlf", "smallest_last", "largest_first", "greedy")  # priority order: on a tie the earlier wins
ITERATED_GREEDY_ITERATIONS = 300


@dataclass(frozen=True)
class HeuristicRun:
    """One line of the portfolio table."""

    name: str
    slots: int
    ms: float


@dataclass
class OptimizationResult:
    """Everything the optimizer found out, for the report, --verbose and schedule.json."""

    schedule: Schedule  # the final canonical schedule
    verification: VerificationResult  # its independent verification
    colouring: dict  # the winning colouring before canonical renumbering, {node: slot}
    portfolio: list  # HeuristicRun per heuristic, in priority order
    best_name: str  # the portfolio winner
    best_slots: int  # its slot count
    iterations: int  # iterated-greedy iterations run on it
    polished_slots: int  # slot count after iterated greedy
    lower_bound: LowerBound
    status: str  # PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH or BEST_FOUND
    exact_start: int | None  # slot count handed to the exact search, or None if no search was needed
    node_count: int
    conflict_pairs: int
    total_ms: float


def optimize(G, H, time_limit=10.0, seed=42):
    """Find a short conflict-free frame for the radio graph G (conflict graph H) and prove how good it is.

    `time_limit` (seconds) only bounds the exact search; `seed` drives the iterated greedy.
    """
    validate_time_limit(time_limit)
    started = time.perf_counter()

    portfolio, colourings = [], []
    for key in PORTFOLIO_KEYS:
        run_started = time.perf_counter()
        name, colouring = run_algorithm(key, G, H)
        portfolio.append(HeuristicRun(name, _slots(colouring), (time.perf_counter() - run_started) * 1000))
        colourings.append(colouring)
    # CONCEPT: Algorithm portfolio - no single heuristic always wins (on grid16_400m first-fit needs 6 slots, DSATUR 5), so run them all, keep the shortest frame, and let the earlier one in the priority list win a tie
    winner = min(range(len(portfolio)), key=lambda i: (portfolio[i].slots, i))
    name, colouring = portfolio[winner].name, colourings[winner]
    best_slots = portfolio[winner].slots

    polished = iterated_greedy(H, colouring, ITERATED_GREEDY_ITERATIONS, seed)
    if _slots(polished) < best_slots:
        name += " + iterated greedy"
    colouring = polished
    polished_slots = _slots(colouring)

    bound = lower_bound(G, H)
    if polished_slots < bound.value:
        raise RuntimeError(f"a schedule with {polished_slots} slots is below the proven lower bound {bound.value}: a bug")

    exact_start = None
    status = PROVEN_OPTIMAL_BY_BOUND  # if the frame already equals the lower bound, that is the proof
    if polished_slots > bound.value:
        exact_start = polished_slots
        improved, status = solve_exact(H, bound.value, colouring, time_limit)
        if _slots(improved) < polished_slots:
            name += " + exact search"
        colouring = improved

    schedule = make_schedule(colouring, list(G.nodes), name)
    return OptimizationResult(
        schedule=schedule,
        verification=verify_schedule(G, schedule),
        colouring=colouring,
        portfolio=portfolio,
        best_name=portfolio[winner].name,
        best_slots=best_slots,
        iterations=ITERATED_GREEDY_ITERATIONS,
        polished_slots=polished_slots,
        lower_bound=bound,
        status=status,
        exact_start=exact_start,
        node_count=H.number_of_nodes(),
        conflict_pairs=H.number_of_edges(),
        total_ms=(time.perf_counter() - started) * 1000,
    )


def _slots(colouring):
    return len(set(colouring.values()))
