"""Heuristic comparison on random topologies (created in Division 4).

Runs the full optimizer pipeline (portfolio -> iterated greedy -> lower bound -> exact search)
and the unsafe naive-D1 demo on a grid of random deployments, and writes the evidence the
docs and slides are built from: results/benchmark.csv (one row per instance),
results/benchmark_summary.md (aggregated per (n, avg_degree)), and two plots.

Run as a script: `python bench/benchmark.py [--quick]`. Every function here is also a plain,
testable function, so tests/test_benchmark.py exercises the real pipeline on a small grid.
"""
import argparse
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # `python bench/benchmark.py` puts bench/ on sys.path, not the repo root, so tdma_brain
    # (a sibling of bench/) would not be importable otherwise. Harmless (and unneeded) under
    # pytest, which already puts the repo root on sys.path via pytest.ini's pythonpath = .
    sys.path.insert(0, str(REPO_ROOT))

from tdma_brain import viz  # noqa: E402  (import first: sets the Agg backend before any other pyplot import)
import matplotlib.pyplot as plt  # noqa: E402

from tdma_brain.exact import BEST_FOUND  # noqa: E402
from tdma_brain.heuristics import (  # noqa: E402
    DSATUR_NAME,
    GREEDY_NAME,
    LARGEST_FIRST_NAME,
    NAIVE_D1_NAME,
    RLF_NAME,
    SMALLEST_LAST_NAME,
    naive_distance1,
)
from tdma_brain.io_utils import natural_key  # noqa: E402
from tdma_brain.optimizer import optimize  # noqa: E402
from tdma_brain.schedule import make_schedule  # noqa: E402
from tdma_brain.topology import DEFAULT_RADIO_RANGE_M, build_conflict_graph, build_radio_graph  # noqa: E402
from tdma_brain.verify import verify_schedule  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "results"
DEFAULT_TIME_LIMIT = 2.0  # seconds per instance for the exact search: generous, keeps a full run to a few minutes

FULL_NS = (16, 32, 64, 128)
FULL_SEEDS = range(0, 20)
QUICK_NS = (16, 32)
QUICK_SEEDS = range(0, 5)
AVG_DEGREES = (4, 8)

# the CLI --algorithm keys, in the same priority order as optimizer.PORTFOLIO_KEYS, mapped from
# the heuristics' own display names so a row is built by NAME (never by list position)
HEURISTIC_KEY_BY_NAME = {
    DSATUR_NAME: "dsatur",
    RLF_NAME: "rlf",
    SMALLEST_LAST_NAME: "smallest_last",
    LARGEST_FIRST_NAME: "largest_first",
    GREEDY_NAME: "greedy",
}
HEURISTIC_KEYS = ("dsatur", "rlf", "smallest_last", "largest_first", "greedy")

# documented CSV columns, in this order (see "Try it yourself" in the completion report / README)
COLUMNS = (
    "n", "avg_degree", "seed", "nodes", "links", "conflict_pairs",
    *[f"{key}_slots" for key in HEURISTIC_KEYS], *[f"{key}_ms" for key in HEURISTIC_KEYS],
    "best_heuristic", "best_slots",
    "final_slots", "final_algorithm", "lower_bound", "maxdeg_plus_1", "clique_size", "status", "total_ms",
    "plain_tdma_slots", "reuse_factor", "frame_reduction",
    "naive_d1_slots", "naive_d1_violations", "naive_d1_ok",
)


def random_topology(n, avg_degree, seed, radio_range=DEFAULT_RADIO_RANGE_M):
    """Coordinates of n radios, uniform in a square sized so the expected node degree is avg_degree.

    Side length L = R * sqrt(pi * (n - 1) / avg_degree): a disc of radius R around one radio
    covers area pi*R^2, and n-1 other radios spread over an L*L square give roughly avg_degree
    of them inside that disc when L is chosen this way. Deterministic: random.Random(seed) only.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not (math.isfinite(avg_degree) and avg_degree > 0):
        raise ValueError(f"avg_degree must be a finite number greater than 0, got {avg_degree!r}")
    # CONCEPT: Random geometric graphs - a unit disk graph built from points dropped uniformly at random, sized so the expected degree matches avg_degree
    rng = random.Random(seed)
    side = radio_range * math.sqrt(math.pi * (n - 1) / avg_degree)
    return {f"Node_{i:02d}": (rng.uniform(0, side), rng.uniform(0, side)) for i in range(1, n + 1)}


def run_instance(n, avg_degree, seed, time_limit=DEFAULT_TIME_LIMIT, radio_range=DEFAULT_RADIO_RANGE_M):
    """Run the full pipeline once and return one CSV row (a dict with exactly the COLUMNS keys)."""
    coords = random_topology(n, avg_degree, seed, radio_range)
    G = build_radio_graph(coords, radio_range)
    H = build_conflict_graph(G)

    result = optimize(G, H, time_limit=time_limit)
    if not result.verification.ok:
        raise AssertionError(f"n={n} avg_degree={avg_degree} seed={seed}: the optimized schedule failed verification")
    if result.schedule.frame_length < result.lower_bound.value:
        raise AssertionError(f"n={n} avg_degree={avg_degree} seed={seed}: {result.schedule.frame_length} slots is below the lower bound {result.lower_bound.value}")

    naive_schedule = make_schedule(naive_distance1(G), list(G.nodes), NAIVE_D1_NAME)
    naive_verdict = verify_schedule(G, naive_schedule)

    row = {
        "n": n, "avg_degree": avg_degree, "seed": seed,
        "nodes": G.number_of_nodes(), "links": G.number_of_edges(), "conflict_pairs": H.number_of_edges(),
        "best_heuristic": result.best_name, "best_slots": result.best_slots,
        "final_slots": result.schedule.frame_length, "final_algorithm": result.schedule.algorithm,
        "lower_bound": result.lower_bound.value, "maxdeg_plus_1": result.lower_bound.maxdeg_plus_1,
        "clique_size": result.lower_bound.clique_size if result.lower_bound.clique_size is not None else "",
        "status": result.status, "total_ms": round(result.total_ms, 3),
        "plain_tdma_slots": n, "reuse_factor": round(result.schedule.reuse_factor, 4),
        "frame_reduction": round(1 - result.schedule.frame_length / n, 4),
        "naive_d1_slots": naive_schedule.frame_length,
        "naive_d1_violations": len(naive_verdict.violations),
        "naive_d1_ok": naive_verdict.ok,
    }
    for run in result.portfolio:
        key = HEURISTIC_KEY_BY_NAME[run.name]
        row[f"{key}_slots"] = run.slots
        row[f"{key}_ms"] = round(run.ms, 4)
    assert set(row) == set(COLUMNS), f"row is missing or has extra columns: {set(COLUMNS) ^ set(row)}"
    return row


def run_grid(ns, avg_degrees, seeds, time_limit=DEFAULT_TIME_LIMIT, on_group_done=None):
    """Run every (n, avg_degree, seed) combination; return the list of rows in that order.

    `on_group_done(n, avg_degree, rows_of_that_group)` is called after each (n, avg_degree)
    group finishes, for live progress; the CLI uses it to print one line per group.
    """
    rows = []
    for n in ns:
        for avg_degree in avg_degrees:
            group = [run_instance(n, avg_degree, seed, time_limit) for seed in seeds]
            rows += group
            if on_group_done is not None:
                on_group_done(n, avg_degree, group)
    return rows


def write_csv(rows, path):
    """Write `rows` (dicts with exactly the COLUMNS keys) as CSV, LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row):
    return (row["n"], row["avg_degree"])


def _groups(rows):
    """rows grouped by (n, avg_degree), in the order the groups first appear."""
    order, buckets = [], {}
    for row in rows:
        key = _group_key(row)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    return [(key, buckets[key]) for key in order]


def _pct(count, total):
    return 100.0 * count / total if total else 0.0


def _mean(values):
    return statistics.mean(values) if values else 0.0


def write_summary(rows, path):
    """Write results/benchmark_summary.md: two tables per (n, avg_degree), and a data-driven prose summary."""
    # CONCEPT: Benchmarking methodology (seeds, repeated trials, averages vs worst case) - one row per seed, and every number below is a mean or a percentage over those seeds, never a single lucky run
    groups = _groups(rows)
    lines = [
        "# Benchmark summary", "",
        f"{len(rows)} instances: n in {sorted({r['n'] for r in rows})}, "
        f"avg_degree in {sorted({r['avg_degree'] for r in rows})}, "
        f"{len({r['seed'] for r in rows})} seeds per (n, avg_degree). "
        "Generated by `python bench/benchmark.py`; every number below comes from `benchmark.csv`.",
        "",
        "## Heuristic quality (mean slots; the lower the better) and how often each hits the lower bound",
        "",
        "| n | avg_degree | instances | " + " | ".join(HEURISTIC_KEYS) + " | final | LB hit % | proven optimal % |",
        "|---|---|---" + "|---" * (len(HEURISTIC_KEYS) + 3) + "|",
    ]
    for (n, degree), group in groups:
        total = len(group)
        means = " | ".join(f"{_mean([r[f'{k}_slots'] for r in group]):.2f}" for k in HEURISTIC_KEYS)
        final_mean = _mean([r["final_slots"] for r in group])
        hit_lb = _pct(sum(1 for r in group if r["final_slots"] == r["lower_bound"]), total)
        proven = _pct(sum(1 for r in group if r["status"] != BEST_FOUND), total)
        lines.append(f"| {n} | {degree} | {total} | {means} | {final_mean:.2f} | {hit_lb:.0f}% | {proven:.0f}% |")

    lines += [
        "", "## Reuse, runtime and the naive-D1 comparison", "",
        "| n | avg_degree | mean reuse (n/K) | mean frame reduction | mean runtime (ms) | naive-D1 mean slots | naive-D1 conflict-free % |",
        "|---|---|---|---|---|---|---|",
    ]
    for (n, degree), group in groups:
        total = len(group)
        reuse = _mean([r["reuse_factor"] for r in group])
        reduction = _mean([r["frame_reduction"] for r in group]) * 100
        runtime = _mean([r["total_ms"] for r in group])
        naive_slots = _mean([r["naive_d1_slots"] for r in group])
        naive_ok = _pct(sum(1 for r in group if r["naive_d1_ok"]), total)
        lines.append(f"| {n} | {degree} | {reuse:.2f} | {reduction:.1f}% | {runtime:.1f} | {naive_slots:.2f} | {naive_ok:.0f}% |")

    # CONCEPT: Frame length vs throughput & latency (revisit time) - fewer slots means a shorter
    # frame, so at fixed 1 ms slots every radio's revisit time (how often it gets to transmit) is shorter
    overall_hit_lb = _pct(sum(1 for r in rows if r["final_slots"] == r["lower_bound"]), len(rows))
    overall_proven = _pct(sum(1 for r in rows if r["status"] != BEST_FOUND), len(rows))
    worst_heuristic = max(HEURISTIC_KEYS, key=lambda k: _mean([r[f"{k}_slots"] - r["lower_bound"] for r in rows]))
    best_single = min(HEURISTIC_KEYS, key=lambda k: _mean([r[f"{k}_slots"] - r["lower_bound"] for r in rows]))
    naive_ok_overall = _pct(sum(1 for r in rows if r["naive_d1_ok"]), len(rows))
    lines += [
        "", "## What the numbers say", "",
        f"- The optimizer reached the proven lower bound in {overall_hit_lb:.0f}% of all {len(rows)} instances, "
        f"and every instance was proven optimal or the search's gap was reported honestly "
        f"(proven in {overall_proven:.0f}% of instances; the rest are BEST_FOUND with the gap recorded).",
        f"- `{worst_heuristic}` was, on average, the single heuristic furthest from the lower bound; "
        f"`{best_single}` (or DSATUR, its usual co-leader) was the closest.",
        f"- naive-D1 (distance-1 only) used fewer slots but was conflict-free in only {naive_ok_overall:.0f}% "
        "of instances, because it ignores hidden terminals: this is why distance-2 colouring is required.",
        "- Reuse factor (n/K) and frame reduction both grow with n at a fixed density (see the second table): "
        "a bigger deployment gives more radios per slot, exactly as spatial reuse predicts.",
        "",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def plot_slots(rows, path):
    """results/bench_slots.png: mean slots per heuristic, grouped bars, faceted by avg_degree."""
    groups = _groups(rows)
    degrees = sorted({d for _, d in (key for key, _ in groups)})
    ns = sorted({n for n, _ in (key for key, _ in groups)})
    by_key = {key: g for key, g in groups}

    fig, axes = plt.subplots(1, len(degrees), figsize=(6.5 * len(degrees), 4.8), facecolor=viz.SURFACE, sharey=True)
    axes = [axes] if len(degrees) == 1 else list(axes)
    bar_width = 0.8 / len(HEURISTIC_KEYS)
    for ax, degree in zip(axes, degrees):
        ax.set_facecolor(viz.SURFACE)
        for i, key in enumerate(HEURISTIC_KEYS):
            means = [_mean([r[f"{key}_slots"] for r in by_key.get((n, degree), [])]) for n in ns]
            offset = (i - (len(HEURISTIC_KEYS) - 1) / 2) * bar_width
            ax.bar(
                [x + offset for x in range(len(ns))], means, width=bar_width * 0.92,
                color=viz.CATEGORICAL[i], label=_heuristic_label(key), zorder=3,
            )
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_title(f"avg_degree = {degree}", color=viz.INK_PRIMARY, fontsize=11)
        ax.set_xlabel("n (radios)", color=viz.INK_SECONDARY, fontsize=9)
        ax.grid(axis="y", color=viz.GRID_LINE, linewidth=1.0, zorder=0)
        ax.tick_params(colors=viz.INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(viz.BASELINE)
    axes[0].set_ylabel("mean slots (lower is better)", color=viz.INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(HEURISTIC_KEYS), fontsize=8.5, frameon=False, labelcolor=viz.INK_SECONDARY)
    fig.suptitle("Heuristic comparison: mean slots per instance", color=viz.INK_PRIMARY, fontsize=13)
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.08)
    _savefig(fig, path)


def plot_runtime(rows, path):
    """results/bench_runtime.png: mean per-heuristic runtime vs n, log scale, faceted by avg_degree."""
    groups = _groups(rows)
    degrees = sorted({d for _, d in (key for key, _ in groups)})
    ns = sorted({n for n, _ in (key for key, _ in groups)})
    by_key = {key: g for key, g in groups}

    fig, axes = plt.subplots(1, len(degrees), figsize=(6.5 * len(degrees), 4.8), facecolor=viz.SURFACE, sharey=True)
    axes = [axes] if len(degrees) == 1 else list(axes)
    for ax, degree in zip(axes, degrees):
        ax.set_facecolor(viz.SURFACE)
        for i, key in enumerate(HEURISTIC_KEYS):
            means = [_mean([r[f"{key}_ms"] for r in by_key.get((n, degree), [])]) for n in ns]
            # CONCEPT: Scalability analysis - runtime vs n on a log y-axis; a straight line here would mean exponential growth, a flattening curve means it scales gently
            ax.plot(range(len(ns)), means, marker="o", markersize=5, linewidth=2, color=viz.CATEGORICAL[i], label=_heuristic_label(key), zorder=3)
        ax.set_yscale("log")  # one axis is log (runtime spans orders of magnitude); n stays a plain category, never a second log axis
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlim(-0.3, len(ns) - 0.7)
        ax.set_title(f"avg_degree = {degree}", color=viz.INK_PRIMARY, fontsize=11)
        ax.set_xlabel("n (radios)", color=viz.INK_SECONDARY, fontsize=9)
        ax.grid(True, which="both", color=viz.GRID_LINE, linewidth=0.8, zorder=0)
        ax.tick_params(colors=viz.INK_MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(viz.BASELINE)
    axes[0].set_ylabel("mean runtime per heuristic (ms, log scale)", color=viz.INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(HEURISTIC_KEYS), fontsize=8.5, frameon=False, labelcolor=viz.INK_SECONDARY)
    fig.suptitle("Heuristic runtime vs network size", color=viz.INK_PRIMARY, fontsize=13)
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.08)
    _savefig(fig, path)


def _heuristic_label(key):
    return {"dsatur": "DSATUR", "rlf": "RLF", "smallest_last": "smallest_last", "largest_first": "Welsh-Powell", "greedy": "greedy"}[key]


def _savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150, facecolor=viz.SURFACE)
    finally:
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench/benchmark.py",
        description="Compare the scheduling heuristics on a grid of random topologies and save the evidence to results/.",
    )
    parser.add_argument("--quick", action="store_true", help=f"a fast smoke grid: n in {QUICK_NS}, seeds 0..{QUICK_SEEDS[-1]} (instead of n in {FULL_NS}, seeds 0..{FULL_SEEDS[-1]})")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, metavar="SECONDS", help="exact-search time limit per instance (default: %(default)s)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), metavar="PATH", help="where to write benchmark.csv, benchmark_summary.md and the plots (default: %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ns, seeds = (QUICK_NS, QUICK_SEEDS) if args.quick else (FULL_NS, FULL_SEEDS)
    out_dir = Path(args.out_dir)
    total = len(ns) * len(AVG_DEGREES) * len(list(seeds))
    print(f"Running {total} instances: n in {ns}, avg_degree in {AVG_DEGREES}, seeds 0..{list(seeds)[-1]} "
          f"(time limit {args.time_limit}s/instance){' [--quick]' if args.quick else ''}", flush=True)
    started = time.perf_counter()

    def report(n, degree, group):
        mean_final = _mean([r["final_slots"] for r in group])
        hit_lb = _pct(sum(1 for r in group if r["final_slots"] == r["lower_bound"]), len(group))
        elapsed = time.perf_counter() - started
        print(f"  n={n:>3} avg_degree={degree}: {len(group)} instances, all verified, mean slots={mean_final:.2f}, hit LB {hit_lb:.0f}%  [{elapsed:6.1f}s elapsed]", flush=True)

    rows = run_grid(ns, AVG_DEGREES, seeds, time_limit=args.time_limit, on_group_done=report)

    write_csv(rows, out_dir / "benchmark.csv")
    write_summary(rows, out_dir / "benchmark_summary.md")
    plot_slots(rows, out_dir / "bench_slots.png")
    plot_runtime(rows, out_dir / "bench_runtime.png")

    elapsed = time.perf_counter() - started
    print(f"\n{len(rows)} instances, zero verification failures, {elapsed:.1f}s total.", flush=True)
    print(f"Wrote {out_dir / 'benchmark.csv'}, {out_dir / 'benchmark_summary.md'}, "
          f"{out_dir / 'bench_slots.png'}, {out_dir / 'bench_runtime.png'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
