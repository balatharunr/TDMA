"""Division 4 tests: the benchmark harness (bench.benchmark)."""
import csv
import math
import struct
import subprocess
import sys

import pytest

from bench.benchmark import (
    COLUMNS,
    HEURISTIC_KEYS,
    QUICK_NS,
    QUICK_SEEDS,
    build_parser,
    random_topology,
    run_grid,
    run_instance,
    write_csv,
    write_summary,
)
from tdma_brain import viz
from tdma_brain.exact import BEST_FOUND
from tdma_brain.io_utils import natural_key
from tdma_brain.topology import DEFAULT_RADIO_RANGE_M, build_conflict_graph, build_radio_graph
from tests.golden import REPO_ROOT

# a tiny grid, just big enough to exercise every code path without a slow test suite
TINY_NS = (6, 12)
TINY_DEGREES = (3, 5)
TINY_SEEDS = range(2)


def png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def tiny_rows():
    return run_grid(TINY_NS, TINY_DEGREES, TINY_SEEDS, time_limit=2.0)


# --------------------------------------------------------------------------- random_topology
def test_random_topology_is_deterministic_for_a_seed():
    a = random_topology(20, 6, seed=1)
    b = random_topology(20, 6, seed=1)
    assert a == b


def test_random_topology_names_and_count():
    coords = random_topology(16, 4, seed=0)
    assert list(coords) == [f"Node_{i:02d}" for i in range(1, 17)]  # natural order, "Node_01" style
    assert list(coords) == sorted(coords, key=natural_key)
    coords128 = random_topology(128, 4, seed=0)
    assert list(coords128)[-1] == "Node_128" and len(coords128) == 128  # 2-digit padding never truncates 3-digit numbers


def test_random_topology_points_are_inside_the_square():
    n, avg_degree, seed = 40, 6, 3
    side = DEFAULT_RADIO_RANGE_M * math.sqrt(math.pi * (n - 1) / avg_degree)
    coords = random_topology(n, avg_degree, seed)
    assert all(0 <= x <= side and 0 <= y <= side for x, y in coords.values())


def test_random_topology_gives_roughly_the_requested_average_degree():
    """Not exact (it is random), but the mean degree over many seeds must be in the right ballpark."""
    avg_degree = 6
    degrees = []
    for seed in range(30):
        coords = random_topology(60, avg_degree, seed)
        G = build_radio_graph(coords, DEFAULT_RADIO_RANGE_M)
        degrees += [d for _, d in G.degree()]
    mean_degree = sum(degrees) / len(degrees)
    assert avg_degree * 0.7 <= mean_degree <= avg_degree * 1.3


@pytest.mark.parametrize("bad_n", [0, -1, 1.5])
def test_random_topology_rejects_a_bad_n(bad_n):
    with pytest.raises(ValueError, match="n must be"):
        random_topology(bad_n, 4, seed=0)


@pytest.mark.parametrize("bad_degree", [0, -1, float("nan"), float("inf")])
def test_random_topology_rejects_a_bad_avg_degree(bad_degree):
    with pytest.raises(ValueError, match="avg_degree"):
        random_topology(16, bad_degree, seed=0)


# --------------------------------------------------------------------------- run_instance / run_grid
def test_run_instance_returns_exactly_the_documented_columns():
    row = run_instance(10, 4, seed=0, time_limit=2.0)
    assert set(row) == set(COLUMNS)


def test_run_instance_is_internally_consistent():
    row = run_instance(12, 5, seed=1, time_limit=2.0)
    assert row["final_slots"] >= row["lower_bound"] >= row["maxdeg_plus_1"]  # Portfolio K >= LB, always
    assert row["final_slots"] <= min(row[f"{k}_slots"] for k in HEURISTIC_KEYS)  # never worse than every heuristic
    assert row["best_slots"] == min(row[f"{k}_slots"] for k in HEURISTIC_KEYS)
    assert row["reuse_factor"] == pytest.approx(row["nodes"] / row["final_slots"], rel=1e-3)
    assert row["frame_reduction"] == pytest.approx(1 - row["final_slots"] / row["nodes"], rel=1e-3)
    assert row["naive_d1_violations"] >= 0 and row["naive_d1_ok"] == (row["naive_d1_violations"] == 0)
    assert row["plain_tdma_slots"] == row["nodes"] == 12


def test_run_grid_calls_the_progress_callback_once_per_group():
    calls = []
    rows = run_grid(TINY_NS, TINY_DEGREES, TINY_SEEDS, time_limit=2.0, on_group_done=lambda n, d, group: calls.append((n, d, len(group))))
    assert len(rows) == len(TINY_NS) * len(TINY_DEGREES) * len(list(TINY_SEEDS))
    assert calls == [(n, d, len(list(TINY_SEEDS))) for n in TINY_NS for d in TINY_DEGREES]


def test_zero_verification_failures_across_a_wider_grid():
    """The guide's hard requirement, run on a grid wider than the quick smoke test."""
    for n in (16, 24):
        for degree in (4, 8):
            for seed in range(3):
                row = run_instance(n, degree, seed, time_limit=2.0)
                assert row["final_slots"] >= row["lower_bound"]  # would have raised AssertionError inside run_instance otherwise


# --------------------------------------------------------------------------- write_csv / write_summary
def test_write_csv_has_the_documented_columns_and_lf_endings(tmp_path):
    rows = tiny_rows()
    path = tmp_path / "benchmark.csv"
    write_csv(rows, path)
    assert b"\r" not in path.read_bytes()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(COLUMNS)
        read_rows = list(reader)
    assert len(read_rows) == len(rows)
    assert int(read_rows[0]["n"]) == rows[0]["n"] and read_rows[0]["status"] == rows[0]["status"]


def test_write_csv_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "benchmark.csv"
    write_csv(tiny_rows(), path)
    assert path.exists()


def test_write_summary_every_number_traces_back_to_the_rows(tmp_path):
    rows = tiny_rows()
    path = tmp_path / "summary.md"
    write_summary(rows, path)
    text = path.read_text(encoding="utf-8")
    assert f"{len(rows)} instances" in text
    for n in TINY_NS:
        assert str(n) in text
    assert "LB hit" in text and "naive-D1" in text and "proven optimal" in text
    proven = sum(1 for r in rows if r["status"] != BEST_FOUND)
    assert f"proven in {100 * proven / len(rows):.0f}%" in text  # the prose is computed, not typed


def test_write_summary_handles_a_single_group_without_crashing(tmp_path):
    rows = run_grid((8,), (4,), range(2), time_limit=2.0)
    write_summary(rows, tmp_path / "summary.md")
    assert (tmp_path / "summary.md").exists()


# --------------------------------------------------------------------------- plots
def test_plot_slots_and_plot_runtime_write_non_empty_pngs(tmp_path):
    from bench.benchmark import plot_runtime, plot_slots

    rows = tiny_rows()
    slots_path, runtime_path = tmp_path / "slots.png", tmp_path / "runtime.png"
    plot_slots(rows, slots_path)
    plot_runtime(rows, runtime_path)
    for path in (slots_path, runtime_path):
        assert path.stat().st_size > 0
        width, height = png_size(path)
        assert width > 100 and height > 100


def test_plots_use_the_shared_style_colours(tmp_path):
    from bench.benchmark import HEURISTIC_KEYS as KEYS

    assert len(KEYS) <= len(viz.CATEGORICAL)  # every heuristic gets its own validated hue, none reused


# --------------------------------------------------------------------------- CLI (argparse) and a real quick run
def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.quick is False and args.time_limit == 2.0 and args.out_dir


def test_build_parser_quick_flag():
    assert build_parser().parse_args(["--quick"]).quick is True


def test_quick_grid_constants_match_the_guide():
    assert QUICK_NS == (16, 32) and list(QUICK_SEEDS) == list(range(5))


def test_the_quick_run_as_a_real_process(tmp_path):
    result = subprocess.run(
        [sys.executable, "bench/benchmark.py", "--quick", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "zero verification failures" in result.stdout
    for name in ("benchmark.csv", "benchmark_summary.md", "bench_slots.png", "bench_runtime.png"):
        path = tmp_path / name
        assert path.exists() and path.stat().st_size > 0, name
    with open(tmp_path / "benchmark.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 2 * 5  # QUICK_NS x AVG_DEGREES x QUICK_SEEDS
