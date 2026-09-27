"""Division 2-3 tests: the ASCII report and the --verbose block (tdma_brain.report)."""
import dataclasses
import re

import pytest

from tdma_brain import optimizer
from tdma_brain.bounds import LowerBound
from tdma_brain.exact import BEST_FOUND, PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH
from tdma_brain.heuristics import DSATUR_NAME, dsatur, greedy_first_fit, naive_distance1, smallest_last
from tdma_brain.io_utils import natural_key
from tdma_brain.optimizer import optimize
from tdma_brain.report import (
    BAR,
    RULE,
    TITLE,
    column_labels,
    describe_algorithm,
    format_report,
    format_verbose,
    optimality_text,
)
from tdma_brain.schedule import make_schedule
from tdma_brain.verify import VerificationResult, verify_schedule
from tests.golden import GOLDEN, REPO_ROOT
from tests.helpers import build_graphs, raw_schedule

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "grid16_300m_report.txt"  # copied verbatim from the guide's expected output


def report_for(name, algorithm="dsatur"):
    G, H = build_graphs(name)
    nodes = list(G.nodes)
    if algorithm == "dsatur":
        schedule = make_schedule(dsatur(H), nodes, DSATUR_NAME)
    elif algorithm == "greedy":
        schedule = make_schedule(greedy_first_fit(H, sorted(nodes, key=natural_key)), nodes, "greedy (natural order)")
    else:
        schedule = make_schedule(naive_distance1(G), nodes, "naive-d1")
    return format_report(schedule, verify_schedule(G, schedule), 500.0), schedule


def bar_positions(line):
    return [i for i, ch in enumerate(line) if ch == "|"]


# --------------------------------------------------------------------------- the official output
def test_official_report_matches_the_guides_expected_output_line_by_line():
    report, _ = report_for("grid16_300m")
    expected = FIXTURE.read_text(encoding="utf-8").splitlines()
    actual = report.splitlines()
    assert [line.rstrip() for line in actual] == [line.rstrip() for line in expected]


def test_the_report_has_no_trailing_newline_or_trailing_spaces():
    report, _ = report_for("grid16_300m")
    assert not report.endswith("\n")
    assert all(line == line.rstrip() for line in report.split("\n"))


# --------------------------------------------------------------------------- layout rules, everywhere
@pytest.mark.parametrize("algorithm", ["dsatur", "greedy", "naive-d1"])
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_layout_rules_hold_for_every_example_and_algorithm(golden, algorithm):
    report, schedule = report_for(golden.name, algorithm)
    lines = report.split("\n")
    report.encode("ascii")
    assert "\t" not in report and all(line == line.rstrip() for line in lines)
    # the frame of the report
    assert lines[0] == lines[2] == lines[-1] == "=" * 64 == BAR
    assert lines[1] == " " * 15 + "TDMA TOPOLOGY OPTIMIZATION REPORT" == TITLE
    assert lines[8] == "-" * 65 == RULE
    assert lines[9] == " NODE -> SLOT ASSIGNMENTS:"
    n, k = len(schedule.nodes), schedule.frame_length
    assert lines[10:10 + n] == [f"  {node}: Slot {schedule.node_to_slot[node]}" for node in schedule.nodes]
    assert lines[10 + n] == ""
    assert lines[11 + n] == " STRUCTURAL TDMA SCHEDULE MATRIX (Slot x Node Boolean Matrix):"
    # the matrix: header, separator, K rows, separator - the separators are as long as the header
    header, rule, *rest = lines[12 + n:]
    rows, closing_rule = rest[:k], rest[k]
    assert header.startswith("Slot \\ Node |")
    assert rule == closing_rule == "-" * len(header)
    assert len(rows) == k and all(row.startswith(f" Slot {s:02d}    |") for s, row in enumerate(rows))
    assert all(bar_positions(row) == bar_positions(header) for row in rows)  # every | lines up
    assert [[int(cell) for cell in row.split("|")[1:]] for row in rows] == schedule.matrix
    # the summary block
    assert lines[3] == f"Total Nodes Processed   : {n}"
    assert lines[4] == "Configured Radio Range  : 500.0 meters"
    assert lines[5] == f"Optimized Frame Length  : {k} unique timeslots (Lower is better)"
    assert lines[6].startswith("Scheduling Algorithm    : ")
    assert lines[7] == f"Spatial Reuse Factor    : {n / k:.2f} nodes/slot (plain TDMA = 1.00)"


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_verified_schedules_end_with_the_clean_line(golden):
    report, _ = report_for(golden.name)
    assert report.split("\n")[-2:] == ["Execution finalized cleanly. Schedule verified conflict-free.", BAR]
    assert "VERIFICATION FAILED" not in report


# --------------------------------------------------------------------------- column labels
def test_labels_are_the_trailing_digits_zero_padded_to_two():
    assert column_labels(["Node_01", "Node_02", "Node_16"]) == (["01", "02", "16"], False)
    assert column_labels(["Node_1", "Node_2", "Node_10"]) == (["01", "02", "10"], False)
    assert column_labels(["Node_100", "Node_101"]) == (["100", "101"], False)  # never truncated
    assert column_labels(["Node_007"]) == (["007"], False)


@pytest.mark.parametrize(
    "nodes",
    [["Node_A", "Node_B", "Node_C"], ["Node_01", "Node_1"], ["A1", "B1"], ["Node_01", "Base"], ["Node_1a"]],
    ids=["no digits", "01 vs 1 collide", "A1 vs B1 collide", "one name without digits", "digits not trailing"],
)
def test_labels_fall_back_to_one_based_indexes(nodes):
    labels, is_legend = column_labels(nodes)
    assert is_legend and labels == [f"{i:02d}" for i in range(1, len(nodes) + 1)]


def test_report_with_index_labels_prints_a_legend():
    report, _ = report_for("hidden_terminal3")
    assert report == "\n".join([
        "=" * 64,
        " " * 15 + "TDMA TOPOLOGY OPTIMIZATION REPORT",
        "=" * 64,
        "Total Nodes Processed   : 3",
        "Configured Radio Range  : 500.0 meters",
        "Optimized Frame Length  : 3 unique timeslots (Lower is better)",
        "Scheduling Algorithm    : DSATUR on 2-hop conflict graph (G^2)",
        "Spatial Reuse Factor    : 1.00 nodes/slot (plain TDMA = 1.00)",
        "-" * 65,
        " NODE -> SLOT ASSIGNMENTS:",
        "  Node_A: Slot 0",
        "  Node_B: Slot 1",
        "  Node_C: Slot 2",
        "",
        " STRUCTURAL TDMA SCHEDULE MATRIX (Slot x Node Boolean Matrix):",
        "Slot \\ Node | 01 | 02 | 03",
        "-" * 26,
        " Slot 00    | 1  | 0  | 0",
        " Slot 01    | 0  | 1  | 0",
        " Slot 02    | 0  | 0  | 1",
        "-" * 26,
        "Legend: 01=Node_A, 02=Node_B, 03=Node_C",
        "Execution finalized cleanly. Schedule verified conflict-free.",
        "=" * 64,
    ])


def test_report_without_index_labels_has_no_legend():
    report, _ = report_for("grid16_300m")
    assert "Legend" not in report


# --------------------------------------------------------------------------- sizes beyond the brief's 16
def test_columns_stay_aligned_with_more_than_99_nodes_and_slots():
    nodes = {f"Node_{i:03d}": i - 1 for i in range(1, 106)}  # 105 radios, one slot each
    schedule = raw_schedule(nodes)
    report = format_report(schedule, VerificationResult(ok=True), 500.0)
    lines = report.split("\n")
    header = next(line for line in lines if line.startswith("Slot \\ Node"))
    rows = [line for line in lines if line.startswith(" Slot ")]
    assert len(rows) == 105 and rows[0].startswith(" Slot 000    |") and rows[-1].startswith(" Slot 104    |")
    assert all(bar_positions(row) == bar_positions(header) for row in rows)
    assert header.split("|")[1:4] == [" 001 ", " 002 ", " 003 "]
    rule = lines[lines.index(header) + 1]
    assert rule == "-" * len(header) and all(line == line.rstrip() for line in lines)


def test_a_single_radio():
    schedule = make_schedule({"Node_01": 5}, ["Node_01"], "DSATUR")
    report = format_report(schedule, VerificationResult(ok=True), 500.0)
    assert "Optimized Frame Length  : 1 unique timeslots (Lower is better)" in report
    assert "Spatial Reuse Factor    : 1.00 nodes/slot" in report
    assert " Slot 00    | 1" in report.split("\n")


# --------------------------------------------------------------------------- summary details
def test_spatial_reuse_factor_is_nodes_per_slot_with_two_decimals():
    assert "Spatial Reuse Factor    : 1.78 nodes/slot (plain TDMA = 1.00)" in report_for("grid16_300m")[0]
    assert "Spatial Reuse Factor    : 16.00 nodes/slot" in report_for("isolated16_1000m")[0]
    assert "Spatial Reuse Factor    : 1.00 nodes/slot" in report_for("cluster16_30m")[0]
    assert "Spatial Reuse Factor    : 3.20 nodes/slot" in report_for("grid16_400m")[0]  # 16 radios in 5 slots


def test_algorithm_names_in_the_report():
    assert describe_algorithm("DSATUR") == "DSATUR on 2-hop conflict graph (G^2)"
    assert describe_algorithm("greedy (natural order)") == "greedy (natural order) on 2-hop conflict graph (G^2)"
    assert describe_algorithm("naive-d1") == "naive-d1: first-fit on the radio graph G only (UNSAFE demo)"
    assert "Scheduling Algorithm    : naive-d1: first-fit on the radio graph G only (UNSAFE demo)" in report_for("grid16_300m", "naive-d1")[0]


def test_the_radio_range_is_shown_as_given():
    schedule = make_schedule({"Node_1": 0}, ["Node_1"], "DSATUR")
    assert "Configured Radio Range  : 250.0 meters" in format_report(schedule, VerificationResult(ok=True), 250.0)


# --------------------------------------------------------------------------- failure
def test_failed_verification_lists_every_conflict_in_place_of_the_clean_line():
    report, schedule = report_for("grid16_300m", "naive-d1")
    lines = report.split("\n")
    assert "Optimized Frame Length  : 4 unique timeslots (Lower is better)" in lines
    assert "Execution finalized cleanly" not in report
    start = lines.index("VERIFICATION FAILED: 24 conflicts")
    conflicts = lines[start + 1:-1]
    assert lines[-1] == BAR and len(conflicts) == 24
    assert conflicts[0] == "  Node_01 & Node_03 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_02, Node_06"
    assert all(line.startswith("  Node_") and "HIDDEN TERMINAL (2-hop) via " in line for line in conflicts)
    assert lines[start - 1].startswith("-----")  # the matrix is still printed above the verdict


def test_structural_problems_are_listed_too():
    schedule = raw_schedule({"Node_1": 0, "Node_2": 1}, k=3)
    verdict = VerificationResult(ok=False, violations=[], problems=["empty slot(s) that no radio uses: 2"])
    lines = format_report(schedule, verdict, 500.0).split("\n")
    assert lines[-3:] == ["VERIFICATION FAILED: 1 structural problems", "  PROBLEM: empty slot(s) that no radio uses: 2", BAR]


def test_the_report_is_a_pure_function():
    assert report_for("ring16_r1150m")[0] == report_for("ring16_r1150m")[0]


# --------------------------------------------------------------------------- Division 3: lower bound and optimality lines
FIXTURE_AUTO = REPO_ROOT / "tests" / "fixtures" / "grid16_300m_report_auto.txt"  # the D2 fixture + the guide's two new lines


def optimized(name, **kwargs):
    """(report text with the two proof lines, the OptimizationResult) for examples/<name>.json."""
    G, H = build_graphs(name)
    result = optimize(G, H, **kwargs)
    return format_report(result.schedule, result.verification, 500.0, result), result


def test_official_auto_report_matches_the_guides_expected_lines_line_by_line():
    report, _ = optimized("grid16_300m")
    expected = FIXTURE_AUTO.read_text(encoding="utf-8").splitlines()
    assert [line.rstrip() for line in report.splitlines()] == [line.rstrip() for line in expected]


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_the_proof_lines_sit_after_the_reuse_factor_and_change_nothing_else(golden):
    report, result = optimized(golden.name)
    lines = report.split("\n")
    assert lines[7].startswith("Spatial Reuse Factor    : ")
    assert lines[8].startswith("Lower Bound (proof)     : ") and lines[9].startswith("Optimality              : ")
    assert lines[10] == RULE
    without = lines[:8] + lines[10:]  # drop the two new lines ...
    plain = format_report(result.schedule, result.verification, 500.0)  # ... and it is exactly the Division 2 report
    assert without == plain.split("\n")
    assert all(line == line.rstrip() for line in lines)
    report.encode("ascii")


@pytest.mark.parametrize(
    "name, lower_bound_line, optimality_line",
    [
        ("grid16_300m", "9 timeslots (max 2-hop clique = 9, maxdeg+1 = 9)", "PROVEN OPTIMAL (frame length = lower bound)"),
        ("grid16_250m", "12 timeslots (max 2-hop clique = 12, maxdeg+1 = 11)", "PROVEN OPTIMAL (frame length = lower bound)"),
        ("ring16_r1150m", "3 timeslots (max 2-hop clique = 3, maxdeg+1 = 3)", "PROVEN OPTIMAL (exact search: 3 slots impossible)"),
        ("isolated16_1000m", "1 timeslots (max 2-hop clique = 1, maxdeg+1 = 1)", "PROVEN OPTIMAL (frame length = lower bound)"),
        ("cluster16_30m", "16 timeslots (max 2-hop clique = 16, maxdeg+1 = 16)", "PROVEN OPTIMAL (frame length = lower bound)"),
    ],
)
def test_lower_bound_and_optimality_lines(name, lower_bound_line, optimality_line):
    lines = optimized(name)[0].split("\n")
    assert f"Lower Bound (proof)     : {lower_bound_line}" in lines
    assert f"Optimality              : {optimality_line}" in lines


def test_best_found_shows_the_open_gap():
    report, result = optimized("ring16_r1150m", time_limit=0)  # no time for the search: 4 slots vs a bound of 3
    assert result.status == BEST_FOUND
    assert "Optimality              : BEST FOUND (gap = 1, time limit reached)" in report.split("\n")
    assert report.split("\n")[-2] == "Execution finalized cleanly. Schedule verified conflict-free."  # the schedule itself is fine


def test_the_report_says_when_the_clique_search_was_skipped():
    report, result = optimized("hidden_terminal3")
    skipped = dataclasses.replace(result, lower_bound=LowerBound(3, 3, None, ()))
    text = format_report(skipped.schedule, skipped.verification, 500.0, skipped)
    assert "Lower Bound (proof)     : 3 timeslots (max 2-hop clique skipped: more than 300 nodes, maxdeg+1 = 3)" in text.split("\n")


def test_optimality_text_for_every_status():
    assert optimality_text(PROVEN_OPTIMAL_BY_BOUND, 9, 9) == "PROVEN OPTIMAL (frame length = lower bound)"
    assert optimality_text(PROVEN_OPTIMAL_BY_SEARCH, 4, 3) == "PROVEN OPTIMAL (exact search: 3 slots impossible)"
    assert optimality_text(BEST_FOUND, 7, 2) == "BEST FOUND (gap = 5, time limit reached)"
    with pytest.raises(ValueError, match="unknown optimality status"):
        optimality_text("MAYBE", 4, 3)


# --------------------------------------------------------------------------- Division 3: the --verbose block
def masked(text):
    """The verbose block with the timings (which change from run to run) replaced by <ms>."""
    return re.sub(r"(?m)^(  .+\S\s+\d+)\s+\d+\.\d$", r"\1   <ms>", text)


def test_verbose_block_for_grid_400m_has_the_guides_layout():
    _, result = optimized("grid16_400m")
    text = format_verbose(result)
    lines = masked(text).split("\n")
    assert lines[:8] == [
        "HEURISTIC PORTFOLIO (conflict graph: 16 nodes, 58 conflict pairs)",
        "  Heuristic                        Slots   Time (ms)",
        "  DSATUR                               5   <ms>",
        "  RLF                                  5   <ms>",
        "  smallest_last                        6   <ms>",
        "  largest_first (Welsh-Powell)         6   <ms>",
        "  greedy (natural order)               6   <ms>",
        "  -> best: DSATUR ; iterated greedy (300 it): 5 (no change)",
    ]
    lower_bound = re.fullmatch(r"LOWER BOUND: maxdeg\+1 = 5, max clique = 5 \{(.+)\} -> LB = 5", lines[8])
    assert lower_bound, lines[8]
    clique = lower_bound.group(1).split(", ")  # any maximum clique is acceptable: check that it really is one
    _, H = build_graphs("grid16_400m")
    assert len(clique) == 5 and all(H.has_edge(u, v) for i, u in enumerate(clique) for v in clique[i + 1:])
    assert lines[9] == "RESULT: 5 slots = lower bound -> PROVEN OPTIMAL (frame length = lower bound)"
    assert len(lines) == 10


def test_verbose_columns_line_up_and_the_block_is_plain_ascii():
    for name in ("grid16_300m", "grid16_250m", "ring16_r1150m"):
        _, result = optimized(name)
        text = format_verbose(result)
        text.encode("ascii")
        lines = text.split("\n")
        assert all(line == line.rstrip() for line in lines)
        table = [line for line in lines if re.match(r"  (Heuristic|DSATUR|RLF|smallest_last|largest_first|greedy)", line)]
        assert len(table) == 6  # the header and the five heuristics
        assert len({len(line) for line in table}) == 1  # every line ends in the same column: right-aligned numbers
        for line in table:  # a 2-space indent and a 33-wide name column: the Slots field is columns 35-39
            assert line[35:40] == "Slots" or line[35:40].strip().isdigit(), line


def test_verbose_shows_the_exact_search_proof_for_the_ring():
    _, result = optimized("ring16_r1150m")
    lines = format_verbose(result).split("\n")
    assert lines[-2].startswith("LOWER BOUND: maxdeg+1 = 3, max clique = 3 {") and lines[-2].endswith("} -> LB = 3")
    assert lines[-1] == "RESULT: 4 slots, lower bound 3 -> exact search: 3 slots impossible -> PROVEN OPTIMAL (exact search)"


def test_verbose_shows_a_gap_that_stayed_open():
    _, result = optimized("ring16_r1150m", time_limit=0)
    assert format_verbose(result).split("\n")[-1] == "RESULT: 4 slots, lower bound 3 -> BEST FOUND (gap = 1, time limit reached)"


def test_verbose_reports_what_iterated_greedy_and_the_exact_search_improved(monkeypatch):
    G, H = build_graphs("grid16_400m")
    wasteful = smallest_last(H)  # 6 slots
    monkeypatch.setattr(optimizer, "run_algorithm", lambda key, G_, H_: (key.upper(), dict(wasteful)))
    polished = format_verbose(optimize(G, H)).split("\n")
    assert "  -> best: DSATUR ; iterated greedy (300 it): 5 (improved from 6)" in polished
    rainbow = {v: i for i, v in enumerate(sorted(H.nodes, key=natural_key))}
    monkeypatch.setattr(optimizer, "run_algorithm", lambda key, G_, H_: (key.upper(), dict(rainbow)))
    monkeypatch.setattr(optimizer, "iterated_greedy", lambda H_, start, iterations, seed: dict(start))
    searched = format_verbose(optimize(G, H)).split("\n")
    assert "  -> exact search improved 16 to 5 slots" in searched
    assert searched[-1] == "RESULT: 5 slots = lower bound -> PROVEN OPTIMAL (frame length = lower bound)"


def test_verbose_says_when_the_clique_search_was_skipped():
    _, result = optimized("hidden_terminal3")
    skipped = dataclasses.replace(result, lower_bound=LowerBound(3, 3, None, ()))
    assert "LOWER BOUND: maxdeg+1 = 3, max clique skipped (more than 300 nodes) -> LB = 3" in format_verbose(skipped).split("\n")
