"""Division 1-3 tests: the command line (tdma_brain.cli and the tdma_schedule.py entry point)."""
import json
import os
import re
import subprocess
import sys

import pytest

from tdma_brain.cli import compute_schedule, main
from tdma_brain.schedule import Schedule
from tests.golden import GOLDEN, REPO_ROOT, example_path
from tests.helpers import build_graphs
from tests.test_topology import GRID_300M_TEXT, HIDDEN_TERMINAL3_TEXT

SCRIPT = REPO_ROOT / "tdma_schedule.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
# the guide's expected output for the official input (copied by script), without the final newline:
OFFICIAL_REPORT = (FIXTURES / "grid16_300m_report_auto.txt").read_text(encoding="utf-8").rstrip("\n")  # the default (--algorithm auto)
DSATUR_REPORT = (FIXTURES / "grid16_300m_report.txt").read_text(encoding="utf-8").rstrip("\n")  # a single heuristic: no proof lines
GRID = str(example_path("grid16_300m"))


def without_timings(text):
    """The --verbose block prints milliseconds, which change from run to run; mask them."""
    return re.sub(r"(?m)^(  .+\S\s+\d+)\s+\d+\.\d$", r"\1   <ms>", text)


def run_main(capsys, *argv):
    """Call main() in-process; return (exit code, stdout, stderr)."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_script(*args, cwd=REPO_ROOT, env_extra=None):
    """Run `python tdma_schedule.py ...` as a real process, as the graders will."""
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd, env=env
    )


# --------------------------------------------------------------------------- the official run (Divisions 2 and 3)
def test_default_run_prints_the_official_report_from_a_file(capsys):
    code, out, err = run_main(capsys, "--file", GRID)
    assert (code, out, err) == (0, OFFICIAL_REPORT + "\n", "")  # stdout is the report and nothing else


def test_default_run_from_the_json_string_gives_the_same_report(capsys):
    text = example_path("grid16_300m").read_text(encoding="utf-8").strip()  # the brief's input, as one argument
    assert run_main(capsys, text) == (0, OFFICIAL_REPORT + "\n", "")


def test_auto_is_the_default_algorithm(capsys):
    assert run_main(capsys, "--file", GRID, "--algorithm", "auto") == (0, OFFICIAL_REPORT + "\n", "")
    assert "Lower Bound (proof)     : 9 timeslots (max 2-hop clique = 9, maxdeg+1 = 9)" in OFFICIAL_REPORT


def test_dsatur_algorithm_prints_the_division_2_report(capsys):
    # one heuristic on its own proves nothing, so the two proof lines are absent
    assert run_main(capsys, "--file", GRID, "--algorithm", "dsatur") == (0, DSATUR_REPORT + "\n", "")


@pytest.mark.parametrize(
    "algorithm, name",
    [("rlf", "RLF"), ("smallest_last", "smallest_last"), ("largest_first", "largest_first (Welsh-Powell)")],
)
def test_the_other_single_heuristics_give_valid_schedules(capsys, algorithm, name):
    code, out, err = run_main(capsys, "--file", GRID, "--algorithm", algorithm)
    assert (code, err) == (0, "") and "Schedule verified conflict-free." in out
    assert f"Scheduling Algorithm    : {name} on 2-hop conflict graph (G^2)" in out
    assert "Lower Bound (proof)" not in out and "Optimality" not in out


def test_greedy_algorithm_gives_a_valid_schedule(capsys):
    code, out, _ = run_main(capsys, "--file", str(example_path("grid16_400m")), "--algorithm", "greedy")
    assert code == 0 and "Optimized Frame Length  : 6 unique timeslots" in out  # DSATUR needs 5 here
    assert "greedy (natural order) on 2-hop conflict graph (G^2)" in out
    assert "Schedule verified conflict-free." in out


def test_naive_d1_shows_why_distance_2_matters(capsys):
    code, out, err = run_main(capsys, "--file", GRID, "--algorithm", "naive-d1")
    assert code == 1 and err == ""  # the failure is in the report and in the exit code
    assert "Optimized Frame Length  : 4 unique timeslots (Lower is better)" in out
    assert "VERIFICATION FAILED: 24 conflicts" in out and "Schedule verified" not in out
    assert "  Node_01 & Node_03 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_02, Node_06" in out


def test_naive_d1_is_fine_when_there_are_no_hidden_terminals(capsys):
    code, out, _ = run_main(capsys, "--file", str(example_path("isolated16_1000m")), "--algorithm", "naive-d1")
    assert code == 0 and "Schedule verified conflict-free." in out


def test_hidden_terminal3_report_uses_index_labels_and_a_legend(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("hidden_terminal3")))
    assert code == 0 and "3 node(s)" in err  # not 16 nodes: a warning, not an error
    assert "Legend: 01=Node_A, 02=Node_B, 03=Node_C" in out.split("\n")
    assert "Optimized Frame Length  : 3 unique timeslots (Lower is better)" in out


def test_show_topology_prints_the_summary_before_the_report(capsys):
    code, out, err = run_main(capsys, "--file", GRID, "--show-topology")
    assert (code, err) == (0, "")
    assert out == GRID_300M_TEXT + "\n\n" + OFFICIAL_REPORT + "\n"


def test_show_topology_hidden_terminal3(capsys):
    code, out, _ = run_main(capsys, "--file", str(example_path("hidden_terminal3")), "--show-topology")
    assert code == 0 and out.startswith(HIDDEN_TERMINAL3_TEXT + "\n\n" + "=" * 64 + "\n")


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_every_example_gets_a_verified_schedule_at_the_golden_optimum(capsys, golden):
    code, out, _ = run_main(capsys, "--file", str(example_path(golden.name)), "--show-topology")
    assert code == 0
    assert f"Radio links (1-hop)        : {golden.links}\n" in out
    assert f"Conflict pairs (<= 2 hops) : {golden.conflict_pairs}\n" in out
    assert f"Hidden-terminal pairs      : {golden.hidden_pairs} (2-hop only)\n" in out
    assert f"Optimized Frame Length  : {golden.optimal_slots} unique timeslots (Lower is better)\n" in out
    assert out.endswith("Execution finalized cleanly. Schedule verified conflict-free.\n" + "=" * 64 + "\n")


@pytest.mark.parametrize(
    "argv, code",
    [
        (["--file", GRID], 0),
        (["--file", GRID, "--algorithm", "greedy"], 0),
        (["--file", GRID, "--algorithm", "rlf"], 0),
        (["--file", GRID, "--algorithm", "naive-d1"], 1),
        (["--file", str(example_path("isolated16_1000m")), "--algorithm", "naive-d1"], 0),
        (["--file", str(example_path("ring16_r1150m")), "--time-limit", "0"], 0),  # a gap left open is not a failure
        (['{"Node_01":[0,"x"]}'], 2),
        (["--file", "does_not_exist.json"], 2),
        (["--file", GRID, "--time-limit", "-1"], 2),
    ],
    ids=["auto ok", "greedy ok", "rlf ok", "naive-d1 fails", "naive-d1 harmless", "gap left open", "bad input", "missing file", "bad time limit"],
)
def test_exit_codes(capsys, argv, code):
    assert run_main(capsys, *argv)[0] == code


def test_compute_schedule_returns_the_canonical_schedule():
    G, H = build_graphs("grid16_300m")
    schedule = compute_schedule(G, H, "dsatur")
    assert isinstance(schedule, Schedule) and schedule.algorithm == "DSATUR" and schedule.frame_length == 9
    assert compute_schedule(G, H, "naive-d1").algorithm == "naive-d1"


# --------------------------------------------------------------------------- --export-json
def test_export_json_writes_the_schedule_and_creates_parent_folders(capsys, tmp_path):
    target = tmp_path / "results" / "nested" / "grid16_300m.schedule.json"
    code, out, err = run_main(capsys, "--file", GRID, "--export-json", str(target))
    assert code == 0 and out == OFFICIAL_REPORT + "\n"  # the report on stdout is unchanged
    assert "exported" in err and str(target) in err
    text = target.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["verified"] is True and data["frame_length"] == 9 and data["algorithm"] == "DSATUR"
    assert data["lower_bound"] == 9 and data["optimality"] == "PROVEN_OPTIMAL_BY_BOUND"  # filled in since Division 3
    G, H = build_graphs("grid16_300m")
    assert Schedule.from_json(text) == compute_schedule(G, H, "dsatur")
    assert b"\r" not in target.read_bytes()  # LF endings even on Windows: Part 2 runs on Linux


def test_export_json_of_a_single_heuristic_has_no_proof_fields(capsys, tmp_path):
    target = tmp_path / "dsatur.schedule.json"
    run_main(capsys, "--file", GRID, "--algorithm", "dsatur", "--export-json", str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["lower_bound"] is None and data["optimality"] is None and data["verified"] is True


@pytest.mark.parametrize(
    "name, lower_bound, optimality",
    [
        ("ring16_r1150m", 3, "PROVEN_OPTIMAL_BY_SEARCH"),
        ("grid16_250m", 12, "PROVEN_OPTIMAL_BY_BOUND"),
        ("line16_400m", 3, "PROVEN_OPTIMAL_BY_BOUND"),
    ],
)
def test_export_json_records_the_proof(capsys, tmp_path, name, lower_bound, optimality):
    target = tmp_path / "proof.schedule.json"
    assert run_main(capsys, "--file", str(example_path(name)), "--export-json", str(target))[0] == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert (data["lower_bound"], data["optimality"]) == (lower_bound, optimality)


def test_export_json_records_an_open_gap(capsys, tmp_path):
    target = tmp_path / "gap.schedule.json"
    run_main(capsys, "--file", str(example_path("ring16_r1150m")), "--time-limit", "0", "--export-json", str(target))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert (data["frame_length"], data["lower_bound"], data["optimality"]) == (4, 3, "BEST_FOUND")


# --------------------------------------------------------------------------- Division 3: --verbose, --time-limit, --seed
def test_verbose_prints_the_portfolio_and_the_bound_before_the_report(capsys):
    code, out, err = run_main(capsys, "--file", GRID, "--verbose")
    assert (code, err) == (0, "")
    verbose, report = without_timings(out).split("\n\n", 1)
    assert verbose.startswith("HEURISTIC PORTFOLIO (conflict graph: 16 nodes, 90 conflict pairs)\n")
    assert "  -> best: DSATUR ; iterated greedy (300 it): 9 (no change)" in verbose
    assert verbose.endswith("RESULT: 9 slots = lower bound -> PROVEN OPTIMAL (frame length = lower bound)")
    assert report == OFFICIAL_REPORT + "\n"  # and the report itself is exactly the same


def test_verbose_on_the_ring_shows_the_exact_search(capsys):
    out = run_main(capsys, "--file", str(example_path("ring16_r1150m")), "--verbose")[1]
    assert "RESULT: 4 slots, lower bound 3 -> exact search: 3 slots impossible -> PROVEN OPTIMAL (exact search)" in out
    assert "Optimality              : PROVEN OPTIMAL (exact search: 3 slots impossible)" in out


def test_verbose_on_grid_250m_shows_that_the_clique_bound_beats_maxdeg_plus_one(capsys):
    out = run_main(capsys, "--file", str(example_path("grid16_250m")), "--verbose")[1]
    assert re.search(r"LOWER BOUND: maxdeg\+1 = 11, max clique = 12 \{.*\} -> LB = 12", out)


def test_verbose_together_with_show_topology_keeps_the_order(capsys):
    out = run_main(capsys, "--file", GRID, "--show-topology", "--verbose")[1]
    assert out.index("TOPOLOGY SUMMARY") < out.index("HEURISTIC PORTFOLIO") < out.index("TDMA TOPOLOGY OPTIMIZATION REPORT")


def test_verbose_with_a_single_heuristic_only_leaves_a_note(capsys):
    code, out, err = run_main(capsys, "--file", GRID, "--algorithm", "dsatur", "--verbose")
    assert code == 0 and out == DSATUR_REPORT + "\n"
    assert err == "Note: --verbose shows the portfolio table for --algorithm auto only.\n"


def test_a_zero_time_limit_reports_the_open_gap_and_still_succeeds(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("ring16_r1150m")), "--time-limit", "0")
    assert (code, err) == (0, "")
    assert "Optimality              : BEST FOUND (gap = 1, time limit reached)" in out
    assert "Schedule verified conflict-free." in out


def test_a_generous_time_limit_changes_nothing_for_small_networks(capsys):
    assert run_main(capsys, "--file", GRID, "--time-limit", "60") == (0, OFFICIAL_REPORT + "\n", "")


@pytest.mark.parametrize("bad", ["-1", "nan", "inf"])
def test_an_invalid_time_limit_is_a_clean_error(capsys, bad):
    code, out, err = run_main(capsys, "--file", GRID, "--time-limit", bad)
    assert code == 2 and out == "" and err.startswith("Error: time limit must be") and err.count("\n") == 1


def test_the_time_limit_is_checked_before_the_input_is_read(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("hidden_terminal3")), "--time-limit", "-5")
    assert code == 2 and "Warning" not in err  # no "continuing anyway" before the error


def test_the_seed_can_be_given_and_gives_a_reproducible_valid_schedule(capsys):
    first = run_main(capsys, "--file", str(example_path("grid16_400m")), "--seed", "7")
    second = run_main(capsys, "--file", str(example_path("grid16_400m")), "--seed", "7")
    assert first == second and first[0] == 0 and "PROVEN OPTIMAL" in first[1]


@pytest.mark.parametrize("argv", [["--seed", "abc"], ["--seed", "1.5"], ["--time-limit", "soon"]])
def test_a_seed_or_time_limit_that_is_not_a_number_is_a_usage_error(capsys, argv):
    with pytest.raises(SystemExit) as info:
        main(["--file", GRID, *argv])
    assert info.value.code == 2 and "usage:" in capsys.readouterr().err


def test_export_json_is_still_written_when_verification_fails(capsys, tmp_path):
    target = tmp_path / "naive.schedule.json"
    code, out, _ = run_main(capsys, "--file", GRID, "--algorithm", "naive-d1", "--export-json", str(target))
    assert code == 1 and "VERIFICATION FAILED: 24 conflicts" in out
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["verified"] is False and data["frame_length"] == 4 and data["algorithm"] == "naive-d1"


@pytest.mark.parametrize("kind", ["parent is a file", "path is a folder"])
def test_export_json_to_an_unwritable_path_is_a_clean_error(capsys, tmp_path, kind):
    if kind == "parent is a file":
        (tmp_path / "afile").write_text("x", encoding="utf-8")
        target = tmp_path / "afile" / "schedule.json"
    else:
        target = tmp_path
    code, _, err = run_main(capsys, "--file", GRID, "--export-json", str(target))
    assert code == 2 and err.startswith("Error: cannot write ") and "Traceback" not in err


# --------------------------------------------------------------------------- input handling (Division 1)
@pytest.mark.parametrize(
    "x, links",
    [(500, 1), (500.0, 1), (499.999, 1), (500.001, 0)],
    ids=["500 int", "500.0", "499.999", "500.001"],
)
def test_500m_boundary_from_the_command_line(capsys, x, links):
    code, out, _ = run_main(capsys, f'{{"Node_01":[0,0],"Node_02":[{x},0]}}', "--show-topology")
    assert code == 0 and f"Radio links (1-hop)        : {links}\n" in out
    assert f"Optimized Frame Length  : {2 if links else 1} unique timeslots" in out  # linked radios need 2 slots


def test_range_option_changes_the_links(capsys):
    assert "Radio links (1-hop)        : 0\n" in run_main(capsys, "--file", GRID, "--show-topology", "--range", "250")[1]
    code, out, _ = run_main(capsys, "--file", GRID, "--show-topology", "--range", "300")
    assert "Radio links (1-hop)        : 24\n" in out and "Radio range                : 300.0 m" in out
    assert "Configured Radio Range  : 300.0 meters" in out


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (['{"Node_01":[0,"x"]}'], "Node_01"),  # the guide's failing example
        (["not json"], "not valid JSON"),
        (["{}"], "no nodes"),
        (['{"Node_01":[0,0],"Node_01":[1,1]}'], "duplicate"),
        (["[1,2]"], "JSON object"),
        ([], "no input"),
        (["--show-topology"], "no input"),
        (['{"Node_01":[0,0]}', "--file", "x.json"], "not both"),
        (["--file", "does_not_exist.json"], "file not found"),
        (['{"Node_01":[0,0]}', "--range", "nan"], "radio range"),
        (['{"Node_01":[0,0]}', "--range", "0"], "radio range"),
        (['{"Node_01":[0,0]}', "--range", "-5"], "radio range"),
    ],
)
def test_bad_input_is_a_clean_error_with_exit_code_2(capsys, argv, fragment):
    code, out, err = run_main(capsys, *argv)
    assert code == 2 and out == ""
    assert err.startswith("Error: ") and fragment in err
    assert "Traceback" not in err and err.count("\n") == 1  # one clean line, no traceback


def test_bad_range_fails_before_the_node_count_warning(capsys):
    # otherwise the user would read "continuing anyway" and then an error
    code, out, err = run_main(capsys, "--file", str(example_path("hidden_terminal3")), "--range", "0")
    assert code == 2 and out == ""
    assert "radio range" in err and "Warning" not in err


@pytest.mark.parametrize(
    "argv", [["--range", "abc", "{}"], ["--no-such-option"], ["--algorithm", "magic", "{}"]],
    ids=["range not a number", "unknown option", "unknown algorithm"],
)
def test_argparse_usage_errors_also_exit_with_2(capsys, argv):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_help_lists_the_options(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    for option in ("COORDS_JSON", "--file", "--range", "--algorithm", "auto", "dsatur", "rlf", "smallest_last",
                   "largest_first", "greedy", "naive-d1", "--time-limit", "--seed", "--verbose", "--export-json",
                   "--show-topology"):
        assert option in out
    assert "exit codes" in out


# --------------------------------------------------------------------------- real processes
def test_official_command_as_a_real_process():
    result = run_script("--file", "examples/grid16_300m.json")
    assert result.returncode == 0 and result.stderr == ""
    assert result.stdout == OFFICIAL_REPORT + "\n"


def test_official_input_as_one_json_argument_in_a_real_process():
    text = example_path("grid16_300m").read_text(encoding="utf-8").strip()  # quotes must survive the OS command line
    result = run_script(text)
    assert result.returncode == 0 and result.stdout == OFFICIAL_REPORT + "\n"


def test_naive_d1_exits_1_in_a_real_process():
    result = run_script("--file", "examples/grid16_300m.json", "--algorithm", "naive-d1")
    assert result.returncode == 1 and "VERIFICATION FAILED: 24 conflicts" in result.stdout
    assert "Traceback" not in result.stderr


def test_guides_failing_command_exits_2_without_a_traceback():
    result = run_script('{"Node_01":[0,"x"]}')
    assert result.returncode == 2
    assert "Traceback" not in result.stderr and "Node_01" in result.stderr and result.stdout == ""


def test_export_json_with_a_relative_path_in_a_real_process(tmp_path):
    result = run_script("--file", GRID, "--export-json", "results/grid16_300m.schedule.json", cwd=tmp_path)
    assert result.returncode == 0
    data = json.loads((tmp_path / "results" / "grid16_300m.schedule.json").read_text(encoding="utf-8"))
    assert data["verified"] is True and data["frame_length"] == 9


def test_python_dash_m_works_like_the_script():
    module = subprocess.run(
        [sys.executable, "-m", "tdma_brain", "--file", "examples/grid16_300m.json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert module.returncode == 0 and module.stdout == OFFICIAL_REPORT + "\n"


def test_script_works_from_any_working_directory(tmp_path):
    result = run_script("--file", GRID, cwd=tmp_path)
    assert result.returncode == 0 and result.stdout == OFFICIAL_REPORT + "\n"


@pytest.mark.parametrize("name", ["grid16_300m", "ring16_r1150m", "two_clusters16"])
def test_output_does_not_depend_on_python_hash_randomisation(name):
    outputs = {
        without_timings(
            run_script("--file", str(example_path(name)), "--show-topology", "--verbose", env_extra={"PYTHONHASHSEED": seed}).stdout
        )  # the milliseconds in the --verbose table are the only thing allowed to change
        for seed in ("0", "1", "2", "12345")
    }
    assert len(outputs) == 1
    text = outputs.pop()
    assert "TOPOLOGY SUMMARY" in text and "LOWER BOUND:" in text and "Schedule verified conflict-free." in text


@pytest.mark.parametrize(
    "name, key_lines",
    [
        ("grid16_300m", ["RESULT: 9 slots = lower bound -> PROVEN OPTIMAL (frame length = lower bound)"]),
        ("grid16_400m", ["RESULT: 5 slots = lower bound -> PROVEN OPTIMAL (frame length = lower bound)", "  RLF                                  5   <ms>"]),
        ("ring16_r1150m", ["RESULT: 4 slots, lower bound 3 -> exact search: 3 slots impossible -> PROVEN OPTIMAL (exact search)"]),
        ("grid16_250m", ["Lower Bound (proof)     : 12 timeslots (max 2-hop clique = 12, maxdeg+1 = 11)"]),
    ],
)
def test_the_guides_verbose_commands_as_real_processes(name, key_lines):
    result = run_script("--file", f"examples/{name}.json", "--verbose")
    assert result.returncode == 0 and result.stderr == ""
    lines = without_timings(result.stdout).split("\n")
    for line in key_lines:
        assert line in lines, line


def test_non_ascii_node_name_cannot_crash_an_ascii_only_console(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"Nöde_01": [0, 0]}', encoding="utf-8")
    result = run_script("--file", str(path), env_extra={"PYTHONIOENCODING": "ascii"})
    assert result.returncode == 0 and "Traceback" not in result.stderr
    assert "N\\xf6de_01: Slot 0" in result.stdout  # printed as an escape sequence, not a crash
    result.stdout.encode("ascii")
