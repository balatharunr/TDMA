"""Division 1-2 tests: the command line (tdma_brain.cli and the tdma_schedule.py entry point)."""
import json
import os
import subprocess
import sys

import pytest

from tdma_brain.cli import compute_schedule, main
from tdma_brain.schedule import Schedule
from tests.golden import GOLDEN, REPO_ROOT, example_path
from tests.helpers import build_graphs
from tests.test_topology import GRID_300M_TEXT, HIDDEN_TERMINAL3_TEXT

SCRIPT = REPO_ROOT / "tdma_schedule.py"
# the guide's expected output for the official input (copied by script), without the final newline
OFFICIAL_REPORT = (REPO_ROOT / "tests" / "fixtures" / "grid16_300m_report.txt").read_text(encoding="utf-8").rstrip("\n")
GRID = str(example_path("grid16_300m"))


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


# --------------------------------------------------------------------------- the official run (Division 2)
def test_default_run_prints_the_official_report_from_a_file(capsys):
    code, out, err = run_main(capsys, "--file", GRID)
    assert (code, out, err) == (0, OFFICIAL_REPORT + "\n", "")  # stdout is the report and nothing else


def test_default_run_from_the_json_string_gives_the_same_report(capsys):
    text = example_path("grid16_300m").read_text(encoding="utf-8").strip()  # the brief's input, as one argument
    assert run_main(capsys, text) == (0, OFFICIAL_REPORT + "\n", "")


def test_dsatur_is_the_default_algorithm(capsys):
    assert run_main(capsys, "--file", GRID, "--algorithm", "dsatur")[1] == OFFICIAL_REPORT + "\n"


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
        (["--file", GRID, "--algorithm", "naive-d1"], 1),
        (["--file", str(example_path("isolated16_1000m")), "--algorithm", "naive-d1"], 0),
        (['{"Node_01":[0,"x"]}'], 2),
        (["--file", "does_not_exist.json"], 2),
    ],
    ids=["dsatur ok", "greedy ok", "naive-d1 fails", "naive-d1 harmless", "bad input", "missing file"],
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
    G, H = build_graphs("grid16_300m")
    assert Schedule.from_json(text) == compute_schedule(G, H, "dsatur")
    assert b"\r" not in target.read_bytes()  # LF endings even on Windows: Part 2 runs on Linux


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
    for option in ("COORDS_JSON", "--file", "--range", "--algorithm", "dsatur", "naive-d1", "--export-json", "--show-topology"):
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
        run_script("--file", str(example_path(name)), "--show-topology", env_extra={"PYTHONHASHSEED": seed}).stdout
        for seed in ("0", "1", "2", "12345")
    }
    assert len(outputs) == 1
    text = outputs.pop()
    assert "TOPOLOGY SUMMARY" in text and "Schedule verified conflict-free." in text


def test_non_ascii_node_name_cannot_crash_an_ascii_only_console(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"Nöde_01": [0, 0]}', encoding="utf-8")
    result = run_script("--file", str(path), env_extra={"PYTHONIOENCODING": "ascii"})
    assert result.returncode == 0 and "Traceback" not in result.stderr
    assert "N\\xf6de_01: Slot 0" in result.stdout  # printed as an escape sequence, not a crash
    result.stdout.encode("ascii")
