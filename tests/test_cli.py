"""Division 1 tests: the command line (tdma_brain.cli and the tdma_schedule.py entry point)."""
import os
import subprocess
import sys

import pytest

from tdma_brain.cli import main
from tests.golden import GOLDEN, REPO_ROOT, example_path
from tests.test_topology import GRID_300M_TEXT, HIDDEN_TERMINAL3_TEXT

SCRIPT = REPO_ROOT / "tdma_schedule.py"


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


# --------------------------------------------------------------------------- in-process
def test_show_topology_from_file(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("grid16_300m")), "--show-topology")
    assert (code, out, err) == (0, GRID_300M_TEXT + "\n", "")


def test_show_topology_from_json_string_matches_the_file(capsys):
    text = example_path("grid16_300m").read_text(encoding="utf-8").strip()
    code, out, err = run_main(capsys, text, "--show-topology")
    assert (code, out, err) == (0, GRID_300M_TEXT + "\n", "")


def test_show_topology_hidden_terminal3(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("hidden_terminal3")), "--show-topology")
    assert code == 0 and out == HIDDEN_TERMINAL3_TEXT + "\n"
    assert "3 node(s)" in err  # not 16 nodes: a warning, not an error


@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_every_example_reports_the_golden_counts(capsys, golden):
    code, out, _ = run_main(capsys, "--file", str(example_path(golden.name)), "--show-topology")
    assert code == 0
    assert f"Radio links (1-hop)        : {golden.links}\n" in out
    assert f"Conflict pairs (<= 2 hops) : {golden.conflict_pairs}\n" in out
    assert f"Hidden-terminal pairs      : {golden.hidden_pairs} (2-hop only)\n" in out


@pytest.mark.parametrize(
    "x, links",
    [(500, 1), (500.0, 1), (499.999, 1), (500.001, 0)],
    ids=["500 int", "500.0", "499.999", "500.001"],
)
def test_500m_boundary_from_the_command_line(capsys, x, links):
    code, out, _ = run_main(capsys, f'{{"Node_01":[0,0],"Node_02":[{x},0]}}', "--show-topology")
    assert code == 0 and f"Radio links (1-hop)        : {links}\n" in out


def test_range_option_changes_the_links(capsys):
    grid = str(example_path("grid16_300m"))
    assert "Radio links (1-hop)        : 0\n" in run_main(capsys, "--file", grid, "--show-topology", "--range", "250")[1]
    code, out, _ = run_main(capsys, "--file", grid, "--show-topology", "--range", "300")
    assert "Radio links (1-hop)        : 24\n" in out and "Radio range                : 300.0 m" in out


def test_without_show_topology_it_only_leaves_a_note_on_stderr(capsys):
    code, out, err = run_main(capsys, "--file", str(example_path("grid16_300m")))
    assert code == 0 and out == ""
    assert "--show-topology" in err


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


@pytest.mark.parametrize("argv", [["--range", "abc", "{}"], ["--no-such-option"]])
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
    for option in ("COORDS_JSON", "--file", "--range", "--show-topology"):
        assert option in out


# --------------------------------------------------------------------------- real processes
def test_official_command_as_a_real_process():
    result = run_script("--file", "examples/grid16_300m.json", "--show-topology")
    assert result.returncode == 0
    assert result.stdout == GRID_300M_TEXT + "\n"
    assert result.stderr == ""


def test_guides_failing_command_exits_2_without_a_traceback():
    result = run_script('{"Node_01":[0,"x"]}')  # the JSON quotes survive the OS command line
    assert result.returncode == 2
    assert "Traceback" not in result.stderr and "Node_01" in result.stderr and result.stdout == ""


def test_python_dash_m_works_like_the_script():
    module = subprocess.run(
        [sys.executable, "-m", "tdma_brain", "--file", "examples/grid16_300m.json", "--show-topology"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert module.returncode == 0 and module.stdout == GRID_300M_TEXT + "\n"


def test_script_works_from_any_working_directory(tmp_path):
    result = run_script("--file", str(example_path("grid16_300m")), "--show-topology", cwd=tmp_path)
    assert result.returncode == 0 and result.stdout == GRID_300M_TEXT + "\n"


@pytest.mark.parametrize("name", ["grid16_300m", "ring16_r1150m", "two_clusters16"])
def test_output_does_not_depend_on_python_hash_randomisation(name):
    outputs = {
        run_script("--file", str(example_path(name)), "--show-topology", env_extra={"PYTHONHASHSEED": seed}).stdout
        for seed in ("0", "1", "2", "12345")
    }
    assert len(outputs) == 1 and "TOPOLOGY SUMMARY" in outputs.pop()


def test_non_ascii_node_name_cannot_crash_an_ascii_only_console(tmp_path):
    path = tmp_path / "unicode.json"
    path.write_text('{"Nöde_01": [0, 0]}', encoding="utf-8")
    result = run_script("--file", str(path), "--show-topology", env_extra={"PYTHONIOENCODING": "ascii"})
    assert result.returncode == 0 and "Traceback" not in result.stderr
    assert "N\\xf6de_01" in result.stdout  # printed as an escape sequence, not a crash
    result.stdout.encode("ascii")
