"""Division 6: tests for the scenario generator and the pathloss bridge (Windows, no EMANE needed).

Both scripts are plain files, not a package, so they are loaded by path. They must stay standard-library only,
because the container's python3 has no tdma_brain and the generator is meant to be read as a standalone tool.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GRID_SCHEDULE = ROOT / "results" / "grid16_300m.schedule.json"
GEN_PATH = ROOT / "emane" / "gen" / "generate_scenario.py"
PATHLOSS_PATH = ROOT / "emane" / "bridge" / "pathloss.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module(GEN_PATH, "generate_scenario")
pathloss = load_module(PATHLOSS_PATH, "pathloss_bridge")


def params(elem: ET.Element) -> dict[str, str]:
    return {p.get("name"): p.get("value") for p in elem.findall("param")}


def small_schedule(nodes, links, frame_length=3):
    """A minimal tdma-schedule/v1 document (only the keys the generator reads matter)."""
    return {
        "schema": "tdma-schedule/v1",
        "generator": "test",
        "radio_range_m": 500.0,
        "nodes": list(nodes),
        "coordinates": {n: [0.0, 0.0] for n in nodes},
        "links": [list(pair) for pair in links],
        "frame_length": frame_length,
        "node_to_slot": {n: i % frame_length for i, n in enumerate(nodes)},
        "slot_to_nodes": [],
        "extra_tx_slots": {},
        "matrix": [],
        "lower_bound": None,
        "optimality": None,
        "algorithm": "test",
        "verified": True,
    }


@pytest.fixture(scope="module")
def grid_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("scenario") / "grid16_300m"
    gen.write_scenario(gen.load_schedule(GRID_SCHEDULE), out)
    return out


@pytest.mark.parametrize("path", [GEN_PATH, PATHLOSS_PATH])
def test_scripts_are_standard_library_only(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    imported.discard("emane")   # the python3-emane bindings, imported only inside publish()
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


def test_generator_writes_the_expected_files(grid_dir: Path) -> None:
    names = sorted(p.name for p in grid_dir.iterdir())
    expected = sorted([f"platform{i:02d}.xml" for i in range(1, 17)] + ["nemmap.json", "tdmanem.xml", "tdmaradiomodel.xml", "transvirtual.xml", "links.csv"])
    assert names == expected


def test_nemmap_is_in_natural_order_with_nem_ids_from_1(grid_dir: Path) -> None:
    nemmap = json.loads((grid_dir / "nemmap.json").read_text(encoding="utf-8"))
    assert list(nemmap) == [f"Node_{i:02d}" for i in range(1, 17)]
    for i, (name, entry) in enumerate(nemmap.items(), start=1):
        assert entry == {"nem": i, "netns": f"node{i:02d}", "backchan_ip": f"10.99.0.{i}", "radio_ip": f"10.100.0.{i}"}


def test_platform_files(grid_dir: Path) -> None:
    seen_ids, seen_ips = [], []
    for i in range(1, 17):
        root = ET.parse(grid_dir / f"platform{i:02d}.xml").getroot()
        assert root.tag == "platform"
        p = params(root)
        assert p["otamanagerchannelenable"] == "on"
        assert p["otamanagergroup"] == "224.1.2.8:45702"
        assert p["eventservicegroup"] == "224.1.2.8:45703"
        assert p["otamanagerdevice"] == p["eventservicedevice"] == "backchan0"
        assert p["controlportendpoint"] == "0.0.0.0:47000"
        nems = root.findall("nem")
        assert len(nems) == 1, "one NEM per emane process"
        assert nems[0].get("definition") == "tdmanem.xml"
        seen_ids.append(int(nems[0].get("id")))
        t = params(nems[0].find("transport"))
        assert t["device"] == "emane0" and t["mask"] == "255.255.255.0"
        seen_ips.append(t["address"])
    assert seen_ids == list(range(1, 17))
    assert seen_ips == [f"10.100.0.{i}" for i in range(1, 17)]
    assert len(set(seen_ips)) == 16


def test_shared_xml(grid_dir: Path) -> None:
    nem = ET.parse(grid_dir / "tdmanem.xml").getroot()
    assert nem.find("transport").get("definition") == "transvirtual.xml"
    assert nem.find("mac").get("definition") == "tdmaradiomodel.xml"
    phy = params(nem.find("phy"))
    assert phy["noisemode"] == "all"
    assert phy["propagationmodel"] == "precomputed"
    assert phy["subid"] == "7"
    assert phy["fading.model"] == "none"
    assert phy["bandwidth"] == "1M"
    mac = ET.parse(grid_dir / "tdmaradiomodel.xml").getroot()
    assert mac.get("library") == "tdmaeventschedulerradiomodel"
    assert params(mac)["pcrcurveuri"] == gen.DEFAULT_PCR_URI
    transport = ET.parse(grid_dir / "transvirtual.xml").getroot()
    assert transport.get("library") == "transvirtual"
    assert params(transport)["devicepath"] == "/dev/net/tun"


def test_links_csv_matches_the_schedule(grid_dir: Path) -> None:
    schedule = json.loads(GRID_SCHEDULE.read_text(encoding="utf-8"))
    with (grid_dir / "links.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["node_a", "node_b"]
    assert len(rows) - 1 == 42 == len(schedule["links"])
    assert {frozenset(r) for r in rows[1:]} == {frozenset(pair) for pair in schedule["links"]}


def test_files_use_lf_line_endings(grid_dir: Path) -> None:
    for path in grid_dir.iterdir():
        assert b"\r" not in path.read_bytes(), path


def test_generator_is_deterministic(tmp_path: Path) -> None:
    schedule = gen.load_schedule(GRID_SCHEDULE)
    gen.write_scenario(schedule, tmp_path / "a")
    gen.write_scenario(schedule, tmp_path / "b")
    for path in sorted((tmp_path / "a").iterdir()):
        assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()


def test_non_numeric_names_and_natural_order(tmp_path: Path) -> None:
    schedule = small_schedule(["Node_10", "Node_B", "Node_2"], [("Node_10", "Node_2")])
    gen.write_scenario(schedule, tmp_path)
    nemmap = json.loads((tmp_path / "nemmap.json").read_text(encoding="utf-8"))
    assert list(nemmap) == ["Node_2", "Node_10", "Node_B"]
    assert [e["nem"] for e in nemmap.values()] == [1, 2, 3]
    assert [e["netns"] for e in nemmap.values()] == ["node01", "node02", "node03"]
    assert sorted(p.name for p in tmp_path.glob("platform*.xml")) == ["platform01.xml", "platform02.xml", "platform03.xml"]


def test_three_digit_names_for_100_or_more_radios(tmp_path: Path) -> None:
    nodes = [f"N{i}" for i in range(1, 101)]
    gen.write_scenario(small_schedule(nodes, []), tmp_path)
    assert (tmp_path / "platform100.xml").exists() and (tmp_path / "platform001.xml").exists()
    assert gen.id_width(16) == 2 and gen.id_width(100) == 3


def test_too_many_radios_is_rejected(tmp_path: Path) -> None:
    nodes = [f"N{i}" for i in range(1, 255)]
    with pytest.raises(gen.ScenarioError):
        gen.write_scenario(small_schedule(nodes, []), tmp_path)


@pytest.mark.parametrize("breaker", ["wrong-schema", "missing-key", "unknown-link", "not-json"])
def test_bad_schedule_is_rejected(tmp_path: Path, breaker: str) -> None:
    schedule = small_schedule(["A", "B"], [("A", "B")])
    if breaker == "wrong-schema":
        schedule["schema"] = "tdma-schedule/v9"
    elif breaker == "missing-key":
        del schedule["links"]
    elif breaker == "unknown-link":
        schedule["links"] = [["A", "Z"]]
    path = tmp_path / "s.json"
    path.write_text("{not json" if breaker == "not-json" else json.dumps(schedule), encoding="utf-8")
    with pytest.raises(gen.ScenarioError):
        gen.load_schedule(path)


def test_cli(tmp_path: Path, capsys) -> None:
    assert gen.main(["--schedule", str(GRID_SCHEDULE), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "16 platform files" in out and "42 links" in out
    assert gen.main(["--schedule", str(tmp_path / "missing.json"), "--out", str(tmp_path / "x")]) == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------- pathloss bridge (pure parts)

def test_pathloss_matrix_mirrors_the_grid(grid_dir: Path) -> None:
    nemmap, links = pathloss.load_scenario(grid_dir)
    assert len(links) == 42
    matrix = pathloss.build_matrix(nemmap, links)
    assert len(matrix) == 16 * 15
    assert sum(1 for db in matrix.values() if db == 80.0) == 84      # every link counted in both directions
    assert sum(1 for db in matrix.values() if db == 200.0) == 156
    for (rx, tx), db in matrix.items():
        assert matrix[(tx, rx)] == db                                  # symmetric
    assert matrix[(1, 2)] == 80.0 and matrix[(1, 6)] == 80.0          # Node_01 hears Node_02 (300 m) and Node_06 (424 m)
    assert matrix[(1, 3)] == 200.0 and matrix[(1, 16)] == 200.0       # 600 m and 1273 m: out of range


def test_pathloss_matrix_hidden_terminal(tmp_path: Path) -> None:
    gen.write_scenario(small_schedule(["Node_A", "Node_B", "Node_C"], [("Node_A", "Node_B"), ("Node_B", "Node_C")]), tmp_path)
    nemmap, links = pathloss.load_scenario(tmp_path)
    matrix = pathloss.build_matrix(nemmap, links)
    assert matrix == {(1, 2): 80.0, (2, 1): 80.0, (2, 3): 80.0, (3, 2): 80.0, (1, 3): 200.0, (3, 1): 200.0}


def test_pathloss_csv(grid_dir: Path, tmp_path: Path) -> None:
    nemmap, links = pathloss.load_scenario(grid_dir)
    matrix = pathloss.build_matrix(nemmap, links)
    out = tmp_path / "pathloss_matrix.csv"
    pathloss.write_csv(matrix, nemmap, out)
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 17 and len(rows[0]) == 17
    assert rows[1][0] == "Node_01(nem1)" and rows[1][1] == "" and rows[1][2] == "80" and rows[1][3] == "200"
    assert b"\r" not in out.read_bytes()


EMANESH_SAMPLE = """nem 1   phy PathlossEventInfoTable
| NEM | Forward Pathloss | Reverse Pathloss |
| 2   | 80.0             | 80.0             |
| 3   | 200.0            | 200.0            |

"""


def test_parse_and_compare_emanesh_readback() -> None:
    rows = pathloss.parse_emanesh_table(EMANESH_SAMPLE)
    assert rows == [["2", "80.0", "80.0"], ["3", "200.0", "200.0"]]
    matrix = {(1, 2): 80.0, (1, 3): 200.0, (2, 1): 80.0, (2, 3): 80.0, (3, 1): 200.0, (3, 2): 80.0}
    assert pathloss.compare_readback(matrix, 1, rows) == []
    assert pathloss.compare_readback(matrix, 1, rows[:1]) == ["NEM 1: no entry for transmitter 3"]
    wrong = [["2", "80.0", "80.0"], ["3", "80.0", "80.0"]]
    assert pathloss.compare_readback(matrix, 1, wrong) == ["NEM 1: transmitter 3 has 80/80 dB, expected 200/200"]
    assert pathloss.compare_readback(matrix, 1, rows + [["9", "80.0", "80.0"]]) == ["NEM 1: unexpected entries for [9]"]
