"""Division 5: sanity checks on the hand-written EMANE smoke-test files (run on Windows, no EMANE needed).

They catch the classic Part 2 time-wasters before anything is copied into the container:
malformed XML, a wrong NEM id or address, a schedule that EMANE's XSD would reject, and CRLF line
endings in bash scripts.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "emane" / "smoke"
XML_FILES = ["platform1.xml", "platform2.xml", "tdmanem.xml", "tdmaradiomodel.xml", "transvirtual.xml", "schedule.xml"]


def params(elem: ET.Element) -> dict[str, str]:
    return {p.get("name"): p.get("value") for p in elem.findall("param")}


@pytest.mark.parametrize("name", XML_FILES)
def test_smoke_xml_is_well_formed(name: str) -> None:
    ET.parse(SMOKE / name)  # raises ParseError on malformed XML


@pytest.mark.parametrize("nem_id", [1, 2])
def test_platform_files(nem_id: int) -> None:
    root = ET.parse(SMOKE / f"platform{nem_id}.xml").getroot()
    assert root.tag == "platform"
    p = params(root)
    assert p["otamanagerchannelenable"] == "on"
    assert p["otamanagergroup"] == "224.1.2.8:45702"
    assert p["eventservicegroup"] == "224.1.2.8:45703"
    assert p["otamanagerdevice"] == p["eventservicedevice"] == "backchan0"
    assert p["controlportendpoint"] == "0.0.0.0:47000"
    nems = root.findall("nem")
    assert len(nems) == 1, "one NEM per emane process"
    nem = nems[0]
    assert nem.get("id") == str(nem_id)
    assert nem.get("definition") == "tdmanem.xml"
    t = params(nem.find("transport"))
    assert t == {"device": "emane0", "address": f"10.100.0.{nem_id}", "mask": "255.255.255.0"}


def test_nem_phy_settings() -> None:
    root = ET.parse(SMOKE / "tdmanem.xml").getroot()
    assert root.find("transport").get("definition") == "transvirtual.xml"
    assert root.find("mac").get("definition") == "tdmaradiomodel.xml"
    phy = params(root.find("phy"))
    assert phy["noisemode"] == "all", "collisions are only modelled with noisemode=all"
    assert phy["propagationmodel"] == "precomputed", "topology must come from our pathloss events"
    assert phy["subid"] == "7"
    assert phy["fading.model"] == "none"


def test_radio_model() -> None:
    root = ET.parse(SMOKE / "tdmaradiomodel.xml").getroot()
    assert root.get("library") == "tdmaeventschedulerradiomodel"
    p = params(root)
    assert p["pcrcurveuri"] == "file:///usr/share/emane/xml/models/mac/tdmaeventscheduler/tdmabasemodelpcr.xml"
    assert p["queue.strictdequeueenable"] == "off"  # the real parameter name (emane.io prose says stricttxdequeue)


def test_transport() -> None:
    root = ET.parse(SMOKE / "transvirtual.xml").getroot()
    assert root.get("library") == "transvirtual"
    assert params(root)["devicepath"] == "/dev/net/tun"


def test_schedule_follows_the_xsd_rules() -> None:
    root = ET.parse(SMOKE / "schedule.xml").getroot()
    assert root.tag == "emane-tdma-schedule"
    structure = root.find("structure")
    assert structure.attrib == {"frames": "1", "slots": "2", "slotoverhead": "100", "slotduration": "1000", "bandwidth": "1M"}
    multiframes = root.findall("multiframe")
    assert len(multiframes) == 1
    mf = multiframes[0]
    assert "frames" not in mf.attrib, "<multiframe> has no frames attribute; adding one fails validation"
    assert mf.attrib == {"frequency": "2.4G", "power": "0", "class": "0", "datarate": "11M"}
    frames = mf.findall("frame")
    assert [f.get("index") for f in frames] == ["0"]
    slots = frames[0].findall("slot")
    assert [(s.get("index"), s.get("nodes")) for s in slots] == [("0", "1"), ("1", "2")]
    node_pattern = re.compile(r"^([1-9][0-9]*,?|[1-9][0-9]*:[1-9][0-9]*,?)+$")  # NodeType in tdmaschedule.xsd
    for s in slots:
        assert node_pattern.match(s.get("nodes")), "no spaces allowed in nodes"
        assert len(list(s)) == 0, "no child element means TX"


@pytest.mark.parametrize("script", sorted((ROOT / "emane").rglob("*.sh")))
def test_shell_scripts_use_lf_line_endings(script: Path) -> None:
    data = script.read_bytes()
    assert b"\r" not in data, f"{script} has CRLF line endings; bash would fail with $'\\r': command not found"
    assert data.startswith(b"#!/usr/bin/env bash")
