#!/usr/bin/env python3
"""Division 6: make EMANE's radio connectivity identical to our 500 m graph, with precomputed pathloss events.

    python3 emane/bridge/pathloss.py --scenario emane/scenario/grid16_300m --publish --verify \
        --csv results/emane/pathloss_matrix.csv --readback-out results/emane/d6_pathloss_readback.txt

For every receiver r one PathlossEvent is published to NEM r. It holds an entry for EVERY other NEM t:
    forward = 80 dB   if (r, t) is a link in links.csv  -> 0 dBm - 80 dB = -80 dBm at r: 30 dB above sensitivity
    forward = 200 dB  otherwise                          -> -200 dBm: not decodable AND adds nothing to the noise
Explicit entries for all pairs, because EMANE never deletes a pathloss entry: a later update (mobility) just overwrites.

Only --publish needs the emane.events Python bindings (Linux, inside the container); the matrix, the CSV and the
emanesh table parser are plain Python, unit-tested on Windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

LINK_DB = 80.0
NOLINK_DB = 200.0
EVENT_GROUP = "224.1.2.8"
EVENT_PORT = 45703
DEVICE = "tdmabr0"
PATHLOSS_TABLE = "PathlossEventInfoTable"


def load_scenario(scenario_dir: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Read nemmap.json and links.csv written by emane/gen/generate_scenario.py."""
    scenario_dir = Path(scenario_dir)
    nemmap = json.loads((scenario_dir / "nemmap.json").read_text(encoding="utf-8"))
    with (scenario_dir / "links.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["node_a", "node_b"]:
            raise ValueError(f"{scenario_dir / 'links.csv'}: unexpected header {header}")
        links = [(a, b) for a, b in reader]
    unknown = sorted({n for pair in links for n in pair if n not in nemmap})
    if unknown:
        raise ValueError(f"links.csv mentions nodes missing from nemmap.json: {unknown}")
    return nemmap, links


def build_matrix(nemmap: dict, links, link_db: float = LINK_DB, nolink_db: float = NOLINK_DB) -> dict[tuple[int, int], float]:
    """(receiver NEM, transmitter NEM) -> pathloss in dB, for every ordered pair of distinct NEMs."""
    nems = sorted(entry["nem"] for entry in nemmap.values())
    linked = set()
    for a, b in links:
        na, nb = nemmap[a]["nem"], nemmap[b]["nem"]
        linked.add((na, nb))
        linked.add((nb, na))
    # CONCEPT: Pathloss & propagation models - with propagationmodel=precomputed EMANE has no idea where the radios
    # are: the ONLY thing deciding whether NEM t is heard at NEM r is this table entry (dB), so we set it from our
    # own 500 m graph. In range = 80 dB (a clean link), out of range = 200 dB (silence, not even interference).
    return {(rx, tx): (link_db if (rx, tx) in linked else nolink_db) for rx in nems for tx in nems if rx != tx}


def write_csv(matrix: dict[tuple[int, int], float], nemmap: dict, path: Path) -> None:
    """Receiver x transmitter matrix of pathloss (dB), rows and columns labelled by node name and NEM id."""
    by_nem = {entry["nem"]: name for name, entry in nemmap.items()}
    nems = sorted(by_nem)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["rx \\ tx (dB)"] + [f"{by_nem[tx]}(nem{tx})" for tx in nems])
        for rx in nems:
            writer.writerow([f"{by_nem[rx]}(nem{rx})"] + ["" if rx == tx else f"{matrix[(rx, tx)]:g}" for tx in nems])


def publish(matrix: dict[tuple[int, int], float], group: str = EVENT_GROUP, port: int = EVENT_PORT, device: str = DEVICE) -> int:
    """Publish one PathlossEvent per receiver on the event channel. Returns the number of events published."""
    from emane.events import EventService, PathlossEvent  # Linux only: the python3-emane bindings

    service = EventService((group, port, device))
    receivers = sorted({rx for rx, _ in matrix})
    for rx in receivers:
        event = PathlossEvent()
        for tx in sorted(tx for r, tx in matrix if r == rx):
            db = matrix[(rx, tx)]
            # CONCEPT: Receiver sensitivity & SINR - the receiver's PHY computes rx power = tx power - pathloss and
            # compares it with its -110 dBm sensitivity; a link (-80 dBm) sits 30 dB above it, far beyond the 8 dB
            # where the 11 Mbps PCR curve reaches 100 % delivery; a non-link (-200 dBm) is below the noise floor.
            event.append(tx, forward=db, reverse=db)
        service.publish(rx, event)   # the TARGET of a pathloss event is the receiver; each entry is a transmitter
    return len(receivers)


def parse_emanesh_table(text: str) -> list[list[str]]:
    """Rows of an emanesh table dump as lists of cell strings, header rows excluded."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == "NEM":
            continue   # the header row
        rows.append(cells)
    return rows


def expected_rows(matrix: dict[tuple[int, int], float], rx: int) -> dict[int, tuple[float, float]]:
    """What NEM rx's PathlossEventInfoTable must show: transmitter -> (forward dB, reverse dB)."""
    return {tx: (db, db) for (r, tx), db in matrix.items() if r == rx}


def compare_readback(matrix: dict[tuple[int, int], float], rx: int, rows: list[list[str]]) -> list[str]:
    """Mismatches between the expected pathloss table of NEM rx and the rows read back from emanesh (empty = OK)."""
    seen: dict[int, tuple[float, float]] = {}
    problems: list[str] = []
    for cells in rows:
        try:
            seen[int(cells[0])] = (float(cells[1]), float(cells[2]))
        except (IndexError, ValueError):
            problems.append(f"NEM {rx}: unreadable row {cells}")
    for tx, (fwd, rev) in expected_rows(matrix, rx).items():
        got = seen.get(tx)
        if got is None:
            problems.append(f"NEM {rx}: no entry for transmitter {tx}")
        elif abs(got[0] - fwd) > 1e-6 or abs(got[1] - rev) > 1e-6:
            problems.append(f"NEM {rx}: transmitter {tx} has {got[0]:g}/{got[1]:g} dB, expected {fwd:g}/{rev:g}")
    extra = sorted(set(seen) - set(expected_rows(matrix, rx)))
    if extra:
        problems.append(f"NEM {rx}: unexpected entries for {extra}")
    return problems


def readback(matrix: dict[tuple[int, int], float], nemmap: dict, timeout_s: float = 5.0) -> tuple[int, int, list[str], str]:
    """Read every radio's PathlossEventInfoTable with emanesh and compare it with the matrix.

    Returns (radios OK, radios total, problems, report text)."""
    by_nem = {entry["nem"]: (name, entry["backchan_ip"]) for name, entry in nemmap.items()}
    ok = 0
    problems: list[str] = []
    lines = [f"{'NEM':>3}  {'Node':<10} {'entries':>7}  {'80 dB':>5}  {'200 dB':>6}  match"]
    for rx in sorted(by_nem):
        name, ip = by_nem[rx]
        # CONCEPT: EMANE control port & emanesh - every radio answers TCP 47000 on its backplane IP; "get table"
        # dumps the PHY's pathloss table, so we can prove the topology EMANE holds is exactly ours.
        try:
            proc = subprocess.run(["emanesh", ip, "get", "table", "nems", "phy", PATHLOSS_TABLE],
                                  capture_output=True, text=True, timeout=timeout_s)
            text = proc.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            text = ""
            problems.append(f"NEM {rx}: emanesh failed: {exc}")
        rows = parse_emanesh_table(text)
        issues = compare_readback(matrix, rx, rows)
        problems.extend(issues)
        n80 = sum(1 for c in rows if len(c) > 1 and c[1] in ("80", "80.0"))
        n200 = sum(1 for c in rows if len(c) > 1 and c[1] in ("200", "200.0"))
        lines.append(f"{rx:>3}  {name:<10} {len(rows):>7}  {n80:>5}  {n200:>6}  {'OK' if not issues else 'MISMATCH'}")
        ok += not issues
    lines.append(f"{ok}/{len(by_nem)} radios hold exactly the planned pathloss table")
    lines.extend(problems)
    return ok, len(by_nem), problems, "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish (and verify) the pathloss events that mirror the 500 m radio graph.")
    parser.add_argument("--scenario", required=True, type=Path, help="scenario directory with nemmap.json and links.csv")
    parser.add_argument("--publish", action="store_true", help="publish one pathloss event per receiver (needs emane.events)")
    parser.add_argument("--verify", action="store_true", help="read every radio's PathlossEventInfoTable back with emanesh and compare")
    parser.add_argument("--csv", type=Path, help="also write the receiver x transmitter matrix to this CSV file")
    parser.add_argument("--readback-out", type=Path, help="with --verify: save the read-back report to this file")
    parser.add_argument("--link-db", type=float, default=LINK_DB, help="pathloss of a link in dB (default: %(default)s)")
    parser.add_argument("--nolink-db", type=float, default=NOLINK_DB, help="pathloss of a non-link in dB (default: %(default)s)")
    parser.add_argument("--group", default=EVENT_GROUP)
    parser.add_argument("--port", type=int, default=EVENT_PORT)
    parser.add_argument("--device", default=DEVICE, help="event channel multicast device in the root namespace (default: %(default)s)")
    parser.add_argument("--settle", type=float, default=0.5, help="seconds to wait between publishing and verifying")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        nemmap, links = load_scenario(args.scenario)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    matrix = build_matrix(nemmap, links, args.link_db, args.nolink_db)
    n = len(nemmap)
    n_link = sum(1 for db in matrix.values() if db == args.link_db)
    print(f"pathloss matrix: {n} radios, {len(links)} links -> {len(matrix)} directed entries "
          f"({n_link} at {args.link_db:g} dB, {len(matrix) - n_link} at {args.nolink_db:g} dB)")
    if args.csv:
        write_csv(matrix, nemmap, args.csv)
        print(f"matrix saved to {args.csv}")
    if args.publish:
        count = publish(matrix, args.group, args.port, args.device)
        print(f"published {count} pathloss events (one per receiver) on {args.group}:{args.port} via {args.device}")
    if args.verify:
        if args.publish:
            time.sleep(args.settle)
        ok, total, problems, report = readback(matrix, nemmap)
        print(report, end="")
        if args.readback_out:
            args.readback_out.parent.mkdir(parents=True, exist_ok=True)
            args.readback_out.write_text(report, encoding="utf-8")
            print(f"read-back report saved to {args.readback_out}")
        if problems:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
