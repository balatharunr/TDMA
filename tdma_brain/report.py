"""The official ASCII report (created in Division 2).

Prints a schedule in the brief's format: a summary block, the Node -> Slot assignments and
the Slot x Node Boolean matrix, then either the "verified" line or the list of conflicts.
ASCII only, because the Windows console crashes on characters such as arrows.
"""
import re

from .heuristics import NAIVE_D1_NAME

BAR = "=" * 64
RULE = "-" * 65
TITLE = " " * 15 + "TDMA TOPOLOGY OPTIMIZATION REPORT"
LABEL_WIDTH = 24  # summary labels are padded to this width, then ": "


def format_report(schedule, verification, radio_range):
    """Render the report as text (no trailing newline, no trailing spaces on any line).

    `verification` is the VerificationResult of verify_schedule(); a failed one replaces the
    final "verified" line by the list of conflicts. `radio_range` is in metres.
    """
    lines = [BAR, TITLE, BAR]
    lines += _summary(schedule, radio_range)
    lines.append(RULE)
    lines.append(" NODE -> SLOT ASSIGNMENTS:")
    lines += [f"  {node}: Slot {schedule.node_to_slot.get(node, '?')}" for node in schedule.nodes]
    lines.append("")
    lines.append(" STRUCTURAL TDMA SCHEDULE MATRIX (Slot x Node Boolean Matrix):")
    lines += _matrix(schedule)
    lines += _verdict(verification)
    lines.append(BAR)
    return "\n".join(line.rstrip() for line in lines)


def describe_algorithm(name):
    """The 'Scheduling Algorithm' text for a Schedule.algorithm name."""
    if name == NAIVE_D1_NAME:
        return "naive-d1: first-fit on the radio graph G only (UNSAFE demo)"
    return f"{name} on 2-hop conflict graph (G^2)"


def column_labels(nodes):
    """Matrix column labels, and whether they are 1-based indexes (a legend is then printed).

    A label is the trailing digits of the node name, zero-padded to 2 characters
    (Node_01 -> 01). If a name has no trailing digits, or two labels would collide, every
    column falls back to its 1-based index.
    """
    labels = []
    for node in nodes:
        digits = re.search(r"([0-9]+)\Z", str(node))
        labels.append(digits.group(1).zfill(2) if digits else None)
    if None in labels or len(set(labels)) != len(labels):
        return [f"{index:02d}" for index in range(1, len(nodes) + 1)], True
    return labels, False


def _summary(schedule, radio_range):
    # CONCEPT: TDMA (time slots, frames, frame length) - K unique timeslots form one frame that repeats; fewer slots means every radio gets to transmit more often
    rows = [
        ("Total Nodes Processed", str(len(schedule.nodes))),
        ("Configured Radio Range", f"{radio_range} meters"),
        ("Optimized Frame Length", f"{schedule.frame_length} unique timeslots (Lower is better)"),
        ("Scheduling Algorithm", describe_algorithm(schedule.algorithm)),
        ("Spatial Reuse Factor", f"{schedule.reuse_factor:.2f} nodes/slot (plain TDMA = 1.00)"),
    ]
    return [f"{label:<{LABEL_WIDTH}}: {value}" for label, value in rows]


def _matrix(schedule):
    """Header, separator, one row per slot, separator, and the legend when labels are indexes."""
    labels, is_legend = column_labels(schedule.nodes)
    width = max((len(label) for label in labels), default=2)
    width = max(width, 2)  # every cell is at least 2 wide: "01", "1 "
    slot_width = max(2, len(str(schedule.frame_length - 1)))
    prefix_width = len(" Slot ") + slot_width + 4  # 12 for up to 100 slots
    header = "Slot \\ Node".ljust(prefix_width) + " ".join("| " + label.ljust(width) for label in labels)
    rule = "-" * len(header.rstrip())
    lines = [header, rule]
    for slot, row in enumerate(schedule.matrix):
        # CONCEPT: Boolean schedule matrix - a row is a slot, a column is a radio, and a 1 means "this radio transmits in this slot"
        lines.append(f" Slot {slot:0{slot_width}d}    " + " ".join("| " + str(cell).ljust(width) for cell in row))
    lines.append(rule)
    if is_legend:
        lines.append("Legend: " + ", ".join(f"{label}={node}" for label, node in zip(labels, schedule.nodes)))
    return lines


def _verdict(verification):
    """The line that closes the report: verified, or the list of what went wrong."""
    if verification.ok:
        return ["Execution finalized cleanly. Schedule verified conflict-free."]
    if verification.violations:
        lines = [f"VERIFICATION FAILED: {len(verification.violations)} conflicts"]
    else:
        lines = [f"VERIFICATION FAILED: {len(verification.problems)} structural problems"]
    lines += [f"  PROBLEM: {problem}" for problem in verification.problems]
    lines += [f"  {violation.describe()}" for violation in verification.violations]
    return lines
