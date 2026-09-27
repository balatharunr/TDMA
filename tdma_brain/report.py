"""The official ASCII report (Division 2) and the --verbose block (Division 3).

Prints a schedule in the brief's format: a summary block, the Node -> Slot assignments and
the Slot x Node Boolean matrix, then either the "verified" line or the list of conflicts.
ASCII only, because the Windows console crashes on characters such as arrows.
"""
import re

from .exact import BEST_FOUND, PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH
from .heuristics import NAIVE_D1_NAME

BAR = "=" * 64
RULE = "-" * 65
TITLE = " " * 15 + "TDMA TOPOLOGY OPTIMIZATION REPORT"
LABEL_WIDTH = 24  # summary labels are padded to this width, then ": "


def format_report(schedule, verification, radio_range, optimization=None):
    """Render the report as text (no trailing newline, no trailing spaces on any line).

    `verification` is the VerificationResult of verify_schedule(); a failed one replaces the
    final "verified" line by the list of conflicts. `radio_range` is in metres. With an
    OptimizationResult (from optimize()) two lines are added to the summary: the proven lower
    bound and the optimality status.
    """
    lines = [BAR, TITLE, BAR]
    lines += _summary(schedule, radio_range, optimization)
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


def optimality_text(status, frame_length, lower_bound):
    """The text after 'Optimality' for a proof status (an exact.py constant)."""
    # CONCEPT: Optimality gap / proof status - frame length = lower bound proves optimality; otherwise the exact search either proves one slot fewer impossible or gives up, leaving a gap of (frame length - lower bound)
    if status == PROVEN_OPTIMAL_BY_BOUND:
        return "PROVEN OPTIMAL (frame length = lower bound)"
    if status == PROVEN_OPTIMAL_BY_SEARCH:
        return f"PROVEN OPTIMAL (exact search: {frame_length - 1} slots impossible)"
    if status == BEST_FOUND:
        return f"BEST FOUND (gap = {frame_length - lower_bound}, time limit reached)"
    raise ValueError(f"unknown optimality status {status!r}")


def format_verbose(optimization):
    """The --verbose block: the portfolio table, the lower bound and the result (times vary run to run)."""
    o = optimization
    bound, k = o.lower_bound, o.schedule.frame_length
    lines = [
        f"HEURISTIC PORTFOLIO (conflict graph: {o.node_count} nodes, {o.conflict_pairs} conflict pairs)",
        f"  {'Heuristic':<33}{'Slots':>5}   {'Time (ms)':>9}",
    ]
    lines += [f"  {run.name:<33}{run.slots:>5}   {run.ms:>9.1f}" for run in o.portfolio]
    change = "no change" if o.polished_slots == o.best_slots else f"improved from {o.best_slots}"
    lines.append(f"  -> best: {o.best_name} ; iterated greedy ({o.iterations} it): {o.polished_slots} ({change})")
    clique = (
        f"max clique = {bound.clique_size} {{{', '.join(bound.clique_nodes)}}}"
        if bound.clique_size is not None
        else "max clique skipped (more than 300 nodes)"
    )
    lines.append(f"LOWER BOUND: maxdeg+1 = {bound.maxdeg_plus_1}, {clique} -> LB = {bound.value}")
    if o.exact_start is not None and k < o.exact_start:
        lines.append(f"  -> exact search improved {o.exact_start} to {k} slots")
    if o.status == PROVEN_OPTIMAL_BY_BOUND:
        lines.append(f"RESULT: {k} slots = lower bound -> {optimality_text(o.status, k, bound.value)}")
    elif o.status == PROVEN_OPTIMAL_BY_SEARCH:
        lines.append(f"RESULT: {k} slots, lower bound {bound.value} -> exact search: {k - 1} slots impossible -> PROVEN OPTIMAL (exact search)")
    else:
        lines.append(f"RESULT: {k} slots, lower bound {bound.value} -> {optimality_text(o.status, k, bound.value)}")
    return "\n".join(line.rstrip() for line in lines)


def _lower_bound_text(bound):
    clique = (
        f"max 2-hop clique = {bound.clique_size}"
        if bound.clique_size is not None
        else "max 2-hop clique skipped: more than 300 nodes"
    )
    return f"{bound.value} timeslots ({clique}, maxdeg+1 = {bound.maxdeg_plus_1})"


def _summary(schedule, radio_range, optimization=None):
    # CONCEPT: TDMA (time slots, frames, frame length) - K unique timeslots form one frame that repeats; fewer slots means every radio gets to transmit more often
    rows = [
        ("Total Nodes Processed", str(len(schedule.nodes))),
        ("Configured Radio Range", f"{radio_range} meters"),
        ("Optimized Frame Length", f"{schedule.frame_length} unique timeslots (Lower is better)"),
        ("Scheduling Algorithm", describe_algorithm(schedule.algorithm)),
        ("Spatial Reuse Factor", f"{schedule.reuse_factor:.2f} nodes/slot (plain TDMA = 1.00)"),
    ]
    if optimization is not None:
        bound = optimization.lower_bound
        rows += [
            ("Lower Bound (proof)", _lower_bound_text(bound)),
            ("Optimality", optimality_text(optimization.status, schedule.frame_length, bound.value)),
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
