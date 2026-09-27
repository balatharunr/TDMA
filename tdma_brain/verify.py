"""Independent verifier for TDMA schedules (created in Division 2).

Answers "is this schedule collision-free?" WITHOUT using the conflict graph H that the
schedulers colour. It re-derives who can collide from the radio graph G alone, by
breadth-first search, so a bug in the construction of H cannot hide a bad schedule. It also
checks that the schedule's own tables (slots, slot lists, matrix) agree with each other.
"""
from dataclasses import dataclass, field

import networkx as nx

from .io_utils import natural_key

DIRECT = "DIRECT (1-hop)"
HIDDEN_TERMINAL = "HIDDEN TERMINAL (2-hop)"
_MAX_LISTED = 8  # how many items of one kind of structural problem are spelled out


@dataclass(frozen=True)
class Violation:
    """Two radios within 2 hops of each other that transmit in the same slot."""

    u: str  # the earlier radio in natural order
    v: str
    slot: int
    kind: str  # DIRECT or HIDDEN_TERMINAL
    via: tuple = ()  # hidden terminal only: the radios that hear both u and v

    def describe(self):
        """One report line, e.g. 'Node_01 & Node_03 share Slot 0 -> HIDDEN TERMINAL (2-hop) via Node_02, Node_06'."""
        text = f"{self.u} & {self.v} share Slot {self.slot} -> {self.kind}"
        return f"{text} via {', '.join(self.via)}" if self.via else text


@dataclass
class VerificationResult:
    """The verdict: `ok` is True only when there are no conflicts and no structural problems."""

    ok: bool
    violations: list = field(default_factory=list)  # Violation objects, sorted in natural order
    problems: list = field(default_factory=list)  # text: missing node, empty slot, inconsistent matrix, ...


def verify_schedule(G, schedule):
    """Check `schedule` against the radio graph `G`; returns a VerificationResult.

    Two kinds of checks: (1) no two radios within 2 hops of each other transmit in a common
    slot, decided by BFS on G; (2) the schedule is well formed: every radio has a slot, the
    slots used are exactly 0..K-1, and slot_to_nodes and the matrix say the same thing.
    """
    problems = _structure_problems(G, schedule)
    violations = _conflicts(G, schedule)
    return VerificationResult(ok=not problems and not violations, violations=violations, problems=problems)


def _transmit_sets(schedule):
    """The slots each radio transmits in, as sets: its own slot plus any extra slots."""
    slots = {node: {slot} for node, slot in schedule.node_to_slot.items()}
    for node, extra in schedule.extra_tx_slots.items():
        slots.setdefault(node, set()).update(extra)
    return slots


def _conflicts(G, schedule):
    """Every pair of radios within 2 hops that share a transmit slot."""
    transmit = _transmit_sets(schedule)
    order = sorted(G.nodes, key=natural_key)
    rank = {v: i for i, v in enumerate(order)}
    violations = []
    for v in order:
        # CONCEPT: Independent verification / invariant checking - a breadth-first search on the radio graph G (never on H) finds every radio within 2 hops of v
        hops = nx.single_source_shortest_path_length(G, v, cutoff=2)
        for u, distance in hops.items():
            if distance == 0 or rank[u] < rank[v]:
                continue  # v itself, or a pair that was already looked at from the other side
            for slot in sorted(transmit.get(v, set()) & transmit.get(u, set())):
                # CONCEPT: Distance-1 vs distance-2 conflicts - 1 hop: the radios hear each other (half-duplex clash); 2 hops: a hidden terminal, both reach a common receiver
                if distance == 1:
                    violations.append(Violation(v, u, slot, DIRECT))
                else:
                    via = tuple(sorted(set(G[v]) & set(G[u]), key=natural_key))
                    violations.append(Violation(v, u, slot, HIDDEN_TERMINAL, via))
    violations.sort(key=lambda x: (rank[x.u], rank[x.v], x.slot))
    return violations


def _structure_problems(G, schedule):
    transmit = _transmit_sets(schedule)
    return _node_problems(G, schedule) + _slot_problems(G, schedule, transmit) + _table_problems(schedule, transmit)


def _node_problems(G, schedule):
    """The node list must be exactly the radios of G, once each, in natural order."""
    nodes = list(schedule.nodes)
    graph_nodes = set(G.nodes)
    problems = []
    duplicates = sorted({n for n in nodes if nodes.count(n) > 1}, key=natural_key)
    missing = sorted(graph_nodes - set(nodes), key=natural_key)
    unknown = sorted(set(nodes) - graph_nodes, key=natural_key)
    if duplicates:
        problems.append("node listed more than once: " + _listed(duplicates))
    if missing:
        problems.append("radio(s) missing from the schedule: " + _listed(missing))
    if unknown:
        problems.append("node(s) that are not in the radio graph: " + _listed(unknown))
    if not (duplicates or missing or unknown) and nodes != sorted(nodes, key=natural_key):
        problems.append("nodes are not in natural order")
    return problems


def _slot_problems(G, schedule, transmit):
    """Every radio has a slot in [0, K), and the slots in use are exactly 0..K-1."""
    problems = []
    without_slot = sorted(set(G.nodes) - set(schedule.node_to_slot), key=natural_key)
    if without_slot:
        problems.append("radio(s) without a slot: " + _listed(without_slot))
    stray = sorted(set(schedule.node_to_slot) - set(G.nodes), key=natural_key)
    if stray:
        problems.append("slot given to node(s) that are not in the radio graph: " + _listed(stray))

    k = schedule.frame_length
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        problems.append(f"frame_length must be a positive integer, got {k!r}")
        return problems
    used = set()
    out_of_range = []
    for node in sorted(transmit, key=natural_key):
        for slot in sorted(transmit[node], key=repr):
            if isinstance(slot, int) and not isinstance(slot, bool) and 0 <= slot < k:
                used.add(slot)
            else:
                out_of_range.append(f"{node} uses Slot {slot}")
    if out_of_range:
        problems.append(f"slot outside the frame 0..{k - 1}: " + _listed(out_of_range))
    empty = [s for s in range(k) if s not in used]
    if empty:
        problems.append("empty slot(s) that no radio uses: " + _listed(empty))
    return problems


def _table_problems(schedule, transmit):
    """slot_to_nodes and the matrix must say the same thing as node_to_slot (+ extra slots)."""
    problems = []
    nodes = list(schedule.nodes)
    k = schedule.frame_length
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        return problems  # already reported; the tables cannot be judged against a bad K

    if len(schedule.slot_to_nodes) != k:
        problems.append(f"slot_to_nodes has {len(schedule.slot_to_nodes)} entries but the frame has {k} slots")
    else:
        for slot, group in enumerate(schedule.slot_to_nodes):
            expected = [n for n in nodes if slot in transmit.get(n, ())]
            if list(group) != expected:
                problems.append(f"slot_to_nodes[{slot}] is {list(group)} but the radios that transmit in Slot {slot} are {expected}")

    # 3. matrix consistency: matrix[s][j] == 1  <=>  nodes[j] transmits in slot s; every column has a 1
    if len(schedule.matrix) != k:
        problems.append(f"matrix has {len(schedule.matrix)} rows but the frame has {k} slots")
    wrong_size = [s for s, row in enumerate(schedule.matrix) if len(row) != len(nodes)]
    if wrong_size:
        problems.append(f"matrix row(s) {_listed(wrong_size)} do not have one column per node ({len(nodes)})")
    mismatches = []
    for s, row in enumerate(schedule.matrix):
        if s in wrong_size:
            continue
        for j, cell in enumerate(row):
            transmits = s in transmit.get(nodes[j], ())
            if cell != (1 if transmits else 0):
                mismatches.append(f"matrix[{s}][{j}] is {cell} but {nodes[j]} {'transmits' if transmits else 'does not transmit'} in Slot {s}")
    if mismatches:
        problems.append("matrix disagrees with the slot assignment: " + _listed(mismatches))
    tidy_rows = [row for row in schedule.matrix if len(row) == len(nodes)]
    silent = [nodes[j] for j in range(len(nodes)) if not any(row[j] == 1 for row in tidy_rows)]
    if silent and tidy_rows:
        problems.append("matrix column(s) with no 1 (the radio never transmits): " + _listed(silent))
    return problems


def _listed(items):
    """Join the first few items with commas, and say how many more there are."""
    shown = ", ".join(str(item) for item in list(items)[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return shown + (f", ... and {extra} more" if extra > 0 else "")
