"""Schedule model for the TDMA planner (created in Division 2).

A Schedule says in which slot(s) of the repeating frame each radio transmits. This module
turns a raw colouring into a canonical schedule, computes its metrics, and reads and writes
the schedule.json contract that Part 2 (EMANE) consumes.
"""
import json
from dataclasses import dataclass, field

from . import __version__
from .io_utils import InputError, natural_key

SCHEMA = "tdma-schedule/v1"


@dataclass
class Schedule:
    """Who transmits when. Slot numbers run 0..frame_length-1 and repeat every frame."""

    nodes: list[str]  # natural order; also the column order of `matrix`
    node_to_slot: dict[str, int]  # each radio's own slot
    frame_length: int  # K, the number of slots in one frame
    slot_to_nodes: list[list[str]]  # slot_to_nodes[s] = the radios that transmit in slot s
    matrix: list[list[int]]  # K x N, matrix[s][j] == 1 when nodes[j] transmits in slot s
    algorithm: str  # name of the heuristic that produced it
    extra_tx_slots: dict[str, list[int]] = field(default_factory=dict)  # extra slots per radio (max-reuse, Division 3); empty for now

    @property
    def reuse_factor(self):
        """Radios per slot, N / K. Plain TDMA (one radio per slot) is 1.0."""
        # CONCEPT: Spatial reuse & Spatial TDMA (STDMA) - N radios share K slots, so on average N / K radios transmit at the same time; plain TDMA gives 1.00
        return len(self.nodes) / self.frame_length if self.frame_length else 0.0

    def occupancy(self):
        """How many radios transmit in each slot, indexed by slot."""
        return [len(group) for group in self.slot_to_nodes]

    def tx_slots(self, node):
        """Every slot in which `node` transmits: its own slot plus any extra slots."""
        return {self.node_to_slot[node], *self.extra_tx_slots.get(node, ())}

    @classmethod
    def from_json(cls, text):
        """Read a schedule from schedule.json text (the schedule fields only).

        The coordinates, links and verification flag in the file are for Part 2 and for
        humans; a Schedule does not keep them. Raises InputError if the text is not a
        tdma-schedule/v1 file.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(f"schedule is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise InputError(f'not a schedule file: expected "schema": "{SCHEMA}"')
        try:
            return cls(
                nodes=[str(n) for n in data["nodes"]],
                node_to_slot={str(n): int(s) for n, s in data["node_to_slot"].items()},
                frame_length=int(data["frame_length"]),
                slot_to_nodes=[[str(n) for n in group] for group in data["slot_to_nodes"]],
                matrix=[[int(cell) for cell in row] for row in data["matrix"]],
                algorithm=str(data["algorithm"]),
                extra_tx_slots={str(n): [int(s) for s in slots] for n, slots in data.get("extra_tx_slots", {}).items()},
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise InputError(f"schedule file is missing or has a malformed field: {exc!r}") from exc


def canonicalize(colouring, nodes):
    """Renumber the slots in order of first appearance, walking the nodes in natural order.

    Node_01 always becomes Slot 0, and the numbers 0..K-1 are all used. Raw colourings
    (DSATUR hands out 0 to the most constrained radio first) turn into a stable, readable
    schedule. Returns {node: slot}.
    """
    # CONCEPT: Graph colouring -> slot assignment - a colour IS a time slot; renumbering by first appearance changes the names of the slots, never who shares one
    renumber, slots = {}, {}
    for node in sorted(nodes, key=natural_key):
        if node not in colouring:
            raise ValueError(f"node {node!r} has no slot in the colouring")
        raw = colouring[node]
        if raw not in renumber:
            renumber[raw] = len(renumber)
        slots[node] = renumber[raw]
    return slots


def make_schedule(colouring, nodes, algorithm):
    """Build the canonical Schedule for a colouring of `nodes`."""
    nodes = sorted(nodes, key=natural_key)
    node_to_slot = canonicalize(colouring, nodes)
    # CONCEPT: TDMA (time slots, frames, frame length) - the K distinct slots form one frame that repeats forever; a shorter frame means every radio transmits more often
    frame_length = len(set(node_to_slot.values()))
    # CONCEPT: Broadcast (node) scheduling vs link scheduling - a slot belongs to a radio, which broadcasts to every neighbour, not to one transmitter-receiver pair
    slot_to_nodes = [[n for n in nodes if node_to_slot[n] == s] for s in range(frame_length)]
    # CONCEPT: Boolean schedule matrix - row = slot, column = radio; a 1 means that radio transmits in that slot
    matrix = [[1 if node_to_slot[n] == s else 0 for n in nodes] for s in range(frame_length)]
    return Schedule(
        nodes=nodes,
        node_to_slot=node_to_slot,
        frame_length=frame_length,
        slot_to_nodes=slot_to_nodes,
        matrix=matrix,
        algorithm=algorithm,
    )


def to_json(schedule, G, *, verified, lower_bound=None, optimality=None):
    """Render the schedule.json contract (tdma-schedule/v1) as text.

    `G` is the radio graph the schedule was made for: its coordinates, links and radio range
    go into the file so Part 2 needs nothing else. `verified` is the verifier's verdict;
    `lower_bound` and `optimality` stay null until Division 3.
    """
    data = {
        "schema": SCHEMA,
        "generator": f"tdma_brain {__version__}",
        "radio_range_m": G.graph.get("radio_range"),
        "nodes": list(schedule.nodes),
        "coordinates": {n: list(G.nodes[n]["pos"]) for n in schedule.nodes},
        "links": _links(G),
        "frame_length": schedule.frame_length,
        "node_to_slot": {n: schedule.node_to_slot[n] for n in schedule.nodes},
        "slot_to_nodes": schedule.slot_to_nodes,
        "extra_tx_slots": schedule.extra_tx_slots,
        "matrix": schedule.matrix,
        "lower_bound": lower_bound,
        "optimality": optimality,
        "algorithm": schedule.algorithm,
        "verified": bool(verified),
    }
    return _dumps(data)


def _links(G):
    """The links of G as [u, v] pairs with u before v in natural order, sorted."""
    pairs = [sorted(edge, key=natural_key) for edge in G.edges]
    return sorted(pairs, key=lambda pair: (natural_key(pair[0]), natural_key(pair[1])))


def _dumps(data):
    """JSON text that is pleasant to read: one key per line, lists compact, the matrix one row per line."""
    lines = []
    for key, value in data.items():
        if key == "matrix" and value:
            body = "[\n" + ",\n".join("    " + json.dumps(row) for row in value) + "\n  ]"
        else:
            body = json.dumps(value, allow_nan=False)
        lines.append(f"  {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(lines) + "\n}\n"
