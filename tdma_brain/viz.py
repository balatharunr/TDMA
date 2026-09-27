"""Plots for the TDMA planner (created in Division 4): topology by slot, and a frame timeline.

Uses matplotlib with the Agg backend (files only, no display window). Colours come verbatim
from a validated colour-blind-safe categorical palette (8 hues); beyond 8 slots the marker
shape changes too (composite encoding), because generating new hues stops being distinguishable.
Every node also carries a text label with its exact slot number, so identity never rests on
colour alone.
"""
import os

import matplotlib

matplotlib.use("Agg")  # save PNG files; never try to open a window (this runs headless and in tests)
import matplotlib.colors as mcolors  # noqa: E402  (must follow matplotlib.use())
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .exact import BEST_FOUND, PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH  # noqa: E402
from .io_utils import InputError, natural_key  # noqa: E402
from .report import column_labels  # noqa: E402

# CONCEPT: Data visualisation with matplotlib - one small, named style shared by every plot in
# this module (and by bench/benchmark.py), instead of scattering literal colours through the code
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_LINE = "#e1e0d9"
BASELINE = "#c3c2b7"
# the 8-hue categorical palette (validated: every adjacent pair clears the colour-blind-safe
# threshold in both light and dark mode); order is fixed and never reassigned per chart
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")  # cycle of shapes; combined with CATEGORICAL this gives 64 slots
ACCENT_1HOP = CATEGORICAL[0]  # blue: reserved for "1-hop neighbour of the highlighted node"
ACCENT_2HOP = CATEGORICAL[1]  # orange: reserved for "2-hop neighbour of the highlighted node"

_STATUS_TEXT = {
    PROVEN_OPTIMAL_BY_BOUND: "proven optimal (frame = lower bound)",
    PROVEN_OPTIMAL_BY_SEARCH: "proven optimal (exact search)",
    BEST_FOUND: "best found (gap open)",
}


def slot_style(slot):
    """(colour, marker) for a slot number: the 8 hues cycle first, then the marker shape changes.

    Never invents a new hue: slots 0-7 are 8 colours with a circle marker, 8-15 are the same 8
    colours with a square, and so on, so two different slots are always visually distinct.
    """
    colour = CATEGORICAL[slot % len(CATEGORICAL)]
    marker = MARKERS[(slot // len(CATEGORICAL)) % len(MARKERS)]
    return colour, marker


def plot_topology(G, schedule, path, highlight=None, lower_bound=None, status=None):
    """Save a picture of the radios at their real coordinates, coloured by slot.

    Links are thin grey lines behind the nodes. Every node is labelled "NN/Sk" (its column
    label and its slot), so the exact slot is always readable even in grayscale. `lower_bound`
    and `status` (an exact.py status constant) are optional and only decorate the title.
    With `highlight` (a node name in G), a dashed circle of radius `G.graph["radio_range"]` is
    drawn around it, and its 1-hop and 2-hop neighbours get a coloured ring.
    """
    if highlight is not None and highlight not in G:
        raise InputError(f"--highlight: {highlight!r} is not one of the radios ({G.number_of_nodes()} given)")

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    positions = {n: G.nodes[n]["pos"] for n in G.nodes}

    # CONCEPT: Visualisation - links behind, thin and muted, so the coloured nodes (the data) stay the loudest thing on the page
    for u, v in G.edges:
        (x1, y1), (x2, y2) = positions[u], positions[v]
        ax.plot([x1, x2], [y1, y2], color=GRID_LINE, linewidth=1.0, zorder=1, solid_capstyle="round")

    if highlight is not None:
        _draw_highlight(ax, G, positions, highlight)

    labels, is_legend = column_labels(schedule.nodes)  # reuses the report's own column-label rule
    used_slots = sorted({schedule.node_to_slot[n] for n in schedule.nodes})
    for node, label in zip(schedule.nodes, labels):
        x, y = positions[node]
        colour, marker = slot_style(schedule.node_to_slot[node])
        # CONCEPT: Spatial-reuse factor - nodes sharing a colour (and marker) transmit in the same slot; several same-coloured nodes spread across the map is spatial reuse made visible
        ax.scatter(
            [x], [y], s=260, c=[colour], marker=marker, edgecolors=SURFACE, linewidths=1.3, zorder=3,
        )
        ax.annotate(
            f"{label}/S{schedule.node_to_slot[node]}", (x, y), xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color=INK_SECONDARY, zorder=4,
        )

    ax.set_aspect("equal", adjustable="datalim")  # metres on both axes must look like metres
    ax.set_xlabel("x (m)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("y (m)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.margins(0.12)

    ax.set_title(_topology_title(schedule, lower_bound, status), color=INK_PRIMARY, fontsize=12, pad=12)
    handles = [
        Line2D([], [], color=slot_style(s)[0], marker=slot_style(s)[1], linestyle="none", markersize=8, label=f"Slot {s}")
        for s in used_slots
    ]
    if highlight is not None:
        handles += _highlight_legend_handles()
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=min(len(handles), 6),
        fontsize=8, frameon=False, labelcolor=INK_SECONDARY,
    )
    _savefig(fig, path)


def plot_timeline(schedule, path, frames=2):
    """Save a slots x radios grid over `frames` consecutive frames, filled where a radio transmits.

    Time runs left to right (each frame is `schedule.frame_length` columns wide); radios are
    the rows, in natural order. Several filled cells stacked in ONE column mean several radios
    transmit at the same instant: that is spatial reuse, drawn directly instead of just counted.
    """
    if not isinstance(frames, int) or frames < 1:
        raise InputError(f"frames must be a positive integer, got {frames!r}")
    k, n = schedule.frame_length, len(schedule.nodes)
    labels, _ = column_labels(schedule.nodes)

    # CONCEPT: TDMA (time slots, frames, frame length) - the frame repeats identically, so frame 1's grid is just frame 0's copied across
    grid = [[1 if s in schedule.tx_slots(node) else 0 for _ in range(frames) for s in range(k)] for node in schedule.nodes]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * k * frames + 2), max(3.5, 0.28 * n + 1.5)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = mcolors.ListedColormap([SURFACE, CATEGORICAL[0]])
    ax.pcolormesh(
        grid, cmap=cmap, vmin=0, vmax=1, edgecolors=GRID_LINE, linewidth=0.6,
    )
    for f in range(1, frames):  # a clearer divider between frame f-1 and frame f
        ax.axvline(f * k, color=INK_MUTED, linewidth=1.2)
    for f in range(frames):
        ax.text(f * k + k / 2, n + 0.35, f"Frame {f}", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks([f * k + s + 0.5 for f in range(frames) for s in range(k)])
    ax.set_xticklabels([str(s) for _ in range(frames) for s in range(k)], fontsize=7, color=INK_MUTED)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(labels, fontsize=7.5, color=INK_SECONDARY)
    ax.invert_yaxis()  # Node_01 (the first row) at the top, reading order
    ax.set_xlabel("Slot (within each frame)", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, k * frames)
    ax.set_ylim(n + 0.7, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"Frame timeline: {k}-slot frame x {frames} frames", color=INK_PRIMARY, fontsize=12, pad=22)
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=9, markerfacecolor=CATEGORICAL[0], markeredgecolor="none", label="transmit (TX)"),
        Line2D([], [], marker="s", linestyle="none", markersize=9, markerfacecolor=SURFACE, markeredgecolor=GRID_LINE, label="idle"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    fig.text(0.5, 0.01, "1 slot = 1 ms (Part 2)", ha="center", fontsize=8.5, color=INK_MUTED, style="italic")
    _savefig(fig, path, bottom=0.20)


def _topology_title(schedule, lower_bound, status):
    title = f"Topology by slot - K = {schedule.frame_length} timeslots"
    if lower_bound is not None:
        title += f", LB = {lower_bound}"
    if status is not None:
        title += f" ({_STATUS_TEXT.get(status, status)})"
    return title


def _draw_highlight(ax, G, positions, highlight):
    """The range circle plus a coloured ring on the 1-hop and 2-hop-only neighbours of `highlight`."""
    radio_range = G.graph.get("radio_range", 500.0)
    hop = nx.single_source_shortest_path_length(G, highlight, cutoff=2)
    one_hop = sorted((n for n, d in hop.items() if d == 1), key=natural_key)
    two_hop = sorted((n for n, d in hop.items() if d == 2), key=natural_key)

    cx, cy = positions[highlight]
    ax.add_patch(Circle((cx, cy), radio_range, fill=False, linestyle="--", edgecolor=INK_MUTED, linewidth=1.3, zorder=2))
    for name, colour, width in ((one_hop, ACCENT_1HOP, 2.6), (two_hop, ACCENT_2HOP, 2.6)):
        for node in name:
            x, y = positions[node]
            ax.scatter([x], [y], s=420, facecolors="none", edgecolors=colour, linewidths=width, zorder=2)
    hx, hy = positions[highlight]
    ax.scatter([hx], [hy], s=520, facecolors="none", edgecolors=INK_PRIMARY, linewidths=2.8, zorder=2)


def _highlight_legend_handles():
    return [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=INK_PRIMARY, markeredgewidth=2, markersize=10, label="highlighted"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=ACCENT_1HOP, markeredgewidth=2, markersize=10, label="1-hop neighbour"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=ACCENT_2HOP, markeredgewidth=2, markersize=10, label="2-hop neighbour"),
    ]


def _savefig(fig, path, bottom=0.16):
    """Write the figure to `path` as a PNG, creating parent folders, and always close the figure."""
    fig.subplots_adjust(bottom=bottom)
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    finally:
        plt.close(fig)
