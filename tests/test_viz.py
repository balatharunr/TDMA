"""Division 4 tests: the plots (tdma_brain.viz)."""
import struct

import pytest

from tdma_brain.exact import BEST_FOUND, PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH
from tdma_brain.io_utils import InputError
from tdma_brain.optimizer import optimize
from tdma_brain.viz import CATEGORICAL, MARKERS, plot_timeline, plot_topology, slot_style
from tests.golden import GOLDEN
from tests.helpers import build_graphs


def png_size(path):
    """(width, height) read straight from the PNG's own IHDR chunk, no PIL needed."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", data[16:24])


# --------------------------------------------------------------------------- slot_style
def test_slot_style_cycles_colour_first_then_marker():
    assert [slot_style(s) for s in range(8)] == [(c, "o") for c in CATEGORICAL]
    assert slot_style(8) == (CATEGORICAL[0], "s") and slot_style(15) == (CATEGORICAL[7], "s")
    assert slot_style(16) == (CATEGORICAL[0], "^")


def test_slot_style_never_repeats_within_64_slots():
    styles = [slot_style(s) for s in range(len(CATEGORICAL) * len(MARKERS))]
    assert len(set(styles)) == len(styles)


# --------------------------------------------------------------------------- plot_topology
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_plot_topology_writes_a_non_empty_png_for_every_example(tmp_path, golden):
    G, H = build_graphs(golden.name)
    result = optimize(G, H)
    path = tmp_path / f"{golden.name}.png"
    plot_topology(G, result.schedule, path, lower_bound=result.lower_bound.value, status=result.status)
    assert path.exists() and path.stat().st_size > 0
    width, height = png_size(path)
    assert width > 100 and height > 100


def test_plot_topology_creates_parent_directories(tmp_path):
    G, H = build_graphs("hidden_terminal3")
    result = optimize(G, H)
    path = tmp_path / "nested" / "deeper" / "topology.png"
    plot_topology(G, result.schedule, path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_topology_accepts_a_string_path(tmp_path):
    G, H = build_graphs("hidden_terminal3")
    result = optimize(G, H)
    path = str(tmp_path / "topology.png")
    plot_topology(G, result.schedule, path)
    import pathlib
    assert pathlib.Path(path).stat().st_size > 0


def test_plot_topology_works_without_lower_bound_or_status(tmp_path):
    G, H = build_graphs("grid16_400m")
    result = optimize(G, H)
    path = tmp_path / "plain.png"
    plot_topology(G, result.schedule, path)  # no lower_bound=, no status=
    assert path.stat().st_size > 0


@pytest.mark.parametrize("status", [PROVEN_OPTIMAL_BY_BOUND, PROVEN_OPTIMAL_BY_SEARCH, BEST_FOUND])
def test_plot_topology_accepts_every_status(tmp_path, status):
    G, H = build_graphs("ring16_r1150m")
    result = optimize(G, H)
    path = tmp_path / f"{status}.png"
    plot_topology(G, result.schedule, path, lower_bound=result.lower_bound.value, status=status)
    assert path.stat().st_size > 0


def test_plot_topology_with_highlight_on_every_node_of_the_official_grid(tmp_path):
    G, H = build_graphs("grid16_300m")
    result = optimize(G, H)
    for node in G.nodes:  # every node, not just Node_06: an isolated node (no 1-hop/2-hop) must not crash either
        path = tmp_path / f"{node}.png"
        plot_topology(G, result.schedule, path, highlight=node)
        assert path.stat().st_size > 0


def test_plot_topology_with_highlight_on_an_isolated_node(tmp_path):
    G, H = build_graphs("isolated16_1000m")
    result = optimize(G, H)
    path = tmp_path / "isolated.png"
    plot_topology(G, result.schedule, path, highlight="Node_01")  # no neighbours at all
    assert path.stat().st_size > 0


def test_plot_topology_rejects_an_unknown_highlight_node(tmp_path):
    G, H = build_graphs("hidden_terminal3")
    result = optimize(G, H)
    with pytest.raises(InputError, match="Node_Z"):
        plot_topology(G, result.schedule, tmp_path / "x.png", highlight="Node_Z")
    assert not (tmp_path / "x.png").exists()  # nothing half-written on a bad argument


def test_plot_topology_is_larger_for_more_nodes(tmp_path):
    # a coarse sanity check that the plot really reflects the data, not a fixed placeholder image
    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    G1, H1 = build_graphs("hidden_terminal3")
    plot_topology(G1, optimize(G1, H1).schedule, small)
    G2, H2 = build_graphs("grid16_300m")
    plot_topology(G2, optimize(G2, H2).schedule, big)
    assert big.stat().st_size != small.stat().st_size


# --------------------------------------------------------------------------- plot_timeline
@pytest.mark.parametrize("golden", GOLDEN, ids=lambda g: g.name)
def test_plot_timeline_writes_a_non_empty_png_for_every_example(tmp_path, golden):
    G, H = build_graphs(golden.name)
    result = optimize(G, H)
    path = tmp_path / f"{golden.name}.png"
    plot_timeline(result.schedule, path)
    assert path.exists() and path.stat().st_size > 0


@pytest.mark.parametrize("frames", [1, 2, 3, 5])
def test_plot_timeline_accepts_different_frame_counts(tmp_path, frames):
    G, H = build_graphs("grid16_400m")
    result = optimize(G, H)
    path = tmp_path / f"frames_{frames}.png"
    plot_timeline(result.schedule, path, frames=frames)
    assert path.stat().st_size > 0


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
def test_plot_timeline_rejects_a_bad_frame_count(tmp_path, bad):
    G, H = build_graphs("hidden_terminal3")
    result = optimize(G, H)
    with pytest.raises(InputError, match="frames"):
        plot_timeline(result.schedule, tmp_path / "x.png", frames=bad)


def test_plot_timeline_creates_parent_directories(tmp_path):
    G, H = build_graphs("hidden_terminal3")
    result = optimize(G, H)
    path = tmp_path / "a" / "b" / "timeline.png"
    plot_timeline(result.schedule, path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_timeline_grows_with_more_frames(tmp_path):
    G, H = build_graphs("grid16_300m")
    result = optimize(G, H)
    one = tmp_path / "one.png"
    many = tmp_path / "many.png"
    plot_timeline(result.schedule, one, frames=1)
    plot_timeline(result.schedule, many, frames=6)
    width_one, _ = png_size(one)
    width_many, _ = png_size(many)
    assert width_many > width_one  # more frames -> a wider grid
