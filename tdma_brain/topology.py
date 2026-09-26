"""Topology model for the TDMA planner (created in Division 1).

Radio graph G: a unit disk graph, radios are nodes and two radios are linked when they are
within radio range of each other. Conflict graph H = G^2: two radios are joined when they
are at most 2 hops apart in G, i.e. when they must NOT share a time slot. Also the
summary that `--show-topology` prints.
"""
import math

import networkx as nx

from .io_utils import InputError, natural_key

DEFAULT_RADIO_RANGE_M = 500.0
RANGE_TOLERANCE_M = 1e-9  # a pair at exactly the range must stay a link despite float noise


def validate_radio_range(radio_range):
    """Raise InputError unless the radio range is a finite number of metres greater than 0."""
    if not (math.isfinite(radio_range) and radio_range > 0):
        raise InputError(f"radio range must be a finite number of metres greater than 0, got {radio_range}")


def build_radio_graph(coords, radio_range=DEFAULT_RADIO_RANGE_M):
    """Build the radio graph G from {name: (x, y)} coordinates in metres.

    Nodes are added in natural order with attribute `pos=(x, y)`. Two radios are linked
    when their distance is <= radio_range (inclusive, plus a 1e-9 tolerance); the edge
    carries the attribute `distance`. The range is kept in `G.graph["radio_range"]`.
    """
    validate_radio_range(radio_range)

    names = sorted(coords, key=natural_key)
    # CONCEPT: Graph basics & NetworkX - a radio becomes a node (pos = its coordinates) and every link below becomes an edge
    G = nx.Graph(radio_range=float(radio_range))
    for name in names:
        x, y = coords[name]
        G.add_node(name, pos=(float(x), float(y)))

    for i, u in enumerate(names):
        ux, uy = G.nodes[u]["pos"]
        for v in names[i + 1:]:
            vx, vy = G.nodes[v]["pos"]
            # CONCEPT: Euclidean distance & floating-point tolerance - straight-line distance in metres; the 1e-9 keeps a pair at exactly 500 m from being lost to float noise
            distance = math.hypot(ux - vx, uy - vy)
            # CONCEPT: Radio range & the Unit Disk Graph model - two radios hear each other exactly when their distance is <= the radio range (500 m)
            if distance <= radio_range + RANGE_TOLERANCE_M:
                G.add_edge(u, v, distance=distance)
    return G


def build_conflict_graph(G):
    """Build the conflict graph H = G^2 and annotate every edge.

    Edge attribute `kind` is "1-hop" when G has the edge (the radios hear each other) or
    "2-hop" when they only share a neighbour; 2-hop edges also get `via`, the natural-sorted
    list of common neighbours. A slot assignment is valid exactly when it is a proper
    colouring of H. Slots are per radio (a radio broadcasts to all neighbours), so unlike
    the exposed-terminal case in CSMA, two neighbours are never allowed to share a slot.
    """
    # CONCEPT: k-hop neighbourhood & the square of a graph (G^2) - nx.power(G, 2) joins every pair of radios that are at most 2 hops apart in G
    # CONCEPT: Conflict / interference graph (protocol interference model) - an edge of H means "these two radios must not transmit in the same slot"
    H = nx.power(G, 2)
    for u, v, attributes in H.edges(data=True):
        if G.has_edge(u, v):
            # CONCEPT: Half-duplex radios / distance-1 conflict - u and v hear each other, and a radio cannot send while it listens, so they need different slots
            attributes["kind"] = "1-hop"
        else:
            # CONCEPT: Multi-hop communication & hop count - u and v are exactly 2 hops apart: no direct link, but some radio hears both of them
            attributes["kind"] = "2-hop"
            attributes["via"] = _common_neighbours(G, u, v)
    return H


def hidden_terminal_pairs(G, H):
    """List the hidden-terminal pairs as (u, v, via), natural-sorted.

    These are the conflicts of H that are not direct links of G: u and v cannot hear each
    other, yet both reach every radio in `via`. u comes before v in natural order.
    """
    pairs = []
    for u, v in H.edges:
        if G.has_edge(u, v):
            continue  # a direct link is a 1-hop conflict, not a hidden terminal
        first, second = sorted((u, v), key=natural_key)
        pairs.append((first, second, _common_neighbours(G, first, second)))
    pairs.sort(key=lambda pair: (natural_key(pair[0]), natural_key(pair[1])))
    return pairs


def topology_summary(G, H):
    """Numbers about the topology, as a dict (formatted by format_topology_summary)."""
    # CONCEPT: Graph basics & NetworkX - degree = how many radios a node hears; a connected component = a group of radios that can reach each other through hops
    degrees = dict(G.degree())
    max_degree = max(degrees.values(), default=0)
    hidden = hidden_terminal_pairs(G, H)
    return {
        "nodes": G.number_of_nodes(),
        "radio_range": G.graph.get("radio_range"),
        "links": G.number_of_edges(),
        "conflict_pairs": H.number_of_edges(),
        "hidden_terminal_pairs": len(hidden),
        "max_degree": max_degree,
        "max_degree_nodes": sorted((n for n, d in degrees.items() if d == max_degree), key=natural_key),
        "components": nx.number_connected_components(G),
        "isolated_nodes": sorted(nx.isolates(G), key=natural_key),
        "example_hidden_terminal": hidden[0] if hidden else None,
    }


def format_topology_summary(summary):
    """Render a topology_summary() dict as the ASCII block printed by --show-topology."""
    nodes = summary["nodes"]
    example = summary["example_hidden_terminal"]
    example_text = "none" if example is None else f"{example[0]} <-> {example[1]} via {', '.join(example[2])}"
    isolated = summary["isolated_nodes"]
    rows = [
        ("Nodes", str(nodes)),
        ("Radio range", f"{summary['radio_range']} m (link if distance <= range)"),
        ("Radio links (1-hop)", str(summary["links"])),
        ("Conflict pairs (<= 2 hops)", str(summary["conflict_pairs"])),
        ("Hidden-terminal pairs", f"{summary['hidden_terminal_pairs']} (2-hop only)"),
        ("Max degree", f"{summary['max_degree']} ({_names(summary['max_degree_nodes'], nodes)})"),
        ("Connected components", str(summary["components"])),
        ("Isolated nodes", _names(isolated, nodes) if isolated else "none"),
        ("Example hidden-terminal", example_text),
    ]
    lines = ["TOPOLOGY SUMMARY"]
    lines += [f"  {label:<26} : {value}" for label, value in rows]
    return "\n".join(lines)


def _common_neighbours(G, u, v):
    """The radios that hear both u and v, natural-sorted (the `via` of a 2-hop pair)."""
    # CONCEPT: Hidden terminal problem - u and v cannot hear each other, but both reach these radios, so if u and v transmit together their packets collide at every one of them
    return sorted(nx.common_neighbors(G, u, v), key=natural_key)


def _names(names, total, limit=8):
    """Comma-separated node names, shortened so one line stays readable on big networks."""
    if total > 1 and len(names) == total:
        return f"all {total} nodes"
    if len(names) > limit:
        return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"
    return ", ".join(names)
