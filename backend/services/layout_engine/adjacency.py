"""
Adjacency graph construction for room layouts.

Builds a NetworkX graph where nodes are rooms and edges connect rooms
that share a boundary segment.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import Polygon


def build_adjacency_graph(rooms: List[dict],
                           tolerance: float = 0.05) -> nx.Graph:
    """
    Build an adjacency graph from a list of room dicts.

    Each room dict must have at least ``room_id`` and ``polygon``
    (a Shapely Polygon).

    Two rooms are adjacent if they share a boundary of length > tolerance.

    Parameters
    ----------
    rooms : list[dict]
        Each dict has keys ``room_id`` (int) and ``polygon`` (Polygon).
    tolerance : float
        Minimum shared boundary length (meters) to count as adjacent.

    Returns
    -------
    nx.Graph
        Undirected graph with room_id as nodes and shared-boundary
        length as edge weight ``shared_length``.
    """
    G = nx.Graph()
    for r in rooms:
        G.add_node(r["room_id"], room_type=r.get("room_type", "unknown"))

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            poly_i: Polygon = rooms[i]["polygon"]
            poly_j: Polygon = rooms[j]["polygon"]
            shared = poly_i.intersection(poly_j)
            length = shared.length if not shared.is_empty else 0.0
            if length > tolerance:
                G.add_edge(
                    rooms[i]["room_id"],
                    rooms[j]["room_id"],
                    shared_length=round(length, 4),
                )
    return G


def is_connected(graph: nx.Graph) -> bool:
    """Return True if every room is reachable from every other room."""
    if graph.number_of_nodes() == 0:
        return True
    return nx.is_connected(graph)


def adjacency_pairs(graph: nx.Graph) -> List[Tuple[int, int]]:
    """List all (room_id_a, room_id_b) pairs that share a wall."""
    return list(graph.edges())


def room_neighbours(graph: nx.Graph, room_id: int) -> List[int]:
    """Return IDs of rooms adjacent to *room_id*."""
    if room_id not in graph:
        return []
    return list(graph.neighbors(room_id))


def shared_wall_midpoint(poly_a: Polygon, poly_b: Polygon) -> Optional[Tuple[float, float]]:
    """
    Compute the midpoint of the shared boundary between two polygons.

    Returns None if there is no shared linear boundary.
    """
    shared = poly_a.intersection(poly_b)
    if shared.is_empty or shared.length < 0.01:
        return None
    # Get the centroid of the shared boundary as the door placement point
    mid = shared.centroid
    return (mid.x, mid.y)
