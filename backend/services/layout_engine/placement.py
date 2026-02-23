"""
Room placement logic.

Decides *where* each room goes on the grid, taking into account:
  - room size priority (larger rooms placed first)
  - distance-from-wall preferences
  - neighbour affinity (rooms that want to be adjacent)
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from .subdivision import SubdivisionGrid, _GridRoom


def compute_room_specs(
    room_requirements: List[dict],
    grid_area: int,
) -> List[_GridRoom]:
    """
    Convert high-level room requirements into internal ``_GridRoom`` objects.

    Parameters
    ----------
    room_requirements : list[dict]
        Each dict has ``room_type`` (str) and ``size`` (int, relative weight).
    grid_area : int
        Total interior cells in the grid.

    Returns
    -------
    list[_GridRoom]
        Ready for placement on a ``SubdivisionGrid``.
    """
    total_wanted = sum(r["size"] ** 2 for r in room_requirements)
    grooms = []
    for idx, req in enumerate(room_requirements):
        grooms.append(
            _GridRoom(
                room_id=idx,
                room_type=req["room_type"],
                size=req["size"],
                total_area=grid_area,
                total_wanted=total_wanted,
            )
        )
    return grooms


def place_all_rooms(
    grid: SubdivisionGrid,
    grooms: List[_GridRoom],
    seed: Optional[int] = None,
) -> List[_GridRoom]:
    """
    Place all rooms on *grid*: seed → rectangular growth → L-growth → fill.

    Parameters
    ----------
    grid : SubdivisionGrid
        The grid to populate.
    grooms : list[_GridRoom]
        Rooms to place (sorted largest first internally).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[_GridRoom]
        The same list, now with populated ``.cells``.
    """
    if seed is not None:
        random.seed(seed)

    # Sort by descending size so large rooms get priority
    grooms_sorted = sorted(grooms, key=lambda g: g.size, reverse=True)

    # Phase 1: seed placement
    for gr in grooms_sorted:
        grid.place_room(gr)

    # Phase 2: rectangular growth
    growable = [g for g in grooms_sorted]
    while growable:
        # Weighted pick: bigger rooms get more attempts
        pool = []
        for g in growable:
            pool.extend([g] * g.size)
        picked = random.choice(pool)
        grid.grow_rect(picked)
        if not picked.can_grow:
            growable.remove(picked)

    # Phase 3: L-shaped growth
    for g in grooms_sorted:
        g.can_grow = True
    growable = list(grooms_sorted)
    while growable:
        pool = []
        for g in growable:
            pool.extend([g] * g.size)
        picked = random.choice(pool)
        grid.grow_l_shape(picked)
        if not picked.can_grow:
            growable.remove(picked)

    # Phase 4: fill any unassigned cells
    grid.fill_empty(grooms_sorted)

    return grooms_sorted
