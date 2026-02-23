"""
Grid-based space subdivision.

Implements a discrete grid that covers a bounding polygon. Each cell in
the grid can be assigned to a room.  Rooms are placed via weighted
selection, then grown rectangularly and with L-shaped extensions until
the grid is fully allocated.

All final room geometries are converted to Shapely Polygons.
"""

import math
import random
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box

from .room_model import Room


# ---------------------------------------------------------------------------
# Direction helpers (row, col offsets on the grid)
# ---------------------------------------------------------------------------

def _up(pos):    return (pos[0] - 1, pos[1])
def _down(pos):  return (pos[0] + 1, pos[1])
def _left(pos):  return (pos[0], pos[1] - 1)
def _right(pos): return (pos[0], pos[1] + 1)

def _ul(pos): return (pos[0] - 1, pos[1] - 1)
def _ur(pos): return (pos[0] - 1, pos[1] + 1)
def _dl(pos): return (pos[0] + 1, pos[1] - 1)
def _dr(pos): return (pos[0] + 1, pos[1] + 1)

DIRS4 = [_up, _left, _right, _down]
DIRS8 = [_ul, _up, _ur, _left, _right, _dl, _down, _dr]


# ---------------------------------------------------------------------------
# Tile walker — walks the perimeter of a room on the grid
# ---------------------------------------------------------------------------

class _TileWalker:
    """Walks around the boundary of a room on the grid."""

    def __init__(self, position: Tuple[int, int], facing: str = "right"):
        self.position = position
        # Direction tuple: (front, left, right, back, UL, UR, DL, DR)
        self.direction = (_up, _left, _right, _down,
                          _ul, _ur, _dl, _dr)
        if facing == "left":
            self._turn_left()
        elif facing == "right":
            self._turn_right()
        elif facing == "down":
            self._turn_right()
            self._turn_right()

    def _turn_left(self):
        d = self.direction
        self.direction = (d[1], d[3], d[0], d[2],
                          d[6], d[4], d[7], d[5])

    def _turn_right(self):
        d = self.direction
        self.direction = (d[2], d[0], d[3], d[1],
                          d[5], d[7], d[4], d[6])

    def _move_forward(self):
        self.position = self.direction[0](self.position)

    def look_left(self):
        return self.direction[1](self.position)

    def look_right(self):
        return self.direction[2](self.position)

    def look_forward(self):
        return self.direction[0](self.position)

    def look_bottom_right(self):
        return self.direction[6](self.position)

    def standard_move(self, grid_data, room_id: int) -> str:
        """Move following the room boundary (left-hand rule)."""
        if grid_data[self.look_left()] == room_id:
            self._turn_left()
            self._move_forward()
            return "left"
        elif grid_data[self.look_forward()] != room_id:
            self._turn_right()
            return "right"
        else:
            self._move_forward()
            return "forward"


# ---------------------------------------------------------------------------
# Subdivision Grid
# ---------------------------------------------------------------------------

class SubdivisionGrid:
    """
    Discrete grid for placing and growing rooms.

    The grid is a 2-D numpy array where:
      * ``None``  → wall / outside boundary
      * ``-1``    → empty (available) interior cell
      * ``>= 0``  → assigned to a room ID
    """

    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        """
        Parameters
        ----------
        width : int
            Number of interior columns.
        height : int
            Number of interior rows.
        cell_size : float
            Real-world size of each grid cell (meters).
        """
        self.cell_size = cell_size
        # +2 for border padding (walls)
        self.rows = height + 2
        self.cols = width + 2
        self.grid = np.full((self.rows, self.cols), None, dtype=object)
        self.grid[1:-1, 1:-1] = -1

        self.weight = np.zeros((self.rows, self.cols), dtype=int)
        # Wall cells get high weight so rooms avoid them
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] is None:
                    self.weight[r, c] = 100

        self.interior_area = int(np.sum(self.grid == -1))
        self.rooms: List[_GridRoom] = []

    # ---- room placement --------------------------------------------------

    def _wall_distance(self, pos: Tuple[int, int]) -> int:
        """Manhattan-ish distance from *pos* to nearest wall/boundary."""
        if self.grid[pos] is None:
            return 0
        dist = 0
        to_check = [(_dir(pos), _dir, dist) for _dir in DIRS8]
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        visited[pos] = True
        while dist < max(self.rows, self.cols):
            if not to_check:
                break
            cur_pos, fn, d = to_check.pop(0)
            dist = d + 1
            r, c = cur_pos
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return dist
            if visited[r, c]:
                continue
            visited[r, c] = True
            if self.grid[cur_pos] is None:
                return dist
            to_check.append((fn(cur_pos), fn, dist))
        return dist

    def place_room(self, groom: "_GridRoom"):
        """Pick the best starting cell and place one seed cell for *groom*."""
        # Build a cost grid: prefer cells far from walls, near wanted neighbours
        cost = self.weight.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                wd = self._wall_distance((r, c))
                if wd < groom.wall_distance:
                    cost[r, c] += groom.wall_distance - wd

        # Find minimum-cost cells among interior
        best_val = None
        candidates = []
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.grid[r, c] != -1:
                    continue
                v = cost[r, c]
                if best_val is None or v < best_val:
                    best_val = v
                    candidates = [(r, c)]
                elif v == best_val:
                    candidates.append((r, c))

        if not candidates:
            return
        pos = random.choice(candidates)
        groom.cells.append(pos)
        self.grid[pos] = groom.room_id

        # Increase weight around seed so next room is placed elsewhere
        for dr in range(-groom.wall_distance, groom.wall_distance + 1):
            for dc in range(-groom.wall_distance, groom.wall_distance + 1):
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    w = groom.wall_distance - max(abs(dr), abs(dc))
                    if w > 0 and self.weight[nr, nc] < w:
                        self.weight[nr, nc] = w
        self.weight[pos] = 100
        self.rooms.append(groom)

    # ---- rectangular growth -----------------------------------------------

    def _find_start(self, groom: "_GridRoom") -> Tuple[int, int]:
        """Top-left cell of a room (minimal row then col)."""
        min_r = min(c[0] for c in groom.cells)
        cands = [c for c in groom.cells if c[0] == min_r]
        return min(cands, key=lambda c: c[1])

    def grow_rect(self, groom: "_GridRoom"):
        """Grow the room by one row/column of empty cells along longest edge."""
        if len(groom.cells) < 1:
            groom.can_grow = False
            return

        start = self._find_start(groom)
        walker = _TileWalker(start, "right")

        edge: List[Tuple[int, int]] = []
        edges: List[List[Tuple[int, int]]] = []
        first = True
        max_steps = self.rows * self.cols * 4  # safety limit

        steps = 0
        while first or not (
            walker.position == start and walker.direction[0] == _right
        ):
            first = False
            steps += 1
            if steps > max_steps:
                break

            left_cell = walker.look_left()
            r, c = left_cell
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[left_cell] == -1:
                edge.append(left_cell)
                fwd = walker.look_forward()
                fr, fc = fwd
                if not (0 <= fr < self.rows and 0 <= fc < self.cols) or self.grid[fwd] != groom.room_id:
                    if edge:
                        edges.append(list(edge))
                    edge = []
                walker.standard_move(self.grid, groom.room_id)
            else:
                edge = []
                fwd = walker.look_forward()
                fr, fc = fwd
                inner_steps = 0
                while 0 <= fr < self.rows and 0 <= fc < self.cols and self.grid[fwd] == groom.room_id:
                    walker._move_forward()
                    fwd = walker.look_forward()
                    fr, fc = fwd
                    inner_steps += 1
                    if inner_steps > max_steps:
                        break
                walker._turn_right()

        if not edges:
            groom.can_grow = False
            return

        longest = max(len(e) for e in edges)
        best = [e for e in edges if len(e) == longest]
        picked = random.choice(best)
        groom.cells.extend(picked)
        for cell in picked:
            self.grid[cell] = groom.room_id

        if len(groom.cells) >= groom.target_cells:
            groom.can_grow = False

    # ---- L-shaped growth --------------------------------------------------

    def grow_l_shape(self, groom: "_GridRoom"):
        """Grow room with an L-shaped extension along its boundary."""
        if len(groom.cells) < 1:
            groom.can_grow = False
            return

        start = self._find_start(groom)
        walker = _TileWalker(start, "right")

        edge: List[Tuple[int, int]] = []
        edges: List[Tuple[List[Tuple[int, int]], bool]] = []
        first = True
        is_l = False
        max_steps = self.rows * self.cols * 4

        steps = 0
        while first or not (
            walker.position == start and walker.direction[0] == _right
        ):
            first = False
            steps += 1
            if steps > max_steps:
                break

            left_cell = walker.look_left()
            r, c = left_cell
            in_bounds = 0 <= r < self.rows and 0 <= c < self.cols

            if in_bounds and self.grid[left_cell] == -1:
                edge.append(left_cell)
                fwd = walker.look_forward()
                fr, fc = fwd
                fwd_ok = 0 <= fr < self.rows and 0 <= fc < self.cols
                if not fwd_ok or self.grid[fwd] != groom.room_id or (
                    in_bounds and self.grid[left_cell] == groom.room_id
                ):
                    if edge:
                        edges.append((list(edge), is_l))
                    edge = []
                    is_l = False
                walker.standard_move(self.grid, groom.room_id)
            else:
                if groom.l_used:
                    edge = []
                    is_l = True
                else:
                    if is_l and edge:
                        edges.append((list(edge), True))
                    edge = []
                    is_l = True
                fwd = walker.look_forward()
                fr, fc = fwd
                fwd_ok = 0 <= fr < self.rows and 0 <= fc < self.cols
                if not fwd_ok or self.grid[fwd] != groom.room_id:
                    is_l = False
                walker.standard_move(self.grid, groom.room_id)

        # filter edges too small
        edges = [(e, l) for e, l in edges if len(e) > 1]

        valid = [(e, l) for e, l in edges if not (groom.l_used and l)]
        if not valid:
            groom.can_grow = False
            return

        longest = max(len(e) for e, _ in valid)
        best = [(e, l) for e, l in valid if len(e) == longest]
        picked_edge, picked_l = random.choice(best)

        if picked_l:
            groom.l_used = True
        groom.cells.extend(picked_edge)
        for cell in picked_edge:
            self.grid[cell] = groom.room_id

    # ---- fill unassigned cells --------------------------------------------

    def fill_empty(self, grooms: List["_GridRoom"]):
        """Assign every remaining -1 cell to the room with most adjacent cells."""
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.grid[r, c] != -1:
                    continue
                # BFS to find the connected empty region
                region = []
                queue = [(r, c)]
                visited = set()
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    region.append((cr, cc))
                    for fn in DIRS4:
                        nr, nc = fn((cr, cc))
                        if (nr, nc) not in visited and 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self.grid[nr, nc] == -1:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                # Count adjacency to each room
                neighbours = {}
                for cell in region:
                    for fn in DIRS4:
                        adj = fn(cell)
                        ar, ac = adj
                        if 0 <= ar < self.rows and 0 <= ac < self.cols:
                            v = self.grid[adj]
                            if v is not None and v != -1:
                                neighbours[v] = neighbours.get(v, 0) + 1

                if neighbours:
                    best_id = max(neighbours, key=neighbours.get)
                else:
                    best_id = grooms[0].room_id if grooms else 0

                best_room = next((g for g in grooms if g.room_id == best_id), grooms[0])
                for cell in region:
                    self.grid[cell] = best_id
                    best_room.cells.append(cell)

    # ---- convert grid cells → Shapely polygons ----------------------------

    def cells_to_polygon(self, cells: List[Tuple[int, int]],
                          origin_x: float = 0.0,
                          origin_y: float = 0.0) -> Polygon:
        """
        Merge grid cells into a single Shapely polygon.

        Each cell (r, c) maps to a unit square offset from
        (origin_x, origin_y), scaled by cell_size.
        """
        from shapely.ops import unary_union

        boxes = []
        for r, c in cells:
            # Subtract 1 to remove border offset
            x0 = origin_x + (c - 1) * self.cell_size
            y0 = origin_y + (r - 1) * self.cell_size
            boxes.append(box(x0, y0, x0 + self.cell_size, y0 + self.cell_size))
        if not boxes:
            return Polygon()
        merged = unary_union(boxes)
        return merged if merged.geom_type == "Polygon" else merged.convex_hull


# ---------------------------------------------------------------------------
# Helper: lightweight room info during grid generation
# ---------------------------------------------------------------------------

class _GridRoom:
    """Internal room tracker used during grid subdivision."""

    def __init__(self, room_id: int, room_type: str, size: int,
                 total_area: int, total_wanted: int):
        self.room_id = room_id
        self.room_type = room_type
        self.size = size
        self.area_wanted = size * size

        scale = total_area / max(total_wanted, 1)
        self.target_cells = self.area_wanted * scale
        self.wall_distance = max(int(math.sqrt(self.target_cells) / 2), 1)

        self.cells: List[Tuple[int, int]] = []
        self.can_grow = True
        self.l_used = False
