"""
Floor Plan Generation Engine.

Implements a BSP-based room placement algorithm with Shapely geometry.
Handles irregular boundaries, room sizing, wall generation, door/window placement.
"""

import math
import random
from typing import Optional
from shapely.geometry import Polygon, box, LineString, MultiPolygon
from shapely.affinity import scale as shapely_scale, translate
from shapely.ops import unary_union
import json


# Default room sizes (sq ft) and aspect ratios
ROOM_DEFAULTS = {
    "master_bedroom": {"area": 200, "aspect": 1.5, "label": "Master Bedroom"},
    "bedroom": {"area": 150, "aspect": 1.5, "label": "Bedroom"},
    "bathroom": {"area": 50, "aspect": 1.0, "label": "Bathroom"},
    "kitchen": {"area": 150, "aspect": 1.2, "label": "Kitchen"},
    "living": {"area": 250, "aspect": 1.3, "label": "Living Room"},
    "dining": {"area": 120, "aspect": 1.2, "label": "Dining Room"},
    "study": {"area": 100, "aspect": 1.2, "label": "Study"},
    "garage": {"area": 200, "aspect": 1.5, "label": "Garage"},
    "hallway": {"area": 60, "aspect": 3.0, "label": "Hallway"},
    "balcony": {"area": 40, "aspect": 2.0, "label": "Balcony"},
    "pooja": {"area": 30, "aspect": 1.0, "label": "Pooja Room"},
    "store": {"area": 40, "aspect": 1.0, "label": "Store Room"},
    "other": {"area": 80, "aspect": 1.2, "label": "Room"},
}

WALL_THICKNESS = 0.5  # feet
MIN_ROOM_DIMENSION = 8.0  # Minimum room width/height in feet
DOOR_WIDTH = 3.0  # Standard door width
WINDOW_WIDTH = 4.0  # Standard window width
CORRIDOR_WIDTH = 3.5  # Hallway/circulation width


def _normalize_boundary(polygon_coords: list, target_area: Optional[float] = None) -> Polygon:
    """
    Normalise polygon: ensure closed, counter-clockwise.
    If target_area given and different from polygon area, scale accordingly.
    """
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    poly = Polygon(polygon_coords)

    # Ensure counter-clockwise
    if not poly.exterior.is_ccw:
        poly = Polygon(list(reversed(list(poly.exterior.coords))))

    # Make valid
    if not poly.is_valid:
        poly = poly.buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)

    # Scale to target area if needed
    if target_area and abs(poly.area - target_area) > 1.0:
        scale_factor = math.sqrt(target_area / poly.area)
        centroid = poly.centroid
        poly = shapely_scale(poly, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    return poly


def _compute_room_targets(rooms: list, total_area: float) -> list:
    """
    Assign target areas to rooms. If user provided desired_area, use that.
    Otherwise use defaults. Scale proportionally to fit total_area.
    """
    result = []
    for room in rooms:
        rtype = room.get("room_type", "other")
        qty = room.get("quantity", 1)
        desired = room.get("desired_area")
        defaults = ROOM_DEFAULTS.get(rtype, ROOM_DEFAULTS["other"])

        for i in range(qty):
            label = defaults["label"]
            if qty > 1:
                label = f"{label} {i + 1}"
            result.append({
                "room_type": rtype,
                "label": label,
                "target_area": desired if desired else defaults["area"],
                "aspect": defaults["aspect"],
            })

    # Scale areas proportionally within the boundary (minus wall space)
    usable_area = total_area * 0.85  # ~15% for walls/corridors
    total_target = sum(r["target_area"] for r in result)

    if total_target > 0:
        scale_factor = usable_area / total_target
        for r in result:
            r["target_area"] = round(r["target_area"] * scale_factor, 1)

    # Sort largest first for BSP placement
    result.sort(key=lambda r: r["target_area"], reverse=True)
    return result


def _split_rect(rect_poly: Polygon, area_ratio: float, split_vertical: bool) -> tuple:
    """
    Split a rectangle (polygon) into two rectangles at the given area ratio.
    Returns (rect_a, rect_b).
    """
    minx, miny, maxx, maxy = rect_poly.bounds
    w = maxx - minx
    h = maxy - miny

    if split_vertical:
        split_x = minx + w * area_ratio
        a = box(minx, miny, split_x, maxy)
        b = box(split_x, miny, maxx, maxy)
    else:
        split_y = miny + h * area_ratio
        a = box(minx, miny, maxx, split_y)
        b = box(minx, split_y, maxx, maxy)

    return a, b


def _bsp_partition(bounding_rect: Polygon, room_targets: list, boundary: Polygon) -> list:
    """
    Binary space partitioning: recursively split bounding rectangle to allocate rooms.
    Clip each result to the boundary polygon.
    Enhanced with minimum room dimensions and cleaner spacing.
    """
    if len(room_targets) == 0:
        return []

    if len(room_targets) == 1:
        clipped = bounding_rect.intersection(boundary)
        if clipped.is_empty:
            clipped = bounding_rect
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        
        # Ensure room meets minimum dimensions
        minx, miny, maxx, maxy = clipped.bounds
        if (maxx - minx) < MIN_ROOM_DIMENSION or (maxy - miny) < MIN_ROOM_DIMENSION:
            # Room too small, expand bounds if possible
            clipped = clipped.buffer(0.5)
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)
        
        return [{"room": room_targets[0], "polygon": clipped}]

    # Find split point with better distribution
    total_area = sum(r["target_area"] for r in room_targets)
    mid_point = len(room_targets) // 2
    area_a = sum(r["target_area"] for r in room_targets[:mid_point])
    ratio = area_a / total_area if total_area > 0 else 0.5
    
    # Clamp ratio to avoid very thin rooms
    ratio = max(0.3, min(0.7, ratio))

    # Decide split direction based on bounds and room aspect ratios
    minx, miny, maxx, maxy = bounding_rect.bounds
    w = maxx - minx
    h = maxy - miny
    
    # Prefer splitting along the longer dimension
    split_vertical = w >= h
    
    # Adjust for room aspect ratios
    avg_aspect_a = sum(r.get("aspect", 1.2) for r in room_targets[:mid_point]) / max(1, mid_point)
    if avg_aspect_a > 2.0:  # Narrow rooms prefer vertical split
        split_vertical = True

    rect_a, rect_b = _split_rect(bounding_rect, ratio, split_vertical)

    result_a = _bsp_partition(rect_a, room_targets[:mid_point], boundary)
    result_b = _bsp_partition(rect_b, room_targets[mid_point:], boundary)

    return result_a + result_b


def _generate_walls(room_results: list, boundary: Polygon) -> list:
    """
    Generate clean wall geometries with proper thickness and alignment.
    Creates double-line walls for professional CAD output.
    """
    walls = []

    for result in room_results:
        poly = result["polygon"]
        if poly.is_empty or not poly.is_valid:
            continue
        
        # Get room boundary coordinates
        coords = list(poly.exterior.coords)
        
        # Create wall segments with proper thickness
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            
            # Calculate perpendicular offset for wall thickness
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 0.1:
                continue
            
            # Perpendicular unit vector
            px = -dy / length * WALL_THICKNESS / 2
            py = dx / length * WALL_THICKNESS / 2
            
            # Create wall geometry as a polygon (rectangle)
            wall_poly = Polygon([
                (x1 + px, y1 + py),
                (x2 + px, y2 + py),
                (x2 - px, y2 - py),
                (x1 - px, y1 - py),
            ])
            
            walls.append({
                "type": "interior_wall",
                "geometry": _poly_to_coords(wall_poly),
                "start": [round(x1, 2), round(y1, 2)],
                "end": [round(x2, 2), round(y2, 2)],
                "thickness": WALL_THICKNESS,
            })

    # Add outer boundary wall with increased thickness
    outer_wall = boundary.boundary.buffer(WALL_THICKNESS)
    walls.append({
        "type": "exterior_wall",
        "geometry": _poly_to_coords(outer_wall),
        "thickness": WALL_THICKNESS * 2,
    })

    return walls


def _generate_doors(room_results: list) -> list:
    """
    Generate door positions on shared edges between rooms with proper swing geometry.
    Creates professional door representations with hinge point and swing arc.
    """
    doors = []

    for i in range(len(room_results)):
        for j in range(i + 1, len(room_results)):
            poly_a = room_results[i]["polygon"]
            poly_b = room_results[j]["polygon"]
            if poly_a.is_empty or poly_b.is_empty:
                continue

            shared_edge = poly_a.boundary.intersection(poly_b.boundary)

            if not shared_edge.is_empty and shared_edge.length > DOOR_WIDTH:
                edge = shared_edge
                if shared_edge.geom_type == "MultiLineString":
                    edge = max(shared_edge.geoms, key=lambda g: g.length)

                if edge.geom_type != "LineString":
                    continue

                # Place door at center of shared edge
                mid = edge.interpolate(0.5, normalized=True)
                
                # Compute direction along the edge
                coords = list(edge.coords)
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                edge_len = math.sqrt(dx * dx + dy * dy)
                if edge_len < 0.1:
                    continue
                ux, uy = dx / edge_len, dy / edge_len
                
                # Perpendicular (swing direction towards room_b)
                px, py = -uy, ux

                half = DOOR_WIDTH / 2
                hinge = [round(mid.x - ux * half, 2), round(mid.y - uy * half, 2)]
                door_end = [round(mid.x + ux * half, 2), round(mid.y + uy * half, 2)]

                # Determine if edge is more vertical or horizontal for proper rendering
                is_vertical = abs(dy) > abs(dx)

                doors.append({
                    "type": "door",
                    "position": [round(mid.x, 2), round(mid.y, 2)],
                    "hinge": hinge,
                    "door_end": door_end,
                    "width": DOOR_WIDTH,
                    "swing_dir": [round(px, 3), round(py, 3)],
                    "is_vertical": is_vertical,
                    "between": [
                        room_results[i]["room"]["label"],
                        room_results[j]["room"]["label"],
                    ],
                })
    return doors


def _generate_windows(room_results: list, boundary: Polygon) -> list:
    """
    Place windows on exterior walls with proper start/end geometry.
    Creates professional window symbols with frame representation.
    """
    windows = []
    window_room_types = {"living", "master_bedroom", "bedroom", "study", "dining", "kitchen"}

    for result in room_results:
        rtype = result["room"]["room_type"]
        if rtype not in window_room_types:
            continue

        poly = result["polygon"]
        if poly.is_empty:
            continue

        room_boundary = poly.boundary
        outer = boundary.boundary
        touching = room_boundary.intersection(outer)

        if not touching.is_empty and touching.length > (WINDOW_WIDTH + 1.0):
            edge = touching
            if touching.geom_type == "MultiLineString":
                edge = max(touching.geoms, key=lambda g: g.length)
                if edge.geom_type != "LineString":
                    continue
            elif touching.geom_type != "LineString":
                continue

            # Determine window size based on room type
            win_width = WINDOW_WIDTH + 1.0 if rtype == "living" else WINDOW_WIDTH
            
            # Place window at center of exterior wall
            mid = edge.interpolate(0.5, normalized=True)

            coords = list(edge.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            edge_len = math.sqrt(dx * dx + dy * dy)
            if edge_len < 0.1:
                continue
            ux, uy = dx / edge_len, dy / edge_len

            half = win_width / 2
            win_start = [round(mid.x - ux * half, 2), round(mid.y - uy * half, 2)]
            win_end = [round(mid.x + ux * half, 2), round(mid.y + uy * half, 2)]
            is_vertical = abs(dy) > abs(dx)

            windows.append({
                "type": "window",
                "position": [round(mid.x, 2), round(mid.y, 2)],
                "start": win_start,
                "end": win_end,
                "width": win_width,
                "is_vertical": is_vertical,
                "room": result["room"]["label"],
            })

    return windows


def _generate_furniture(room_results: list, boundary: Polygon) -> list:
    """
    Generate furniture symbols for each room type.
    Matches professional floor plan standards with beds, counters, fixtures.
    """
    furniture = []
    
    for result in room_results:
        rtype = result["room"]["room_type"]
        poly = result["polygon"]
        if poly.is_empty:
            continue
        
        centroid = poly.centroid
        minx, miny, maxx, maxy = poly.bounds
        room_width = maxx - minx
        room_height = maxy - miny
        
        # BEDROOM FURNITURE - Double bed
        if rtype in ["master_bedroom", "bedroom"]:
            bed_width = 6.0 if rtype == "master_bedroom" else 5.0
            bed_height = 7.0 if rtype == "master_bedroom" else 6.5
            
            # Center bed in room
            bed_x = centroid.x - bed_width / 2
            bed_y = centroid.y - bed_height / 2
            
            # Bed frame
            furniture.append({
                "type": "bed",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x, bed_y],
                    [bed_x + bed_width, bed_y],
                    [bed_x + bed_width, bed_y + bed_height],
                    [bed_x, bed_y + bed_height],
                    [bed_x, bed_y],
                ],
            })
            
            # Pillows (two rectangles at head)
            pillow_width = bed_width / 2 - 0.3
            pillow_height = 1.5
            furniture.append({
                "type": "pillow",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x + 0.3, bed_y + 0.3],
                    [bed_x + 0.3 + pillow_width, bed_y + 0.3],
                    [bed_x + 0.3 + pillow_width, bed_y + 0.3 + pillow_height],
                    [bed_x + 0.3, bed_y + 0.3 + pillow_height],
                    [bed_x + 0.3, bed_y + 0.3],
                ],
            })
            furniture.append({
                "type": "pillow",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3],
                    [bed_x + bed_width - 0.3, bed_y + 0.3],
                    [bed_x + bed_width - 0.3, bed_y + 0.3 + pillow_height],
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3 + pillow_height],
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3],
                ],
            })
        
        # KITCHEN FURNITURE - Counter and stove
        elif rtype == "kitchen":
            counter_depth = 2.0
            stove_size = 2.5
            
            # L-shaped counter along two walls
            counter_x = minx + 1
            counter_y = miny + 1
            
            # Counter along bottom
            furniture.append({
                "type": "counter",
                "room": result["room"]["label"],
                "geometry": [
                    [counter_x, counter_y],
                    [counter_x + room_width - 2, counter_y],
                    [counter_x + room_width - 2, counter_y + counter_depth],
                    [counter_x, counter_y + counter_depth],
                    [counter_x, counter_y],
                ],
            })
            
            # Stove symbols (4 burners)
            stove_x = counter_x + 3
            stove_y = counter_y + 0.25
            burner_spacing = 0.8
            for i in range(2):
                for j in range(2):
                    bx = stove_x + i * burner_spacing
                    by = stove_y + j * burner_spacing
                    furniture.append({
                        "type": "burner",
                        "room": result["room"]["label"],
                        "center": [bx, by],
                        "radius": 0.25,
                    })
        
        # BATHROOM FURNITURE - Toilet and sink
        elif rtype == "bathroom":
            # Toilet
            toilet_x = minx + 1.5
            toilet_y = miny + 1
            furniture.append({
                "type": "toilet",
                "room": result["room"]["label"],
                "center": [toilet_x, toilet_y + 1],
                "radius": 0.8,
            })
            furniture.append({
                "type": "toilet_tank",
                "room": result["room"]["label"],
                "geometry": [
                    [toilet_x - 0.6, toilet_y],
                    [toilet_x + 0.6, toilet_y],
                    [toilet_x + 0.6, toilet_y + 0.6],
                    [toilet_x - 0.6, toilet_y + 0.6],
                    [toilet_x - 0.6, toilet_y],
                ],
            })
            
            # Sink
            sink_x = maxx - 2
            sink_y = miny + 2
            furniture.append({
                "type": "sink",
                "room": result["room"]["label"],
                "center": [sink_x, sink_y],
                "radius": 0.6,
            })
        
        # STUDY FURNITURE - Desk
        elif rtype == "study":
            desk_width = 5.0
            desk_depth = 2.5
            desk_x = centroid.x - desk_width / 2
            desk_y = miny + 1.5
            
            furniture.append({
                "type": "desk",
                "room": result["room"]["label"],
                "geometry": [
                    [desk_x, desk_y],
                    [desk_x + desk_width, desk_y],
                    [desk_x + desk_width, desk_y + desk_depth],
                    [desk_x, desk_y + desk_depth],
                    [desk_x, desk_y],
                ],
            })
    
    return furniture


def _generate_wall_dimensions(room_results: list, boundary: Polygon) -> list:
    """
    Generate wall dimension annotations for professional floor plans.
    Shows measurements on walls like in the reference image.
    """
    dimensions = []
    
    for result in room_results:
        poly = result["polygon"]
        if poly.is_empty:
            continue
        
        coords = list(poly.exterior.coords)
        
        # Add dimensions for each wall segment
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 2.0:  # Skip very short walls
                continue
            
            # Midpoint for dimension text
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset dimension text perpendicular to wall
            if abs(dx) > abs(dy):  # Horizontal wall
                offset_x = 0
                offset_y = 0.8
            else:  # Vertical wall
                offset_x = 0.8
                offset_y = 0
            
            dimensions.append({
                "type": "wall_dimension",
                "length": round(length, 1),
                "position": [round(mid_x + offset_x, 2), round(mid_y + offset_y, 2)],
                "start": [round(x1, 2), round(y1, 2)],
                "end": [round(x2, 2), round(y2, 2)],
                "is_horizontal": abs(dx) > abs(dy),
            })
    
    return dimensions


def _poly_to_coords(poly) -> list:
    """Convert Shapely polygon to coordinate list."""
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type == "Polygon":
        return [[round(x, 2), round(y, 2)] for x, y in poly.exterior.coords]
    return []


def generate_floor_plan(
    boundary_polygon: list,
    rooms: list,
    total_area: Optional[float] = None,
) -> dict:
    """
    Main entry point: generate a complete professional floor plan.
    Includes furniture, dimensions, and architectural elements.

    Args:
        boundary_polygon: List of [x,y] coordinates forming the boundary.
        rooms: List of dicts with room_type, quantity, desired_area.
        total_area: Total area in sq ft (for scaling).

    Returns:
        Dict with 'rooms', 'walls', 'doors', 'windows', 'furniture', 'dimensions', 'boundary' data.
    """
    # Normalize boundary
    boundary = _normalize_boundary(boundary_polygon, total_area)
    actual_area = boundary.area

    if total_area is None:
        total_area = actual_area

    # Compute room targets
    room_targets = _compute_room_targets(rooms, total_area)

    # Get bounding rectangle
    minx, miny, maxx, maxy = boundary.bounds
    bounding_rect = box(minx, miny, maxx, maxy)

    # BSP partition
    room_results = _bsp_partition(bounding_rect, room_targets, boundary)

    # Generate architectural elements
    walls = _generate_walls(room_results, boundary)
    doors = _generate_doors(room_results)
    windows = _generate_windows(room_results, boundary)
    furniture = _generate_furniture(room_results, boundary)
    dimensions = _generate_wall_dimensions(room_results, boundary)

    # Build result
    plan_rooms = []
    for result in room_results:
        coords = _poly_to_coords(result["polygon"])
        poly = result["polygon"]
        plan_rooms.append({
            "label": result["room"]["label"],
            "room_type": result["room"]["room_type"],
            "target_area": result["room"]["target_area"],
            "actual_area": round(poly.area, 2) if not poly.is_empty else 0,
            "polygon": coords,
            "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)] if not poly.is_empty else [0, 0],
        })

    return {
        "boundary": _poly_to_coords(boundary),
        "total_area": round(actual_area, 2),
        "rooms": plan_rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "furniture": furniture,
        "dimensions": dimensions,
    }
