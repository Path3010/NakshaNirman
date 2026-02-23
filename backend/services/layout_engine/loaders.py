"""
Polygon and rule loaders.

Loads the usable boundary polygon from a JSON file and minimum-area
rules from region_rules.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon, shape


# ---------------------------------------------------------------------------
# Step 4 — Load usable polygon
# ---------------------------------------------------------------------------

def load_usable_polygon(path: str) -> Polygon:
    """
    Load a usable polygon from a JSON file.

    Accepted JSON formats::

        # Format A — GeoJSON-style
        {
          "type": "Polygon",
          "coordinates": [[[x1,y1],[x2,y2], ...]]
        }

        # Format B — flat coordinate list
        {
          "polygon": [[x1,y1],[x2,y2], ...]
        }

        # Format C — named vertices
        {
          "vertices": [[x1,y1],[x2,y2], ...]
        }

    Parameters
    ----------
    path : str
        File path to the JSON file.

    Returns
    -------
    Polygon
        Shapely Polygon constructed from the coordinates.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON does not contain a recognizable polygon format.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Polygon file not found: {path}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # GeoJSON-style
    if "type" in data and data["type"] == "Polygon":
        return shape(data)

    # Flat coordinate list under various keys
    coords = None
    for key in ("polygon", "vertices", "coordinates", "points", "boundary"):
        if key in data:
            coords = data[key]
            break

    if coords is None:
        # Maybe the top-level is just a list
        if isinstance(data, list):
            coords = data
        else:
            raise ValueError(
                "Cannot find polygon coordinates in JSON. "
                "Expected keys: polygon, vertices, coordinates, points, boundary"
            )

    # If nested (GeoJSON ring style [[ring]]), unwrap
    if coords and isinstance(coords[0], list) and isinstance(coords[0][0], list):
        coords = coords[0]

    if len(coords) < 3:
        raise ValueError(f"Polygon needs at least 3 vertices, got {len(coords)}")

    polygon = Polygon(coords)
    if not polygon.is_valid:
        from shapely.validation import make_valid
        polygon = make_valid(polygon)

    return polygon


def save_usable_polygon(polygon: Polygon, path: str) -> None:
    """
    Save a Shapely Polygon to a JSON file.

    Output format::

        {
          "type": "Polygon",
          "coordinates": [[[x1,y1],[x2,y2], ...]]
        }
    """
    from shapely.geometry import mapping

    data = mapping(polygon)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Step 7 — Load minimum area rules from region_rules.json
# ---------------------------------------------------------------------------

def load_region_rules(path: str, region: str = "india_mvp") -> dict:
    """
    Load building rules for a specific region.

    Parameters
    ----------
    path : str
        Path to region_rules.json.
    region : str
        Key in the JSON (default ``"india_mvp"``).

    Returns
    -------
    dict
        The full rule set for the region.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    with open(filepath, "r") as f:
        data = json.load(f)

    if region not in data:
        raise KeyError(f"Region '{region}' not found in {path}. "
                        f"Available: {list(data.keys())}")

    return data[region]


def load_min_areas(path: str, region: str = "india_mvp") -> Dict[str, float]:
    """
    Extract per-room-type minimum areas from region_rules.json.

    If the rules file has a ``"min_room_areas"`` section, use it.
    Otherwise return sensible defaults based on common Indian building codes.

    Parameters
    ----------
    path : str
        Path to region_rules.json.
    region : str
        Region key.

    Returns
    -------
    dict
        ``{room_type: min_area_sqm}``
    """
    rules = load_region_rules(path, region)

    # Check for explicit min_room_areas section
    if "min_room_areas" in rules:
        return rules["min_room_areas"]

    # Sensible defaults (sq meters) based on Indian NBC guidelines
    return {
        "living": 9.5,
        "bedroom": 7.5,
        "kitchen": 5.0,
        "bathroom": 1.8,
        "toilet": 1.1,
        "dining": 7.5,
        "pooja": 2.0,
        "store": 2.0,
        "balcony": 2.0,
        "entrance": 1.5,
    }
