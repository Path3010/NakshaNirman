"""
Phase 3 Verification — Steps 11–15.

Run with:
    cd backend
    .\\venv\\Scripts\\activate
    python test_phase3_steps11_15.py

Tests:
  Step 11 — Door Placement
  Step 12 — Corridor Detection
  Step 13 — Candidate Generation Loop (200 layouts)
  Step 14 — Scoring System (area accuracy, adjacency, corridor penalty)
  Step 15 — Best Candidate Selection
"""

import json, sys, os

# Ensure imports work from backend/
sys.path.insert(0, os.path.dirname(__file__))

from shapely.geometry import Polygon
from services.layout_engine.room_model import Room
from services.layout_engine.doors import Door, place_doors
from services.layout_engine.adjacency import (
    build_adjacency_graph,
    is_connected,
    shared_wall_midpoint,
)
from services.layout_engine.scoring import (
    area_accuracy_score,
    adjacency_score,
    corridor_penalty,
    score_layout,
)
from services.layout_engine import LayoutGenerator

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append(condition)
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


# ====================================================================
print("=" * 60)
print("STEP 11 — Door Placement")
print("=" * 60)

# Create rooms with shared walls
Room.reset_counter()
door_rooms = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (6, 0), (6, 5), (0, 5)]),
         target_area=30),
    Room(room_type="bedroom",
         polygon=Polygon([(6, 0), (12, 0), (12, 5), (6, 5)]),
         target_area=30),
    Room(room_type="kitchen",
         polygon=Polygon([(0, 5), (12, 5), (12, 10), (0, 10)]),
         target_area=60),
]

doors = place_doors(door_rooms)
check("Doors are placed for shared walls",
      len(doors) > 0,
      f"door_count={len(doors)}")
check("Door count matches adjacency pairs (3 shared walls)",
      len(doors) == 3,
      f"doors={len(doors)}")

if doors:
    d = doors[0]
    check("Door has room_a_id", hasattr(d, "room_a_id"))
    check("Door has room_b_id", hasattr(d, "room_b_id"))
    check("Door has position tuple",
          isinstance(d.position, tuple) and len(d.position) == 2,
          f"pos={d.position}")
    check("Door has width", d.width > 0, f"width={d.width}")
    check("Door has geometry (Point)",
          d.geometry.geom_type == "Point")

    # Check door position is at midpoint of shared wall
    mid = shared_wall_midpoint(door_rooms[0].polygon, door_rooms[1].polygon)
    living_bedroom_door = [dd for dd in doors
                           if set([dd.room_a_id, dd.room_b_id]) == {0, 1}]
    check("Living–bedroom door exists", len(living_bedroom_door) == 1)
    if living_bedroom_door:
        door_pos = living_bedroom_door[0].position
        check("Door at midpoint of shared wall",
              abs(door_pos[0] - mid[0]) < 0.01 and abs(door_pos[1] - mid[1]) < 0.01,
              f"door={door_pos}, expected={mid}")

    # Test to_dict serialization
    dd = d.to_dict()
    check("Door to_dict() has all keys",
          all(k in dd for k in ("door_id", "room_a_id", "room_b_id", "position", "width")),
          f"keys={list(dd.keys())}")
    check("Door position serialized as {x, y}",
          "x" in dd["position"] and "y" in dd["position"],
          f"position={dd['position']}")

# Test with no shared walls (separated rooms)
Room.reset_counter()
separated_rooms = [
    Room(room_type="a",
         polygon=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
         target_area=4),
    Room(room_type="b",
         polygon=Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
         target_area=4),
]
no_doors = place_doors(separated_rooms)
check("No doors for separated rooms", len(no_doors) == 0)

print()


# ====================================================================
print("=" * 60)
print("STEP 12 — Corridor Detection")
print("=" * 60)

# Perfect coverage — no corridor space
boundary12 = Polygon([(0, 0), (12, 0), (12, 10), (0, 10)])
Room.reset_counter()
full_rooms = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (6, 0), (6, 5), (0, 5)]),
         target_area=30),
    Room(room_type="bedroom",
         polygon=Polygon([(6, 0), (12, 0), (12, 5), (6, 5)]),
         target_area=30),
    Room(room_type="kitchen",
         polygon=Polygon([(0, 5), (12, 5), (12, 10), (0, 10)]),
         target_area=60),
]
# Total room area = 30 + 30 + 60 = 120, boundary area = 120
room_area_sum = sum(r.area for r in full_rooms)
corridor_area_full = boundary12.area - room_area_sum
check("Full coverage: corridor area ≈ 0",
      abs(corridor_area_full) < 0.01,
      f"corridor={corridor_area_full:.4f}")

# Partial coverage — corridor space exists
Room.reset_counter()
partial_rooms = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (6, 0), (6, 5), (0, 5)]),
         target_area=30),
    Room(room_type="bedroom",
         polygon=Polygon([(6, 0), (12, 0), (12, 5), (6, 5)]),
         target_area=30),
    # Kitchen is smaller than its slot
    Room(room_type="kitchen",
         polygon=Polygon([(0, 5), (8, 5), (8, 10), (0, 10)]),
         target_area=40),
]
room_area_partial = sum(r.area for r in partial_rooms)
corridor_area_partial = boundary12.area - room_area_partial
check("Partial coverage: corridor area > 0",
      corridor_area_partial > 0,
      f"corridor={corridor_area_partial:.2f} sq m")
check("Corridor fraction computed correctly",
      abs(corridor_area_partial / boundary12.area - 20 / 120) < 0.01,
      f"fraction={corridor_area_partial / boundary12.area:.4f}")

# Test corridor_penalty from scoring module
penalty_full = corridor_penalty(full_rooms, boundary12)
check("corridor_penalty = 0 for full coverage",
      abs(penalty_full) < 0.01,
      f"penalty={penalty_full:.4f}")

penalty_partial = corridor_penalty(partial_rooms, boundary12)
check("corridor_penalty > 0 for partial coverage",
      penalty_partial > 0,
      f"penalty={penalty_partial:.4f}")

# Test _compute_corridor in generator
gen12 = LayoutGenerator(
    boundary=boundary12,
    room_requirements=[
        {"room_type": "living", "size": 5},
        {"room_type": "bedroom", "size": 5},
        {"room_type": "kitchen", "size": 4},
    ],
)
corridor_info = gen12._compute_corridor(full_rooms)
check("_compute_corridor returns area key",
      "area" in corridor_info,
      f"info={corridor_info}")
check("_compute_corridor returns fraction key",
      "fraction" in corridor_info)
check("_compute_corridor area ≈ 0 for full coverage",
      abs(corridor_info["area"]) < 0.01,
      f"area={corridor_info['area']}")

corridor_partial_info = gen12._compute_corridor(partial_rooms)
check("_compute_corridor area > 0 for partial",
      corridor_partial_info["area"] > 0,
      f"area={corridor_partial_info['area']}")

print()


# ====================================================================
print("=" * 60)
print("STEP 13 — Candidate Generation Loop (200 layouts)")
print("=" * 60)

gen13 = LayoutGenerator.from_json(
    polygon_path="usable_polygon.json",
    room_requirements=[
        {"room_type": "living", "size": 7},
        {"room_type": "bedroom", "size": 5},
        {"room_type": "kitchen", "size": 4},
        {"room_type": "bathroom", "size": 3},
    ],
    rules_path="region_rules.json",
    region="india_mvp",
    desired_adjacencies=[("living", "kitchen"), ("bedroom", "bathroom")],
)

# Generate 200 candidates
result200 = gen13.generate(n_candidates=200, method="mixed")
check("200 candidates attempted",
      result200["candidates_generated"] == 200,
      f"generated={result200['candidates_generated']}")
check("Multiple valid candidates produced",
      result200["candidates_valid"] > 0,
      f"valid={result200['candidates_valid']}")

# Also test generate_all_valid to store each candidate
all_valid = gen13.generate_all_valid(n_candidates=200, method="mixed")
check("generate_all_valid returns list",
      isinstance(all_valid, list))
check("All valid candidates stored",
      len(all_valid) > 0,
      f"stored={len(all_valid)}")
check("Each candidate has layout",
      all("layout" in c for c in all_valid))
check("Each candidate has score",
      all("score" in c for c in all_valid))
check("Each candidate has doors",
      all("doors" in c for c in all_valid))
check("Each candidate has corridor info",
      all("corridor" in c for c in all_valid))

# Verify candidate count matches
check("Stored count matches valid count",
      len(all_valid) == result200["candidates_valid"],
      f"stored={len(all_valid)}, valid={result200['candidates_valid']}")

print()


# ====================================================================
print("=" * 60)
print("STEP 14 — Scoring System")
print("=" * 60)

# Area accuracy score
Room.reset_counter()
perfect_rooms = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (5, 0), (5, 4), (0, 4)]),
         target_area=20.0),
    Room(room_type="bedroom",
         polygon=Polygon([(5, 0), (10, 0), (10, 4), (5, 4)]),
         target_area=20.0),
]
s_area_perfect = area_accuracy_score(perfect_rooms)
check("Area score = 1.0 for exact match",
      abs(s_area_perfect - 1.0) < 0.01,
      f"score={s_area_perfect:.4f}")

# Imperfect area
Room.reset_counter()
imperfect_rooms = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),  # 16 sqm
         target_area=20.0),  # wanted 20
]
s_area_imperfect = area_accuracy_score(imperfect_rooms)
check("Area score < 1.0 for mismatch",
      s_area_imperfect < 1.0,
      f"score={s_area_imperfect:.4f} (area=16, target=20)")

# Adjacency score
boundary14 = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
Room.reset_counter()
adj_rooms14 = [
    Room(room_type="living",
         polygon=Polygon([(0, 0), (5, 0), (5, 4), (0, 4)]),
         target_area=20),
    Room(room_type="kitchen",
         polygon=Polygon([(5, 0), (10, 0), (10, 4), (5, 4)]),
         target_area=20),
    Room(room_type="bedroom",
         polygon=Polygon([(0, 4), (10, 4), (10, 8), (0, 8)]),
         target_area=40),
]
s_adj_all = adjacency_score(adj_rooms14, [("living", "kitchen")])
check("Adjacency score = 1.0 when all desires met",
      abs(s_adj_all - 1.0) < 0.01,
      f"score={s_adj_all:.4f}")

s_adj_none = adjacency_score(adj_rooms14, [("living", "toilet")])  # impossible
check("Adjacency score = 0.0 when desire unmet",
      abs(s_adj_none) < 0.01,
      f"score={s_adj_none:.4f}")

# Corridor penalty
penalty_zero = corridor_penalty(
    [Room(room_type="a",
          polygon=Polygon([(0, 0), (10, 0), (10, 8), (0, 8)]),
          target_area=80)],
    boundary14
)
check("Corridor penalty = 0 when fully covered",
      abs(penalty_zero) < 0.01,
      f"penalty={penalty_zero:.4f}")

# Total score
scores14 = score_layout(adj_rooms14, boundary14, [("living", "kitchen")])
check("score_layout returns total",
      "total" in scores14, f"keys={list(scores14.keys())}")
check("score_layout returns area component",
      "area" in scores14)
check("score_layout returns adjacency component",
      "adjacency" in scores14)
check("score_layout returns corridor component",
      "corridor" in scores14)
check("Total score is weighted sum",
      scores14["total"] > 0,
      f"total={scores14['total']:.4f}")

# Verify weights (default: area=0.4, adjacency=0.35, corridor=0.25)
expected_total = (0.4 * scores14["area"]
                  + 0.35 * scores14["adjacency"]
                  + 0.25 * scores14["corridor"])
check("Total ≈ weighted sum of components",
      abs(scores14["total"] - round(expected_total, 4)) < 0.01,
      f"computed={expected_total:.4f}, reported={scores14['total']}")

print()


# ====================================================================
print("=" * 60)
print("STEP 15 — Best Candidate Selection")
print("=" * 60)

# Use the result from step 13 generation
check("Best layout exists in result",
      len(result200["best_layout"]) > 0,
      f"rooms={len(result200['best_layout'])}")
check("Best score is positive",
      result200["score"]["total"] > 0,
      f"total={result200['score']['total']:.4f}")
check("Doors included in best result",
      "doors" in result200,
      f"door_count={len(result200.get('doors', []))}")
check("Corridor info in best result",
      "corridor" in result200,
      f"corridor={result200.get('corridor')}")

# Verify best candidate is actually the highest score
if all_valid:
    best_score_from_all = all_valid[0]["score"]["total"]
    check("Best candidate has highest score",
          abs(best_score_from_all - result200["score"]["total"]) < 0.01,
          f"best_all={best_score_from_all:.4f}, best_gen={result200['score']['total']:.4f}")

    # Verify sorted order (descending)
    scores_list = [c["score"]["total"] for c in all_valid]
    is_sorted = all(scores_list[i] >= scores_list[i + 1]
                     for i in range(len(scores_list) - 1))
    check("All candidates sorted by score (descending)", is_sorted)

# Verify no invalid layout passes (rejection works)
gen_strict = LayoutGenerator(
    boundary=Polygon([(0, 0), (12, 0), (12, 10), (0, 10)]),
    room_requirements=[{"room_type": "living", "size": 5}],
    min_areas={"living": 99999.0},  # impossibly high
)
strict_result = gen_strict.generate(n_candidates=20)
check("Invalid layouts rejected (min area violation)",
      strict_result["candidates_valid"] == 0,
      f"valid={strict_result['candidates_valid']}")

print()


# ====================================================================
print("=" * 60)
print("COMPLETION CRITERIA — Full System Verification")
print("=" * 60)

check("System loads usable polygon",
      gen13.boundary is not None and gen13.boundary.area > 0)
check("System generates layouts",
      result200["candidates_valid"] > 0)
check("System rejects invalid layouts",
      strict_result["candidates_valid"] == 0)
check("System scores layouts",
      result200["score"]["total"] > 0)
check("System selects best layout",
      len(result200["best_layout"]) > 0)

print()


# ====================================================================
# Summary
# ====================================================================
print("=" * 60)
passed = sum(results)
total = len(results)
pct = (passed / total * 100) if total else 0
color = "\033[92m" if passed == total else "\033[93m"
print(f"{color}RESULTS: {passed}/{total} checks passed ({pct:.0f}%)\033[0m")
print("=" * 60)

sys.exit(0 if passed == total else 1)
