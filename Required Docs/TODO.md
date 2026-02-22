# todo.md
# Project ToDo (Python-only, 12-week execution)

Rule: Geometry generation is procedural/optimization.  
GNN is ranking only.  
Multi-floor is heuristic stacking.

---

## Repository Setup (Day 1–2)

- Current repo structure:

backend/  
backend/main.py  
backend/config.py  
backend/database.py  
backend/models.py  
backend/schemas.py  
backend/routes/  
backend/services/  
backend/requirements.txt  
backend/Dockerfile  
frontend/  
frontend/src/  
frontend/src/components/  
frontend/src/pages/  
frontend/public/  
frontend/package.json  
frontend/vite.config.js  
frontend/Dockerfile  
Required Docs/  
docker-compose.yml  
nginx.conf

- Create virtual environment
- Install packages:
  - fastapi, uvicorn
  - shapely
  - ezdxf
  - numpy
  - networkx
  - torch, torch-geometric
  - matplotlib (debug plots)

- Add config files:
  - region_rules.json
  - scoring_weights.json

Deliverable: Running FastAPI server + clean project skeleton

---

## Phase 1 — Input + Boundary Handling (Week 1–2)

### DXF Upload API
- Endpoint: POST /upload-boundary
- Accept .dxf file
- Save file locally or in DB
- Output: file_id
  
### Boundary Extraction
- Parse DXF with ezdxf
- Extract closed polyline (plot boundary)
- Convert to shapely Polygon
- Validate:
  - closed
  - non-self-intersecting
  - area > 0
- Output: boundary_polygon.json

### Buildable Footprint
- Apply setback offset inward (India MVP rule)
- Remove invalid offsets
- Return usable_polygon
- Output: usable_polygon.json + preview image

Deliverable: Upload DXF → usable build area computed

---

## Phase 2 — Requirements System (Week 2) ✅

### Requirement Schema ✅
Strict JSON format:

Hard constraints:
- floors ✅
- bedrooms ✅
- bathrooms ✅
- kitchen ✅
- max_area ✅

Soft constraints:
- balcony ✅
- parking ✅
- pooja_room ✅

Output: requirements.json ✅ (GET /api/requirements/:id/json)

### Requirement Input API ✅
- Endpoint: POST /requirements ✅
- GET /api/requirements/:id ✅
- GET /api/requirements/project/:project_id ✅
- Store structured requirements ✅
- Frontend RequirementsForm integrated ✅

Deliverable: Boundary + requirements stored ✅

---

## Phase 3 — Single-Floor Layout Engine (Week 3–5)

### Room Object Model
Implement Room class:
- type
- min_area
- target_area
- polygon (shapely)

Deliverable: Room primitives exist

### Candidate Layout Generator v1
- Start with rectangle subdivision (BSP split)
- Generate 20 random partitions
- Assign rooms to partitions
- Output: candidate_floorplan_001.json

### Validity Checks
Reject candidate if:
- overlap exists
- outside usable polygon
- room area < minimum
- missing required rooms

Deliverable: Valid candidates only

### Connectivity + Doors
- Build adjacency graph (rooms touching)
- Ensure all rooms reachable from entrance
- Add door objects between connected rooms

Deliverable: Functional floor plan graph

### Scoring System v1
Score components:
- area accuracy
- adjacency satisfaction
- corridor waste penalty

Output: score per candidate

Deliverable: Best plan selected for 1 floor

---

## Phase 4 — Multi-Floor Expansion (Week 6–8)

### Floor Allocation Heuristic
Split rooms across floors:
- Floor 1: living, kitchen, 1 toilet
- Floor 2+: bedrooms, toilets

Deliverable: Floor-wise room list

### Staircase Constraint
- Add staircase as mandatory node
- Same position across floors

Deliverable: Vertical connectivity works

### Multi-Floor Candidate Generation
For each candidate:
- Generate floor1 layout
- Generate floor2 layout under same footprint
- Stack wet areas vertically (toilet alignment)

Deliverable: Multi-floor candidates produced

### Multi-Floor Scoring
Add penalties:
- staircase inefficiency
- misaligned plumbing
- privacy issues

Deliverable: Best multi-floor plan chosen

---

## Phase 5 — Graph + GNN Ranking (Week 9–10)

### Candidate → Graph Conversion
Nodes: rooms  
Edges: adjacency/doors  

Node features:
- area error
- aspect ratio
- floor index

Deliverable: Graph dataset builder

### Synthetic Label Generator
- Use heuristic score as training label

Deliverable: Training data without manual annotation

### Train GNN Scorer
Model: GraphSAGE or GAT  
Task: predict candidate quality

Deliverable: GNN ranks candidates similar to heuristic

### Hybrid Ranking
Pipeline:
- Generate 200 candidates
- Heuristic filter top 50
- GNN rerank top 10
- Output best

Deliverable: Smarter final selection

---

## Phase 6 — DXF Export + App Completion (Week 11–12)

### DXF Writer
Use ezdxf layers:
- WALLS
- DOORS
- WINDOWS
- LABELS

Export per floor:
- floor1.dxf
- floor2.dxf

Deliverable: Downloadable CAD files

### Preview Generator
- Convert shapely polygons → SVG/PNG preview

Deliverable: User sees plan before download

### Full Workflow API
Endpoints:
- /upload-boundary
- /requirements
- /generate-plan
- /download/{plan_id}

Deliverable: End-to-end automation

### Persistence (Optional DB)
SQLite + SQLAlchemy  
Store:
- boundaries
- requirements
- final plans

Deliverable: Repeatable project runs

### Final Testing
Test cases:
- small plot
- large plot
- 1-floor vs 3-floor
- impossible constraints → graceful failure

Deliverable: Stable demo

---

## Final Output (What Must Work)

- User uploads boundary DXF
- User selects requirements + floors
- System generates 50–200 candidates
- Hard constraints filter
- Scoring + GNN ranking picks best
- Multi-floor DXF files downloadable

---

## Strict MVP Limits

- Rectangular rooms only
- India rules simplified
- Max 3 floors
- GNN only ranks, not draws

This ToDo is sufficient to finish in 12 weeks using only Python.
