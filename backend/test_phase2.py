"""
Phase 2 end-to-end test — Requirements System
Run:  python test_phase2.py
"""
import urllib.request, json, sys

BASE = "http://127.0.0.1:8001"

def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode())

def ok(label):
    print(f"  ✓ {label}")

# ------------------------------------------------------------------
print("Phase 2 — Requirements System Tests")
print("=" * 50)

# 1. POST /api/requirements (no project)
print("\n1. POST /api/requirements (standalone)")
res = api("POST", "/api/requirements", {
    "floors": 2, "bedrooms": 3, "bathrooms": 2, "kitchen": 1,
    "max_area": 250.0, "balcony": True, "parking": True, "pooja_room": False,
})
req_id = res["id"]
assert res["floors"] == 2 and res["bedrooms"] == 3
assert res["balcony"] is True and res["pooja_room"] is False
ok(f"Created requirements {req_id[:8]}…")

# 2. GET /api/requirements/:id
print("\n2. GET /api/requirements/:id")
res2 = api("GET", f"/api/requirements/{req_id}")
assert res2["id"] == req_id
assert res2["max_area"] == 250.0
ok("Retrieved by ID")

# 3. GET /api/requirements/:id/json  (strict output format)
print("\n3. GET /api/requirements/:id/json")
rjson = api("GET", f"/api/requirements/{req_id}/json")
assert "hard_constraints" in rjson and "soft_constraints" in rjson
assert rjson["hard_constraints"]["floors"] == 2
assert rjson["soft_constraints"]["balcony"] is True
ok("Strict JSON format OK")

# 4. POST with project_id, then GET by project
print("\n4. POST with project_id → GET by project")
proj = api("POST", "/api/project", {"session_id": "test_phase2", "total_area": 1500})
pid = proj["project_id"]
res3 = api("POST", "/api/requirements", {
    "floors": 1, "bedrooms": 2, "bathrooms": 1, "kitchen": 1,
    "max_area": 150.0, "balcony": False, "parking": False, "pooja_room": True,
    "project_id": pid,
})
assert res3["project_id"] == pid
ok(f"Linked to project {pid[:8]}…")

res4 = api("GET", f"/api/requirements/project/{pid}")
assert res4["project_id"] == pid and res4["bedrooms"] == 2
ok("Retrieved by project_id")

# 5. Validation — missing required field
print("\n5. Validation — missing required field")
try:
    api("POST", "/api/requirements", {"floors": 1})
    print("  ✗ Should have failed")
    sys.exit(1)
except urllib.error.HTTPError as e:
    assert e.code == 422
    ok("422 for missing fields")

print("\n" + "=" * 50)
print("All Phase 2 tests passed!")
