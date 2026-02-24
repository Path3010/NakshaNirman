"""Quick live API test — run while backend is running on port 8000."""
import requests
import json

BASE = "http://localhost:8000"

# Test 1: API docs
r = requests.get(f"{BASE}/docs")
print(f"1. API Docs: {r.status_code}")

# Test 2: AI analyze
r2 = requests.post(f"{BASE}/api/ai-design/analyze",
    json={"message": "30x40 plot, 3 bedrooms, 2 bathrooms"})
print(f"2. AI Analyze: {r2.status_code}")
if r2.status_code == 200:
    data = r2.json()
    print(f"   Provider: {data.get('provider', '?')}")
    print(f"   Rooms: {len(data.get('rooms', []))}")
    print(f"   Score: {data.get('design_score', 0)}")

# Test 3: Pipeline endpoint
r3 = requests.post(f"{BASE}/api/ai-design/pipeline",
    json={"requirements_json": {
        "plot_width": 30, "plot_length": 40, "total_area": 1200,
        "bedrooms": 3, "bathrooms": 2, "floors": 1, "extras": []
    }})
print(f"3. Pipeline: {r3.status_code}")
if r3.status_code == 200:
    data = r3.json()
    print(f"   Stage: {data.get('stage', '?')}")
    print(f"   Compliant: {data.get('compliant', '?')}")
    layout = data.get("layout_json", {})
    rooms = layout.get("rooms", [])
    print(f"   Rooms generated: {len(rooms)}")
    for rm in rooms[:6]:
        print(f"     - {rm.get('name','?')} ({rm.get('room_type','?')}): {rm.get('area',0)} sqft")

print()
print("All live API tests complete!")
