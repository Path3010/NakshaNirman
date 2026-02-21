"""Generate sample DXF test files for Phase 1 testing.

Creates 6 DXF files:
  - 3 squares  (10m, 15m, 20m side)
  - 3 rectangles (12x8m, 20x15m, 30x18m)

Each file contains a single closed LWPOLYLINE representing the plot boundary.
"""

import ezdxf
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)


def create_dxf(filename: str, points: list[tuple[float, float]], label: str):
    """Create a DXF file with a single closed LWPOLYLINE boundary."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Add boundary polyline
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "BOUNDARY"})

    # Add dimension labels on edges
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        length = (dx**2 + dy**2) ** 0.5
        msp.add_text(
            f"{length:.1f}m",
            dxfattribs={
                "layer": "LABELS",
                "height": max(0.3, length * 0.05),
                "insert": (mx, my),
            },
        )

    # Add plot label at centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    msp.add_text(
        label,
        dxfattribs={
            "layer": "LABELS",
            "height": 0.5,
            "insert": (cx, cy - 0.8),
        },
    )

    filepath = SAMPLES_DIR / filename
    doc.saveas(str(filepath))
    print(f"  Created: {filepath.name}  ({label})")
    return filepath


def main():
    print("Generating sample DXF files...\n")

    # --- Squares ---
    create_dxf(
        "square_10x10.dxf",
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        "Square 10m x 10m (100 sq.m)",
    )
    create_dxf(
        "square_15x15.dxf",
        [(0, 0), (15, 0), (15, 15), (0, 15)],
        "Square 15m x 15m (225 sq.m)",
    )
    create_dxf(
        "square_20x20.dxf",
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        "Square 20m x 20m (400 sq.m)",
    )

    # --- Rectangles ---
    create_dxf(
        "rect_12x8.dxf",
        [(0, 0), (12, 0), (12, 8), (0, 8)],
        "Rectangle 12m x 8m (96 sq.m)",
    )
    create_dxf(
        "rect_20x15.dxf",
        [(0, 0), (20, 0), (20, 15), (0, 15)],
        "Rectangle 20m x 15m (300 sq.m)",
    )
    create_dxf(
        "rect_30x18.dxf",
        [(0, 0), (30, 0), (30, 18), (0, 18)],
        "Rectangle 30m x 18m (540 sq.m)",
    )

    print(f"\nAll 6 sample DXFs saved to: {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
