#!/usr/bin/env python3
"""
Convert data/coordinates.csv and data/data.csv into NeTrainSim .dat input files.

Input files (in data/):
  coordinates.csv  tab-separated, no header: node_id  x_meters  y_meters  (750 nodes, IDs 1-750)
  data.csv         comma-separated, header:  index, Grade(‰), Curvature, Speed limit(m/s)  (749 segments)

Outputs to data/netrainsim/:
  nodesFile.dat   – node definitions
  linksFile.dat   – link definitions (one link per consecutive node pair)
  trainsFile.dat  – single diesel locomotive with a light consist

Unit notes:
  Speed limit: m/s  — values 11.1/16.6/19.4/22.2 = 40/60/70/80 km/h ÷ 3.6
  Grade:       ‰ (per mille) — divided by 10 before writing so NeTrainSim sees % (its expected unit)
               e.g. 6.28‰ → stored as 0.628% in linksFile.dat
  Curvature:   as provided (passed through unchanged)

Train configuration:
  1 diesel locomotive (type 1): 5000 kW, 90 t, 4 axles — can pull consist on max 6.28‰ grade
  4 freight cars (type 1): 50 t gross each — total consist ~290 t
  Max loco speed: 33.33 m/s (simulator default); route speed limits are the binding constraint
"""

import csv, math, os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "netrainsim")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Read coordinates (node_id → (x_m, y_m)) ───────────────────────────────
coords: dict[int, tuple[float, float]] = {}
with open(os.path.join(DATA_DIR, "coordinates.csv")) as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            coords[int(float(parts[0]))] = (float(parts[1]), float(parts[2]))

# ── Read segment data (segment_idx → {grade, curvature, speed_mps}) ────────
segments: dict[int, dict] = {}
with open(os.path.join(DATA_DIR, "data.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        idx = int(row[""])
        segments[idx] = {
            "grade":     float(row["Grade"]),
            "curvature": float(row["Curvature"]),
            "speed_mps": float(row["Speed limit"]),
        }

sorted_node_ids = sorted(coords.keys())
sorted_seg_ids  = sorted(segments.keys())

assert len(sorted_node_ids) == 750, f"Expected 750 nodes, got {len(sorted_node_ids)}"
assert len(sorted_seg_ids)  == 749, f"Expected 749 segments, got {len(sorted_seg_ids)}"

# ── nodesFile.dat ──────────────────────────────────────────────────────────
# Line 1: header text
# Line 2: count  xScale  yScale  (scales = 1; coordinates already in meters)
# Lines 3+: id  x  y  isTerminal  dwellTime  desc
NODES_PATH = os.path.join(OUT_DIR, "nodesFile.dat")
with open(NODES_PATH, "w") as f:
    f.write("This is the node file of route1\t\t\n")
    f.write(f"{len(sorted_node_ids)}\t1\t1\n")
    for nid in sorted_node_ids:
        x, y = coords[nid]
        is_terminal = 1 if nid in (sorted_node_ids[0], sorted_node_ids[-1]) else 0
        f.write(f"{nid}\t{x}\t{y}\t{is_terminal}\t0\tND\n")
print(f"Written {NODES_PATH}  ({len(sorted_node_ids)} nodes)")

# ── linksFile.dat ──────────────────────────────────────────────────────────
# Line 1: header text
# Line 2: count  lengthScale  speedScale  (both = 1; lengths in meters, speed in m/s)
# Lines 3+: id  fromNode  toNode  length_m  speed_mps  signalNo  grade  curvature
#           directions  speedVariation  hasCatenary
# directions=1: unidirectional A→B; speedVariation=0.2 (matches sample); hasCatenary=0 (diesel)
LINKS_PATH = os.path.join(OUT_DIR, "linksFile.dat")
with open(LINKS_PATH, "w") as f:
    f.write("This is the link file of route1\t\t\t\t\t\t\t\t\t\t\t\n")
    f.write(f"{len(sorted_seg_ids)}\t1\t1\n")
    for i, seg_idx in enumerate(sorted_seg_ids):
        from_id = sorted_node_ids[i]
        to_id   = sorted_node_ids[i + 1]
        x1, y1  = coords[from_id]
        x2, y2  = coords[to_id]
        length_m = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        seg = segments[seg_idx]
        grade_pct = seg["grade"] / 10.0  # convert ‰ → % (NeTrainSim expects %)
        f.write(
            f"{seg_idx}\t{from_id}\t{to_id}\t{length_m:.4f}\t"
            f"{seg['speed_mps']:.5f}\t0\t{grade_pct:.14f}\t"
            f"{seg['curvature']:.14f}\t1\t0.2\t0\n"
        )
print(f"Written {LINKS_PATH}  ({len(sorted_seg_ids)} links)")

# ── trainsFile.dat ─────────────────────────────────────────────────────────
# Light consist designed for this route's grade profile (max 6.28‰ = 0.628%).
#
# Locomotive field order: Count, Power(kW), TransmissionEff, NoOfAxles,
#   AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), Type
# Car field order:        Count, NoOfAxles, AirDragCoeff, FrontalArea(m²),
#   Length(m), GrossWeight(t), TareWeight(t), Type
#
# 1 diesel loco (type 1): 5000 kW, 90 t, 4 axles
#   adhesion force ≈ 0.30 × 90 t × 9.81 = 265 kN
#   grade resistance at 0.628%: ~290t × 9.81 × 0.00628 ≈ 17.9 kN → well within limits
# 4 freight cars (type 0): 50 t gross, 15 t tare, 4 axles each
#   total consist weight ≈ 90 + 4×50 = 290 t
LOCO_DEF = "1,5000,0.85,4,0.002,10.0,20,90,1"
CAR_DEF  = "4,4,0.0005,8.0,15,50,15,0"

path_str = ",".join(str(nid) for nid in sorted_node_ids)

TRAINS_PATH = os.path.join(OUT_DIR, "trainsFile.dat")
with open(TRAINS_PATH, "w") as f:
    f.write("Automatic Trains Definition\n")
    f.write("1\n")
    f.write(f"1\t{path_str}\t0\t0.25\t{LOCO_DEF}\t{CAR_DEF}\n")
print(f"Written {TRAINS_PATH}  (train path: node 1 → node {sorted_node_ids[-1]})")

# ── Route statistics ───────────────────────────────────────────────────────
total_len_m = sum(
    math.sqrt(
        (coords[sorted_node_ids[i + 1]][0] - coords[sorted_node_ids[i]][0]) ** 2 +
        (coords[sorted_node_ids[i + 1]][1] - coords[sorted_node_ids[i]][1]) ** 2
    )
    for i in range(len(sorted_node_ids) - 1)
)
speed_limits_mps = sorted({v["speed_mps"] for v in segments.values()})
print(f"\nRoute statistics:")
print(f"  Nodes:          {len(sorted_node_ids)}")
print(f"  Links:          {len(sorted_seg_ids)}")
print(f"  Total length:   {total_len_m / 1000:.2f} km")
print(f"  Avg segment:    {total_len_m / len(sorted_seg_ids):.1f} m")
print(f"  Speed limits:   {speed_limits_mps} m/s")
print(f"                  {[round(s * 3.6, 1) for s in speed_limits_mps]} km/h")
