from geneuler.mesh import Mesh
from geneuler.fem import SkFemElasticity
import numpy as np
import meshio, os

# --- helper to emulate a box traction by spreading a force over all nodes in a box ---
def apply_box_nodal_load(fem, mesh, lo_mm, hi_mm, total_force_N, radius_mm=0.5):
    P = mesh.points
    lo = np.array(lo_mm, dtype=float)
    hi = np.array(hi_mm, dtype=float)
    mask = np.all((P >= lo) & (P <= hi), axis=1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        print(f"WARNING: no nodes found in box {lo_mm}–{hi_mm}")
        return idx
    fnode = (np.array(total_force_N, dtype=float) / idx.size)
    for i in idx:
        fem.add_point_load(center_mm=tuple(P[i]), force_N=tuple(fnode), radius_mm=radius_mm)
    print(f"Applied box load to {idx.size} nodes; per-node force = {tuple(fnode)} N")
    return idx

# --- 1) Mesh (mm) ---
m = Mesh.from_bbox(
    {"x": (-60, 60), "y": (0, 150), "z": (-120, 120)},
    h=10.0,
    refine_balls=[((0, 130, -110), 20.0, 5.0)],
    floor_slab=(0.0, 8.0, 6.0),
    embed_grid_h=30.0,
    embed_points_mm=[(0, 75, 0), (20, 90, -40)],
    # remove this line if your Mesh doesn't support it:
    uniform_refine=1,
)
m.save_xdmf("out/box")

# --- 2) FEM setup (mm–N–MPa) ---
fem = SkFemElasticity(m)
# Aluminum-like (E in MPa, rho in kg/mm^3)
fem.set_material(E=70_000.0, nu=0.33, rho=2.70e-6)
fem.set_gravity(9810.0)  # mm/s^2 (9.81 m/s^2)

# --- boundary ---
fem.fix_floor(y_mm=0.0)

# --- loads (moderate; create spatial variation) ---
fem.add_point_load(center_mm=(0, 130, -110), force_N=(0.0, -1000.0, 0.0), radius_mm=6.0)
fem.add_point_load(center_mm=(40, 145,  90), force_N=(0.0,  -600.0, 0.0), radius_mm=8.0)

# optional: spread -2000 N over a small top patch to create a broader gradient
apply_box_nodal_load(
    fem, m,
    lo_mm=(20, 140, 60), hi_mm=(60, 150, 100),
    total_force_N=(0.0, -2000.0, 0.0),
    radius_mm=0.25,
)

# --- 4) Solve ---
result = fem.solve(solver="direct")

# --- 5) Displacements: make sure we have (N,3) ---
u_raw = np.asarray(result["u"])
if u_raw.ndim == 1:
    u_vec = u_raw.reshape((-1, 3))
elif u_raw.ndim == 2 and u_raw.shape[1] == 3:
    u_vec = u_raw
elif u_raw.ndim == 2 and u_raw.shape[0] == 3:
    u_vec = u_raw.T
else:
    raise ValueError(f"Unexpected displacement shape {u_raw.shape}; expected (3N,), (N,3) or (3,N).")

u_mag = np.linalg.norm(u_vec, axis=1)

# Standard exports
fem.export_vtu("out/fem_result.vtu", u_raw)
fem.export_xdmf("out/fem_result", u_raw)

# --- 6) Extras for ParaView: include guaranteed-visible gradients & normalized fields ---
os.makedirs("out", exist_ok=True)

def norm01(a):
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a - lo) / max(hi - lo, 1e-30), 0, 1)

# Simple vertical coordinate gradient (always shows color variation)
y = m.points[:, 1]
coord_y_norm = (y - y.min()) / max((y.max() - y.min()), 1e-30)

# Remove best-fit linear trend in u_y (helps reveal local effects when gravity dominates)
A = np.c_[y, np.ones_like(y)]
coef, *_ = np.linalg.lstsq(A, u_vec[:, 1], rcond=None)  # [a, b] for u_y ≈ a*y + b
uy_trend = A @ coef
u_y_resid = u_vec[:, 1] - uy_trend
u_y_resid_norm = norm01(u_y_resid)

uy = u_vec[:, 1]
uy_demean = uy - np.median(uy)
u_mag_demean = u_mag - np.median(u_mag)

extra = meshio.Mesh(
    points=m.points,
    cells=[("tetra", m.tets)],
    point_data={
        "u": u_vec,                    # vector field
        "u_mag": u_mag,                # scalar magnitude
        "u_mag_norm": norm01(u_mag),   # 0..1
        "u_y": uy,
        "u_y_demean": uy_demean,
        "u_mag_demean": u_mag_demean,
        "u_y_norm": norm01(uy),        # 0..1
        "coord_y_norm": coord_y_norm,  # 0..1 vertical gradient (sanity check)
        "u_y_resid_norm": u_y_resid_norm,  # 0..1 after removing linear trend vs y
    },
)
meshio.write("out/fem_result_extras.vtu", extra)
meshio.write("out/fem_result_extras.xdmf", extra)

# --- 7) quick ranges to sanity-check dynamic range ---
print("u_y range:    ", float(uy.min()), float(uy.max()))
print("|u|  range:    ", float(u_mag.min()), float(u_mag.max()))
print("u_y resid rng:", float(u_y_resid.min()), float(u_y_resid.max()))
print("Wrote out/fem_result_extras.[vtu|xdmf] with u, u_mag, u_y, and normalized/diagnostic fields.")
