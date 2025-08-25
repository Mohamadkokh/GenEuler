# GenEuler — Mesh & FEM

A compact pipeline for going from simple geometry to linear elasticity results:

* **Meshing** with [Gmsh] via a thin wrapper (`geneuler.mesh.Mesh`)
* **Solving** small-strain linear elasticity with [scikit-fem] (`geneuler.fem.SkFemElasticity`)
* **I/O** with [meshio] (Gmsh `.msh`, `.vtu`, `.xdmf`)

> TL;DR: Build a tet mesh (mm), run an SI-unit FEM solve (m / N / Pa), export to ParaView.

---

## Features

### Meshing (`geneuler.mesh`)

* Axis-aligned **box** meshing with local refinement:

  * spherical/box size fields (`refine_balls`, `refine_boxes`)
  * floor-slab refinement band (by **y**)
  * interior seed grids / custom seed points (to help 3D meshing)
  * optional uniform post-refinement (`uniform_refine`)
* Import **STL/OBJ/PLY/STEP/IGES** with robust fallbacks:

  * quick watertight preflight for STL/OBJ/PLY
  * surface classification & volume reconstruction when possible
  * shell-only fallback if not watertight or parametrization fails
* Convenient boundary queries (outer box, floor plane, “attachment” neighborhoods)
* Export to **.msh** and **.xdmf**(+**.h5** when large)

### FEM (`geneuler.fem`)

* 3D, small-strain **linear elasticity** on tetrahedra (P1 vector)
* SI units internally (geometry auto-converted from mm → m)
* Body force from gravity (ρg), traction loads on selected boundary facets
* Handy helpers:

  * **fix\_floor(y=0)**: clamp all dofs on a plane
  * fix facets in a box / by custom selector
  * **add\_point\_load**: distribute a *total* force over facets near a point
* Solvers: direct (`spsolve`) or CG with optional **pyamg**
* Exports **VTU/XDMF** with displacement field
* Returns **compliance** $uᵀf$

---

## Install

```bash
pip install numpy meshio gmsh scikit-fem scipy
# optional for CG preconditioning
pip install pyamg
```

> On macOS/Apple Silicon, use a recent Python & pip for the gmsh wheel.

---

## Units & Conventions

* **Mesh input:** millimeters (mm)
* **FEM (internal):** meters (m), Newtons (N), Pascals (Pa), kg/m³

  * Mesh geometry is converted mm → m in `SkFemElasticity`.
  * `E` in Pa, `rho` in kg/m³, gravity in m/s², tractions in Pa.

**Coordinate system:** right-handed `(x, y, z)`. By convention **y** is “vertical” and used by `fix_floor`/`floor_nodes`.

---

## Quick Start

```python
from geneuler.mesh import Mesh
from geneuler.fem import SkFemElasticity

# 1) Mesh (mm)
m = Mesh.from_bbox(
    {"x": (-60, 60), "y": (0, 150), "z": (-120, 120)},
    h=10.0,
    refine_balls=[((0, 130, -110), 20.0, 5.0)],
    floor_slab=(0.0, 8.0, 6.0),
    embed_grid_h=30.0,
    embed_points_mm=[(0, 75, 0), (20, 90, -40)],
    uniform_refine=1,
)
m.save_xdmf("out/box")  # writes out/box.xdmf

# 2) FEM (SI units)
fem = SkFemElasticity(m)
fem.set_material(E=210e9, nu=0.30, rho=7850.0)  # steel-like (Pa, -, kg/m^3)
fem.set_gravity(9.81)                           # m/s^2 (defaults to -y)

# Fix floor (y=0)
fem.fix_floor(y_mm=0.0)

# Total forces (N) distributed to nearby boundary facets
fem.add_point_load(center_mm=(0, 130, -110), force_N=(0, -800, 0), radius_mm=6.0)
fem.add_point_load(center_mm=(20, 140,    0), force_N=(0, -500, 0), radius_mm=8.0)

# 3) Solve
res = fem.solve(solver="direct")
u = res["u"]
print("Compliance:", res["compliance"])

# 4) Export results
fem.export_vtu("out/fem_result.vtu", u)
fem.export_xdmf("out/fem_result", u)
```

Open `out/fem_result.xdmf` in ParaView and color by **u** (vector) or compute magnitude with “Calculator” (`mag(u)`).

---

## Mesh API

### Class: `geneuler.mesh.Mesh`

```python
@dataclass
class Mesh:
    points: np.ndarray                # (N, 3) float, mm
    tets:   np.ndarray                # (M, 4) int (may be empty for shells)
    tris:   Optional[np.ndarray] = None  # (K, 3) int (outer surface)
```

#### Constructors

* `Mesh.from_msh(path) -> Mesh`
  Load an existing Gmsh mesh. Works for shell or volume.

* `Mesh.from_bbox(bbox, *, h=4.0, refine_balls=None, refine_boxes=None, floor_slab=None, embed_grid_h=None, embed_points_mm=None, msh_path="out/box.msh", msh_include_surface=False, uniform_refine=0) -> Mesh`
  Build a tet mesh of an axis-aligned box, in **mm**.

  Key options:

  * `refine_balls=[((cx,cy,cz), radius, h_local), ...]`
  * `refine_boxes=[(xmin,xmax,ymin,ymax,zmin,zmax,h_local), ...]`
  * `floor_slab=(y, thickness, h_local)` → refinement band across x/z
  * `embed_grid_h` and `embed_points_mm` → embed interior seeds
  * `uniform_refine` → call Gmsh’s refine pass N times after 3D meshing

* `Mesh.from_surface(path, *, h=4.0, reconstruct_volume=True, angle_deg=40.0, include_boundary=True, force_parametrization=False, curve_angle_deg=180.0, msh_path="out/surface.msh", allow_shell_fallback=True, refine_balls=None, refine_boxes=None, floor_slab=None, embed_grid_h=None, embed_points_mm=None, uniform_refine=0) -> Mesh`
  Import STL/OBJ/PLY/STEP/IGES. If watertight (tri meshes) or solid (CAD), attempts a **volume**; else returns **shell**. Same refinement & seeding options as above; refined fields apply to both shell and volume.

#### Helpers

* `save_msh(path)`, `save_xdmf(path_without_ext) -> (xdmf, h5_or_empty)`
* `boundary_nodes(bbox=DEFAULT_BBOX, tol=1e-6) -> np.ndarray`
  Indices on faces of `bbox`.
* `floor_nodes(y=0.0, tol=1e-6) -> np.ndarray`
  Nodes with `Y≈y`; uses min(Y) if `y=None`.
* `attachment_nodes(attachments, default_radius=5.0, return_groups=False)`
  `attachments=[((x,y,z), r), (x2,y2,z2), ...]`. Returns union or groups.

---

## FEM API

### Class: `geneuler.fem.SkFemElasticity`

Small-strain 3D elasticity on tetrahedra (P1 vector, 3 dofs/node). Internally uses SI; mm → m conversion is automatic.

#### Construction

```python
fem = SkFemElasticity(gmesh: Mesh)
```

#### Material & Gravity

```python
fem.set_material(E: float, nu: float, rho: Optional[float] = None)
# E [Pa], nu [-], rho [kg/m^3]

fem.set_gravity(g_xyz: Sequence[float] | float)
# accepts (gx, gy, gz) [m/s^2] or scalar -> (0, -scalar, 0)
```

#### Dirichlet BCs

```python
fem.fix_floor(y_mm: float = 0.0, tol_mm: float = 1e-6)
fem.fix_facets_in_box(lo_mm: Sequence[float], hi_mm: Sequence[float])
fem.fix_facets_selector(selector: Callable[[np.ndarray], np.ndarray])
# selector gets facet centroids in meters, returns boolean mask
```

#### Loads

```python
fem.add_point_load(
    center_mm=(x,y,z),          # where to search facets (mm)
    force_N=(Fx,Fy,Fz),         # TOTAL force to distribute [N]
    radius_mm=6.0,              # select facets within this distance
    fallback_k=20               # if none in radius, take k nearest facets
)
```

> Implementation detail: computes true point-to-triangle distances on the boundary, sums areas of selected facets, applies a **constant traction** `t = F_total / A_total` (Pa) on them.

#### Solve

```python
res = fem.solve(solver="direct" | "cg", cg_tol=1e-8, cg_maxit=5000)
u = res["u"]                # global DOF vector (size = 3*N_nodes)
compliance = res["compliance"]
```

* Direct: `scipy.sparse.linalg.spsolve`
* CG: optional **pyamg** preconditioner if available

#### Export

```python
fem.export_vtu("out/sol.vtu", u, field_name="u")
fem.export_xdmf("out/sol", u, field_name="u")  # writes out/sol.xdmf (+ .h5 if large)
```

---

## Examples

### A) Box + local refinement + two forces

See **Quick Start** above.

### B) Import STL (automatic shell/volume)

```python
m = Mesh.from_surface("part.stl", h=3.0, reconstruct_volume=True, msh_path="out/part.msh")
m.save_xdmf("out/part")
```

### C) Spread a total force over a patch (box selector)

```python
import numpy as np

def apply_box_nodal_load(fem, mesh, lo_mm, hi_mm, total_force_N, radius_mm=0.4):
    P = mesh.points
    lo, hi = np.array(lo_mm), np.array(hi_mm)
    idx = np.where(np.all((P >= lo) & (P <= hi), axis=1))[0]
    if idx.size == 0:
        print("No nodes in patch"); return
    fnode = np.array(total_force_N, float) / idx.size
    for i in idx:
        fem.add_point_load(center_mm=tuple(P[i]), force_N=tuple(fnode), radius_mm=radius_mm)

# usage
apply_box_nodal_load(fem, m, lo_mm=(20,140,60), hi_mm=(60,150,100), total_force_N=(0,-2000,0))
```

### D) ParaView tips (color gradients)

* Open `*.xdmf` and color by **Point Arrays** → `u` (vector) with **Glyph** or **Warp by Vector**.
* For a scalar: add a Calculator filter with `mag(u)` and use that to color.
* Click **Rescale to Data Range** to avoid a flat color.
* If you want a guaranteed gradient for debugging, add a scalar like normalized Y:

  ```python
  import meshio, numpy as np
  y = m.points[:,1]
  coord_y_norm = (y - y.min()) / (y.max() - y.min() + 1e-30)
  meshio.write("out/debug_y.vtu", meshio.Mesh(m.points, [("tetra", m.tets)], point_data={"coord_y_norm": coord_y_norm}))
  ```

---

## Troubleshooting

**“Only triangles/no tets.”**
Your input was not watertight or volume reconstruction failed. Use `from_bbox` or repair the CAD/mesh (close holes), or export STEP/IGES.

**“3D meshing failed (HXT).”**
The wrapper retries with alternative 3D algorithms (10 → 4 → 1). You can also embed more interior points (`embed_grid_h`, `embed_points_mm`) and/or relax `h`.

**“All-blue coloring in ParaView.”**
Hit **Rescale to Data Range**. If still flat, compute `mag(u)` or export normalized fields (e.g., `u_mag_norm`).

**CG didn’t converge.**
Try `solver="direct"`, or install `pyamg` for a preconditioner, or use a coarser mesh.

---

## Design Notes

* Meshing is **mm-based**, but the FEM class always works in **SI**. The conversion happens once in the constructor.
* Surface tractions are assembled as exact integrals on the selected boundary facets; “point loads” are implemented as **distributed** tractions over a small patch.

---

## License

See `LICENSE`.

---

[gmsh]: https://gmsh.info
[scikit-fem]: https://scikit-fem.readthedocs.io
[meshio]: https://github.com/nschloe/meshio
