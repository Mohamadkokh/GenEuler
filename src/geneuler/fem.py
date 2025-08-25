# src/geneuler/fem.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, List, Union, Callable

import numpy as np
from scipy.sparse.linalg import spsolve, cg as cg_solve
import meshio

from skfem import (
    MeshTet, ElementTetP1, ElementVector, Basis, FacetBasis,
    asm, condense, BilinearForm, LinearForm
)
from skfem.helpers import ddot, sym_grad, div, dot

# Optional AMG (recommended for iterative solver)
try:
    import pyamg
    _HAS_PYAMG = True
except Exception:
    _HAS_PYAMG = False

from .mesh import Mesh as GE_Mesh  # your Mesh class (mm units)

# -----------------------------
# Helpers for region selection
# -----------------------------
def _facets_on_plane(m: MeshTet, axis: int, value_m: float, tol: float = 1e-9) -> np.ndarray:
    return m.facets_satisfying(lambda x: np.isclose(x[axis], value_m, atol=tol))

def _facets_in_sphere(m: MeshTet, center_m: Sequence[float], radius_m: float) -> np.ndarray:
    c = np.array(center_m, float).reshape(3, 1)
    return m.facets_satisfying(lambda x: np.linalg.norm(x - c, axis=0) <= radius_m + 1e-12)

def _facets_in_box(m: MeshTet, lo_m: Sequence[float], hi_m: Sequence[float]) -> np.ndarray:
    lo = np.array(lo_m, float).reshape(3, 1)
    hi = np.array(hi_m, float).reshape(3, 1)
    return m.facets_satisfying(lambda x: np.all((x >= lo) & (x <= hi), axis=0))

def _facet_areas(mesh: MeshTet, facets_idx: np.ndarray) -> np.ndarray:
    """Areas of triangular boundary facets given their indices."""
    tri = mesh.facets[:, facets_idx]               # (3, k) vertex ids
    P = mesh.p                                     # (3, n_nodes)
    p0 = P[:, tri[0]]; p1 = P[:, tri[1]]; p2 = P[:, tri[2]]
    v1 = (p1 - p0).T; v2 = (p2 - p0).T             # (k, 3)
    cr = np.cross(v1, v2)                          # (k, 3)
    return 0.5 * np.linalg.norm(cr, axis=1)        # (k,)

def _point_to_tri_distances(mesh: MeshTet, facets_idx: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Vectorized distance from point P (3,) to each triangle facet in facets_idx.
    Returns array (k,) of distances in meters.
    """
    tri = mesh.facets[:, facets_idx]                 # (3, k)
    A = mesh.p[:, tri[0]].T                          # (k, 3)
    B = mesh.p[:, tri[1]].T
    C = mesh.p[:, tri[2]].T
    P = np.asarray(P, float).reshape(1, 3)           # (1, 3), broadcast later

    AB = B - A; AC = C - A
    N = np.cross(AB, AC)                             # (k, 3)
    nn2 = np.einsum('ij,ij->i', N, N)                # (k,)
    nn = np.sqrt(np.maximum(nn2, 1e-30))
    AP = P - A                                       # (k, 3)
    # signed distance to plane
    s = np.einsum('ij,ij->i', AP, N) / nn2           # (k,)
    Q = P - s[:, None] * N                           # projection on plane, (k, 3)
    # inside test using same-side method
    # edge AB
    c1 = np.cross(B - A, Q - A)
    # edge BC
    c2 = np.cross(C - B, Q - B)
    # edge CA
    c3 = np.cross(A - C, Q - C)
    s1 = np.einsum('ij,ij->i', c1, N)
    s2 = np.einsum('ij,ij->i', c2, N)
    s3 = np.einsum('ij,ij->i', c3, N)
    eps = 1e-14
    inside = ((s1 >= -eps) & (s2 >= -eps) & (s3 >= -eps)) | ((s1 <= eps) & (s2 <= eps) & (s3 <= eps))
    d_plane = np.abs(np.einsum('ij,ij->i', AP, N)) / nn    # (k,)

    # distance to segments if outside
    def seg_dist(P, E0, E1):
        V = E1 - E0
        W = P - E0
        VV = np.einsum('ij,ij->i', V, V)
        t = np.einsum('ij,ij->i', W, V) / np.maximum(VV, 1e-30)
        t = np.clip(t, 0.0, 1.0)
        proj = E0 + t[:, None] * V
        return np.linalg.norm(P - proj, axis=1)

    d_ab = seg_dist(P, A, B)
    d_bc = seg_dist(P, B, C)
    d_ca = seg_dist(P, C, A)
    d_edge = np.minimum(d_ab, np.minimum(d_bc, d_ca))

    return np.where(inside, d_plane, d_edge)


# -----------------------------
# Main FEM class
# -----------------------------
@dataclass
class SkFemElasticity:
    """
    Linear elasticity (small strain) on tetrahedral meshes using scikit-fem.

    Units:
      - Input `geneuler.Mesh` is in millimeters; geometry is converted to meters.
      - Forces in Newtons, density in kg/m^3, gravity in m/s^2, E in Pascals.

    DOF space:
      - Vector P1 on tets (3 dofs per node).
    """
    gmesh: GE_Mesh

    # material
    E: float = 210e9           # Pa
    nu: float = 0.30
    rho: float = 7850.0        # kg/m^3

    # gravity (defaults to -y)
    gravity: Sequence[float] = (0.0, -9.81, 0.0)

    # internal
    _m: MeshTet = field(init=False, repr=False)
    _basis: Basis = field(init=False, repr=False)

    # BCs & loads storage
    _fixed_facets: List[np.ndarray] = field(default_factory=list, repr=False)

    # list of (facets, traction_vector_in_Pa)
    _traction_terms: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)

    _body_force_vec: Optional[np.ndarray] = field(default=None, repr=False)  # ρ g (N/m^3)

    def __post_init__(self):
        # Convert your Mesh (mm) -> MeshTet (m)
        pts_m = (self.gmesh.points * 1e-3).astype(float)   # (N,3) -> meters
        tets = np.asarray(self.gmesh.tets, dtype=np.int64) # (M,4)
        if tets.size == 0:
            raise ValueError("FEM requires a volumetric mesh (tetrahedra).")

        self._m = MeshTet(pts_m.T, tets.T)                 # MeshTet wants (3,N), (4,M)

        # Vector P1 space
        e = ElementVector(ElementTetP1())
        self._basis = Basis(self._m, e)

        # (optional) alias for convenience
        self.mesh = self._m

        # Default body force from gravity
        self.set_gravity(self.gravity)

    # -----------------------------
    # Material & gravity
    # -----------------------------
    def set_material(self, E: float, nu: float, rho: Optional[float] = None):
        self.E = float(E)
        self.nu = float(nu)
        if rho is not None:
            self.rho = float(rho)

    def set_gravity(self, g_xyz: Sequence[float] | float):
        """Set gravity vector; accepts (gx, gy, gz) or scalar -> (0, -scalar, 0)."""
        if isinstance(g_xyz, (int, float)):
            g = np.array([0.0, -float(g_xyz), 0.0])
        else:
            g = np.array(g_xyz, float).reshape(3)
        self.gravity = g
        self._body_force_vec = self.rho * g  # N/m^3

    # -----------------------------
    # Dirichlet BCs (fixed regions)
    # -----------------------------
    def fix_floor(self, y_mm: float = 0.0, tol_mm: float = 1e-6):
        """Fix all displacement components on plane y = y_mm; fallback to min(y)."""
        y_m = y_mm * 1e-3
        tol_m = tol_mm * 1e-3
        facets = _facets_on_plane(self._m, axis=1, value_m=y_m, tol=tol_m)
        if facets.size == 0:
            ymin = np.min(self._m.p[1])
            facets = _facets_on_plane(self._m, axis=1, value_m=ymin, tol=1e-12)
        self._fixed_facets.append(facets)

    def fix_facets_in_box(self, lo_mm: Sequence[float], hi_mm: Sequence[float]):
        """Fix all DOFs on facets whose centroid lies in the given (mm) AABB."""
        facets = _facets_in_box(self._m, np.array(lo_mm) * 1e-3, np.array(hi_mm) * 1e-3)
        self._fixed_facets.append(facets)

    def fix_facets_selector(self, selector: Callable[[np.ndarray], np.ndarray]):
        """Fix facets via custom selector on facet centroids (meters)."""
        facets = self._m.facets_satisfying(selector)
        self._fixed_facets.append(facets)

    # -----------------------------
    # Loads: traction & distributed forces
    # -----------------------------
    def add_point_load(self, *, center_mm=(0.0, 130.0, -110.0),
                       force_N=(0.0, -800.0, 0.0), radius_mm=6.0,
                       fallback_k: int = 20):
        """
        Distribute a total force `force_N` as a constant traction over the set of
        boundary facets whose triangle-to-point distance ≤ `radius_mm`.
        If none are within `radius_mm`, use the `fallback_k` nearest facets.
        """
        F = np.asarray(force_N, dtype=float)
        if not np.any(F):
            return

        center_m = np.asarray(center_mm, float) * 1e-3
        radius_m = float(radius_mm) * 1e-3

        bfaces = self._m.boundary_facets()               # (nboundary,)
        if bfaces.size == 0:
            raise ValueError("Mesh has no boundary facets.")

        # true point-to-triangle distances
        dists = _point_to_tri_distances(self._m, bfaces, center_m)  # (nb,)
        pick = np.where(dists <= radius_m)[0]

        if pick.size == 0:
            # fallback: take the k nearest facets
            k = min(fallback_k, bfaces.size)
            idx_sorted = np.argsort(dists)[:k]
            picked_facets = np.asarray(bfaces)[idx_sorted]
        else:
            picked_facets = np.asarray(bfaces)[pick]

        # total area and traction
        areas = _facet_areas(self._m, picked_facets)     # (k,)
        A = float(np.sum(areas))
        if A <= 0.0:
            raise ValueError("Selected facets have zero total area; cannot distribute load.")

        t = F / A                                        # traction (Pa)
        self._traction_terms.append((picked_facets, t))


    # -----------------------------
    # Assemble & solve
    # -----------------------------
    def solve(self, solver: str = "direct", cg_tol: float = 1e-8, cg_maxit: int = 5000
              ) -> Dict[str, np.ndarray]:
        """
        Assemble and solve Ku=f. Returns:
          - "u": global DOF vector (size = 3 * n_nodes)
          - "compliance": u^T f
        """
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.E / (2.0 * (1.0 + self.nu))

        @BilinearForm
        def a(u, v, w):
            return 2.0 * mu * ddot(sym_grad(u), sym_grad(v)) + lam * div(u) * div(v)

        gvec = np.array(self._body_force_vec if self._body_force_vec is not None else (0.0, 0.0, 0.0))

        @LinearForm
        def l_body(v, w):
            return dot(gvec, v)

        K = asm(a, self._basis)
        f = asm(l_body, self._basis)

        # Surface tractions: sum over all stored (facets, traction)
        for facets, traction_Pa in self._traction_terms:
            tvec = np.array(traction_Pa, float)

            @LinearForm
            def l_trac(v, w):
                return dot(tvec, v)

            fb = FacetBasis(self._m, self._basis.elem, facets=facets)
            f += asm(l_trac, fb)

        # Essential BCs (fix all components on selected facets)
        fixed_dofs = None
        for facets in self._fixed_facets:
            D = self._basis.get_dofs(facets=facets)          # a DofsView / DofsCollection
            fixed_dofs = D if fixed_dofs is None else (fixed_dofs | D)

        if fixed_dofs is not None:
            fixed_idx = fixed_dofs.all()                      # -> np.ndarray of constrained dof indices
            Kc, fc, x, I = condense(K, f, D=fixed_idx)        # <-- pass via keyword D=...
        else:
            Kc, fc, x, I = K, f, np.zeros(K.shape[0]), None

        # Solve
        if solver == "direct":
            u_free = spsolve(Kc, fc)
        elif solver == "cg":
            M = None
            if _HAS_PYAMG:
                M = pyamg.smoothed_aggregation_solver(Kc.tocsr()).aspreconditioner()
            u_free, info = cg_solve(Kc, fc, tol=cg_tol, maxiter=cg_maxit, M=M)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info}).")
        else:
            raise ValueError("solver must be 'direct' or 'cg'.")

        u = x + (I @ u_free if I is not None else u_free)
        compliance = float(u @ f)
        return {"u": u, "compliance": compliance}

    # -----------------------------
    # Export
    # -----------------------------
    def export_vtu(self, path: str, u: np.ndarray, field_name: str = "u"):
        """Export displacement to .vtu using meshio."""
        nnode = self._m.p.shape[1]
        U = u.reshape(nnode, 3, order="C")
        points = self._m.p.T  # meters
        cells = [("tetra", self._m.t.T.astype(np.int32))]
        m = meshio.Mesh(points, cells, point_data={field_name: U})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meshio.write(path, m)

    def export_xdmf(self, path_without_ext: str, u: np.ndarray, field_name: str = "u"):
        """Export displacement to .xdmf (+ .h5 sidecar when applicable)."""
        xdmf = f"{path_without_ext}.xdmf"
        nnode = self._m.p.shape[1]
        U = u.reshape(nnode, 3, order="C")
        points = self._m.p.T
        cells = [("tetra", self._m.t.T.astype(np.int32))]
        m = meshio.Mesh(points, cells, point_data={field_name: U})
        os.makedirs(os.path.dirname(xdmf) or ".", exist_ok=True)
        meshio.write(xdmf, m)
