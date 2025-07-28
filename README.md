# GenEuler

```markdown
<!--
──────────────────────────────────────────────────────────────────────────────
 _/_/_/    _/_/    _/_/_/   _/_/_/      _/_/   _/_/_/   _/_/_/    _/_/_/
  _/    _/    _/  _/    _/   _/    _/  _/    _/        _/    _/  _/
 _/    _/    _/  _/_/_/     _/_/_/    _/    _/  _/_/  _/_/_/    _/_/_/
_/    _/    _/  _/         _/    _/  _/    _/    _/  _/    _/  _/
 _/_/_/  _/_/    _/        _/_/_/      _/_/   _/_/_/  _/    _/  _/_/_/

GenEuler · Automated 3-D Mesh Optimisation for Prosthetic Design
──────────────────────────────────────────────────────────────────────────────
-->

<div align="center">

# 🧬 GenEuler

**An end-to-end pipeline that combines high-fidelity FEM (DOLFINx), density-based  
topology optimisation (SIMP) and a mesh-aware Soft-Actor-Critic Graph Neural  
Network (SAC-GNN) to automate patient-specific prosthetic design.**

</div>

<p align="center">
  <a href="https://github.com/your-org/GenEuler/actions"><img
    src="https://github.com/your-org/GenEuler/workflows/CI/badge.svg" alt="CI status"></a>
  <a href="https://zenodo.org/doi/10.5281/zenodo.•••"><img
    src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.•••-blue.svg" alt="DOI"></a>
  <a href="LICENSE"><img
    src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## ✨ Key features

| ✔                                     | Description |
|---------------------------------------|-------------|
| **Full-physics supervision**          | Solves a 3-D elasticity problem with DOLFINx + PETSc/GAMG **every RL step**. |
| **Mesh-aware RL**                     | SAC agent encodes the *tetrahedral* mesh as a PyTorch-Geometric graph and proposes node-wise add/remove edits. |
| **SIMP integration**                  | Classic density-based topology optimisation loop with volume-fraction monitoring. |
| **Pluggable meshing service**         | Gmsh-based generator + JSON diffing utilities to track local edits and mesh health. |
| **Rich telemetry**                    | Logs compliance, volume, reward, mesh quality & conditioning to CSV/MLflow. |
| **One-command reproducibility**       | Docker + `poetry` lockfile; optional Singularity/Apptainer recipe for HPC. |

---

## 📂 Repository layout

```

GenEuler/
├── gen\_euler            # Core Python package
│   ├── fem/             # DOLFINx wrappers, BC helpers, compliance calc
│   ├── topo/            # SIMP update kernels & volume monitors
│   ├── rl/              # SAC-GNN actor/critic, replay buffer, config
│   ├── mesh/            # Gmsh generator, JSON diff utils, quality metrics
│   └── cli.py           # `gen-euler` command-line entry point
├── examples/            # End-to-end scripts & Jupyter notebooks
├── docker/              # Dockerfile, docker-compose.yml, HPC recipe
├── tests/               # PyTest suite (CPU-only mocks)
├── docs/                # Sphinx sources, API reference, figures
├── LICENSE
└── README.md

````

---

## 🚀 Quick start

> **Prerequisites:** Linux/macOS, Python ≥ 3.10, a C++17 compiler, and working
> MPI. For GPU training, install the matching CUDA toolkit.

```bash
# 1. Clone
git clone https://github.com/your-org/GenEuler.git && cd GenEuler

# 2. Create environment (CPU example)
curl -sSL https://install.python-poetry.org | python -
poetry install --with dev

# 3. Run the minimal demo (takes ≈ 5 min on laptop CPU)
poetry run gen-euler run examples/minimal.yaml
````

**Docker instead?**

```bash
docker build -t geneuler .
docker run --gpus all -v $PWD:/workspace geneuler \
       gen-euler run examples/gpu_foot_prosthesis.yaml
```

The first run generates:

* `outputs/meshes/*.xdmf` – meshes per iteration
* `outputs/metrics.csv`   – compliance, reward, volume, quality
* `outputs/checkpoints/*.pt` – SAC agent weights

Open the notebook `examples/visualise.ipynb` to explore the optimisation trail.

---

## 🏗  Architecture overview

```mermaid
flowchart LR
  A[Current mesh (Mₖ)] -->|Graph encode| B(SAC-GNN Actor)
  B -->|Add/remove proposals| C{Mesh service}
  C -->|Edited mesh (Mₖ₊₁)| D[DOLFINx solve]
  C -->|Density update| E[SIMP kernel]
  D -->|Displacements| E
  E -->|ρ update| C
  D -->|Compliance, stresses| F[Reward composer]
  F -->|Rₖ ∈ [-1,1]| G[Replay buffer]
  G -->|Sample batch| H{SAC update}
  H -->|θ←…| B
```

---

## ⚙️  Configuration

All experiment settings live in a single YAML:

```yaml
env:
  material: steel
  vf_target: 0.50
rl:
  algo: SAC
  actor_lr: 1e-5
  critic_lr: 1e-5
  entropy_coeff: 0.2
  max_epochs: 100
mesh:
  h_global: 4.0
  h_attach: 0.8
```

Run with

```bash
gen-euler run configs/my_experiment.yaml
```

---

## 📊  Reproducing the thesis results

```bash
poetry run gen-euler run configs/thesis_foot_C2.yaml
poetry run python scripts/make_plots.py outputs/thesis_foot_C2
```

Figures and tables from the MSc thesis will appear in `outputs/thesis_foot_C2/plots/`.

---

## 🤝  Contributing

1. Fork → feature branch → PR.
2. Follow `pre-commit` hooks (`black`, `ruff`, `isort`, `mypy`).
3. Add a unit test in `tests/` and, if relevant, a minimal YAML example.
4. For major features, open a discussion first!

---

## 📜  License

> MIT – see [`LICENSE`](LICENSE).
> If you use GenEuler in academic work, please cite:

```bibtex
@misc{kokh2025geneuler,
  title   = {GenEuler: Reinforcement Learning Meets Finite-Element Topology Optimisation},
  author  = {Kokh, Mohamad and Lessmann, Stefan},
  year    = {2025},
  note    = {MSc thesis, ESMT Berlin},
  url     = {https://github.com/your-org/GenEuler}
}
```

---

## 🙋‍♂️  Contact

*Issues, questions, or want to collaborate?*
Open an issue or ping **@mohamad-kokh** on GitHub.

Happy meshing! 🦾

```

*Pro-tips*

* Replace `your-org` with the real GitHub namespace.  
* Add or remove badges (e.g. Codecov) as soon as the integrations are live.  
* Keep the **Quick start** section ruthlessly short; move advanced options to `docs/`.
::contentReference[oaicite:0]{index=0}
```
