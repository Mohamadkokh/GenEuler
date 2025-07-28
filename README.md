# GenEuler: Automated 3D Mesh Optimization with FEM + SIMP + SAC‑GNN

GenEuler is an end‑to‑end research pipeline that couples high‑fidelity Finite Element Analysis (FEM) with density‑based topology optimization (SIMP) and a Soft Actor–Critic Graph Neural Network (SAC‑GNN) to automate local mesh edits for prosthetic‑grade structures.

It is designed to be reproducible out‑of‑the‑box with Docker and dead‑simple to run.

---

## Table of Contents

* [Quick Start](#quick-start)
* [What You Get](#what-you-get)
* [Repo Layout](#repo-layout)
* [How It Works](#how-it-works)
* [Running the Pipeline](#running-the-pipeline)
* [Results & Artifacts](#results--artifacts)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Development Tips](#development-tips)
* [Citing](#citing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Quick Start

**One command to build and start all services (detached):**

```bash
docker-compose up --build -d
```

**Run the full pipeline:**

```bash
cd scripts
python pipeline.py
```

**Where are the outputs?**
All results and figures are written under `scripts/` (see [Results & Artifacts](#results--artifacts)).

**Stop everything when you’re done:**

```bash
docker-compose down
```

---

## What You Get

* End‑to‑end automation for **FEM → SIMP → RL mesh editing**
* **Deterministic builds** via Docker
* **Reproducible artifacts** (meshes, CSV metrics, figures)
* **Human‑readable logs** for every optimization step

---

## Repo Layout

> The pipeline is tolerant to missing data and will generate baseline meshes from code. Exact folders may vary slightly; the ones below are the important touch points for users.

```
GenEuler/
├─ docker/                 # Dockerfiles, build context (if present)
├─ scripts/                # Entry points, pipeline, and where results are written
│  ├─ pipeline.py          # Main orchestrator (run this)
│  ├─ *.py                 # Modules for FEM, SIMP, RL, logging, plotting
│  ├─ results*/            # (Created on run) CSVs, images, XDMF/HDF5, JSON meshes
│  └─ logs*/               # (Created on run) pipeline logs
├─ docker-compose.yml
├─ requirements.txt        # (Optional) For local dev outside Docker
└─ README.md               # (This file)
```

---

## How It Works

**High‑level loop (one optimization step):**

```text
[FEM mesh & state]
        ↓
  FEM solve (DOLFINx/PETSc)  →  compliance, stresses
        ↓
  SIMP update (density field, Vf target monitored)
        ↓
  SAC‑GNN policy (PyTorch Geometric) proposes local mesh edits
        ↓
  Mesh update + logging
        ↺ iterate
```

Artifacts are produced at each step so you can inspect progress and diagnose behavior.

---

## Running the Pipeline

**Build & start services (detached):**

```bash
docker-compose up --build -d
```

**Launch the pipeline:**

```bash
cd scripts
python pipeline.py
```

**Follow logs (optional):**

```bash
tail -f logs/pipeline.log
```

> If `logs/` doesn’t exist yet, it will be created at first run.

**Shut down when finished:**

```bash
docker-compose down
```

---

## Results & Artifacts

All outputs are written under `scripts/`. You can expect to see:

### Metrics (CSV)

* `compliance_summary.csv` — iteration, compliance `C(M_k)`, norms
* `reward_summary.csv` — per‑step rewards (FEM, Topology, combined)
* `mesh_size_summary.csv` — nodes, tets, deltas

### Meshes & Topology

* `convex_hull_{k}.json` — nodes + tetra connectivity per iteration
* `topology_density_{k}.(xdmf|h5)` — SIMP density fields

### Figures

* `Reward_combined.png`, `Reward_fem_to.png`
* `cluster_1-2_new.png`, `cluster_1-31_new.png`

### Logs

* `logs/pipeline.log` — orchestrator logs
* Additional per‑module logs if enabled

> **Tip:** If your run created a timestamped `results_YYYYmmdd_HHMMSS/` folder under `scripts/`, everything above will be inside it.

---

## Configuration

Most defaults are sensible for a first run. Common knobs:

* **SAC / RL** — learning rates, entropy coeff, discount factor, replay size
* **Topology (SIMP)** — penalty, density bounds, target volume fraction (monitored)
* **FEM** — material law, BCs, solver tolerances
* **Mesh editing** — add/remove thresholds, max radius, candidate counts

Configuration variables are typically defined in Python modules under `scripts/` (and/or read from environment variables). If you need to change them, edit the corresponding module and re‑run `pipeline.py`.

---

## Troubleshooting

**“Service is up but Python can’t be found”**
Make sure you’re inside the containerized environment or that the service exposes the `scripts/` folder. If you’re running locally, ensure your Python matches `requirements.txt`.

**“No module named …”**
The service may not have built correctly. Rebuild with:

```bash
docker-compose build --no-cache
docker-compose up -d
```

**“Permission denied” on mounted volumes (Linux/WSL2)**
Try setting more permissive mount options or run:

```bash
sudo chown -R $USER:$USER scripts
```

**Memory pressure / slow solves**
FEM + RL can be compute‑heavy. Close other apps or allocate more memory/cores to Docker.

---

## Development Tips

* **Iterate quickly:** after editing code in `scripts/`, you can usually re‑run `python pipeline.py` without rebuilding the container (depending on your compose bind mounts).
* **Clean state:** if you want a fresh run from scratch, remove the previous results directory under `scripts/` before re‑running.
* **Logs:** add `logger.debug(...)` statements liberally; they’ll appear in `logs/pipeline.log`.

---

## Citing

If you use GenEuler in academic work, please cite the associated thesis:

```bibtex
@thesis{Kokh2025GenEuler,
  title   = {GenEuler Pipeline for Automated 3D Mesh Optimization:
             Integrating FEM Simulation, Topology Optimization, and SAC-GNN Modeling},
  author  = {Mohamad Kokh},
  school  = {ESMT Berlin},
  year    = {2025}
}
```

---

## License

Specify your license here (e.g., MIT). Example:

```text
MIT License — see LICENSE file for details.
```

---

## Acknowledgments

* DOLFINx/FEniCSx, PETSc/SLEPc
* PyTorch + PyTorch Geometric
* LindheXtend for the clinical use case
* Advisors and colleagues who provided feedback

---
