# Lumbar MRI demo fork

Repository: [ndreuu/SpinoSarc](https://github.com/ndreuu/SpinoSarc).
Upstream: [neuromath/SpinoSarc](https://github.com/neuromath/SpinoSarc).
Base revision: `63b7405d1276740dfde75e00d1e7ad58da6c10fd`.
Demo implementation branch: `codex/lumbar-mri-demo`.
Standalone workspace branch: `codex/spinosarc-workspace`.

`main` preserves the upstream baseline. The demo changes live on the development
branch; the upstream license, citation, paper and original build files are retained.
The original README below the fork notice describes the upstream application.
Use [demo/README.md](demo/README.md) for this branch's actual capabilities and setup.
Use [docs/workspace_ru.md](docs/workspace_ru.md) for standalone development and
[research/README.md](research/README.md) for research material. Research snapshots
from the original backend project are clearly marked and do not imply its API
is part of this repository.

## Architectural role

This fork implements a local research viewer and measurement pipeline:
native MRI ingestion → physical geometry → TotalSpineSeg anatomical labels and
canal mask → MuscleMap muscle masks → native-plane areas and left/right asymmetry
→ source-linked JSON, masks and annotated images.

It covers the anatomy, geometry, measurement and review components of the lumbar
MRI project. It does not replace that project's HTTP backend. Herniation contours
and dimensions, root compression, validated stenosis grades, conus pathology and
radiology report generation require separate diagnostic components.

## Changes relative to upstream

| Files | Change |
|---|---|
| `spinosarc_app/lumbar_demo.py` | Dedicated demo entry point; anatomy and muscle workflows, native overlays, cache, JSON/PNG export, scrollable controls on laptop displays |
| `spinosarc_app/demo_io.py` | MHA/NIfTI and classic MR DICOM; source-frame identity and exact native-plane geometry |
| `spinosarc_app/totalspineseg/level_mapper.py` | RAS/LPS-aware point-to-plane matching; field-of-view and coverage checks |
| `spinosarc_app/totalspineseg/runner.py` | Local pinned runtime, cancellable owned processes, explicit cleanup recovery |
| `spinosarc_app/analyzer.py`, `inference_engine.py` | Optional MuscleMap, official asset verification, configuration-derived labels, CPU inversion before MPS postprocessing |
| `spinosarc_app/gui.py` | Physical image aspect ratio support |
| `spinosarc_app/totalspineseg/multi_level_analyzer.py` | Canal-only measurement path without nearest-slice substitution or diagnostic threshold classification |
| `demo/` | Reproducible environments, model/data acquisition, runtime adapters, checks and instructions |

No model was retrained. Official checkpoints are downloaded separately and verified.
The MPS adapter retains original TSS plans and patch context; the optional CPU demo
adapter reduces patch context and is explicitly marked experimental.

## Working with Git

```sh
git remote -v
git switch codex/spinosarc-workspace
git diff main...HEAD --stat
git log main..HEAD --oneline
```

`origin` points to `ndreuu/SpinoSarc`; `upstream` points to `neuromath/SpinoSarc`.
Before incorporating future upstream changes, fetch them and inspect the diff:

```sh
git fetch upstream
git diff main..upstream/main --stat
```

Merge selected upstream updates on a separate working branch, run the checks in
`demo/README.md`, then update the main project's pinned submodule commit. A local
branch name is for development; the consuming project pins an exact commit.

Runtime environments, DICOM, weights, model manifests and generated results are
excluded from version control. The public example is acquired from its original
source with attribution; the fork contains the downloader and instructions.

An ignored `.spinosarc.local.json` can select an existing runtime without copying
environments or models. `SPINOSARC_RUNTIME_ROOT` overrides it; with neither,
runtime files live in this checkout. The application comes from the current
checkout; adapters default to it, with explicit provider overrides preserved.
See `demo/runtime.example.json` for the config format.
