# SpinoSarc development workspace

This is the independent `ndreuu/SpinoSarc` fork, not the parent lumbar MRI
HTTP backend. Start with `docs/workspace_ru.md` and `docs/architecture_ru.md`.

## Locations

- `spinosarc_app/`: application, imaging, geometry, model providers and measurements.
- `demo/scripts/`: setup, downloaders, launch and reproducible checks.
- `demo/tests/`: tool tests; pinned dependencies are `demo/requirements-*.lock`.
- `docs/`: documentation of current application behavior and development.
- `research/`: plans, tool comparisons, dataset notes and explicitly marked history.
- `paper/`, citation, license and original build files: retained upstream material.

Keep research Markdown and experiments' written conclusions in `research/`.
Implementation belongs in application/tools paths. A research proposal is not
an implemented capability; describe evidence, hypotheses and missing assessments
separately. Existing backend API/ROI research snapshots refer to another project.

## Runtime and verification

Runtime selection is `SPINOSARC_RUNTIME_ROOT`, then ignored
`.spinosarc.local.json`, then this repository. Never commit machine-local config,
MRI, models, environments or generated analyses. JSON configuration is data;
do not execute it as a shell script. See `demo/runtime.example.json`.

From repository root:

```sh
python3 demo/scripts/setup_spinosarc.py --verify-only
python3 demo/scripts/run_spinosarc.py --example
```

For geometry/UI state changes, use `demo/scripts/smoke_spinosarc.py` with the
selected runtime's `.venv-spinosarc/bin/python`; it does not run full inference.
Downloader tests are in `demo/tests/test_fetch_totalspineseg.py`.
Rerun real inference when the affected behavior requires it, not for notes.
Keep frame identity, physical geometry, coverage, numbering/alignment status and
source-linked output intact. Models have not been retrained in this fork.
