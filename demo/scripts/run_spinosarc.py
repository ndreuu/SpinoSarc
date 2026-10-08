#!/usr/bin/env python3
"""Launch the portable SpinoSarc lumbar demo with pinned local model runtimes."""
import argparse
from pathlib import Path
import os
import sys
from _demo_paths import REPO_ROOT, runtime_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', action='store_true', help='Load the published example and matching saved results.')
    args = parser.parse_args()
    root = runtime_root()
    python = Path(os.environ.get('SPINOSARC_GUI_PYTHON', root / '.venv-spinosarc/bin/python'))
    if not python.is_file():
        raise SystemExit('GUI environment is missing; see demo/README.md and run demo/scripts/setup_spinosarc.py.')
    env = os.environ.copy()
    defaults = dict(
        SPINE_TSS_COMMAND=str(root / '.venv-tss/bin/totalspineseg'),
        SPINE_TSS_PYTHON=str(root / '.venv-tss/bin/python'),
        SPINE_TSS_DATA_DIR=str(root / 'var/totalspineseg'),
        SPINE_TSS_DEVICE=env.get('SPINOSARC_DEVICE', 'mps' if sys.platform == 'darwin' else 'cpu'),
        SPINE_TSS_TIMEOUT_SECONDS='1800',
        SPINE_TSS_MPS_WRAPPER=str(REPO_ROOT / 'demo/scripts/totalspineseg_mps.py'),
        SPINE_TSS_CPU_DEMO_WRAPPER=str(REPO_ROOT / 'demo/scripts/totalspineseg_cpu_demo.py'),
        nnUNet_def_n_proc='4', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        PYTORCH_ENABLE_MPS_FALLBACK='1', SPINOSARC_WORK_DIR=str(root / 'var/spinosarc'),
        MPLCONFIGDIR=str(root / 'var/spinosarc/matplotlib'),
        SPINOSARC_DEMO_MRI=str(root / 'data/spider/images/246_t2.mha'),
        SPINOSARC_AXIAL_EXAMPLE_MANIFEST=str(root / 'data/spinosarc-example/manifest.json'),
        SPINOSARC_MUSCLEMAP_PATH=str(root / 'var/spinosarc/musclemap/source/scripts'),
        SPINOSARC_MUSCLEMAP_MANIFEST=str(root / 'var/spinosarc/musclemap/manifest.json'),
        SPINOSARC_MUSCLEMAP_MODEL_VERSION='0.0',
    )
    for key, value in defaults.items():
        env.setdefault(key, value)
    env.setdefault('SPINOSARC_ENABLE_MUSCLES', '1' if Path(env['SPINOSARC_MUSCLEMAP_MANIFEST']).is_file() else '0')
    env['SPINOSARC_RUNTIME_ROOT'] = str(root)
    env['PYTHONPATH'] = str(REPO_ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    arguments = ['--example'] if args.example else []
    os.execve(str(python), [str(python), '-m', 'spinosarc_app.lumbar_demo', *arguments], env)


if __name__ == '__main__':
    main()
