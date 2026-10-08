"""Portable source/runtime locations for the research desktop demo."""
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[2]


def runtime_root():
    """Keep models, MRI, environments and generated results outside source if requested."""
    return Path(os.environ.get('SPINOSARC_RUNTIME_ROOT', REPO_ROOT)).expanduser().resolve()
