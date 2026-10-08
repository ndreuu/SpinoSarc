"""Portable source/runtime locations for the research desktop demo."""
from pathlib import Path
import json
import os


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO_ROOT / '.spinosarc.local.json'


def runtime_root():
    """Keep models, MRI, environments and generated results outside source if requested."""
    configured = os.environ.get('SPINOSARC_RUNTIME_ROOT')
    if configured is not None:
        return Path(configured).expanduser().resolve()
    if not LOCAL_CONFIG.is_file():
        return REPO_ROOT
    try:
        config = json.loads(LOCAL_CONFIG.read_text(encoding='utf-8'))
        configured = config['runtime_root']
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError('runtime_root must be a nonempty path string')
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Invalid local runtime configuration {LOCAL_CONFIG}: {error}') from error
