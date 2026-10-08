#!/usr/bin/env python3
"""Prepare the pinned demo environments and a local macOS launcher."""
import argparse
from pathlib import Path
import json
import os
import plistlib
import shlex
import subprocess
import sys
from _demo_paths import REPO_ROOT, runtime_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-tss', action='store_true', help='Create/install the pinned TSS runtime first using Python 3.12; no weights downloaded.')
    parser.add_argument('--verify-only', action='store_true', help='Verify installed environments without installation or network access.')
    args = parser.parse_args()
    if args.setup_tss and args.verify_only:
        parser.error('--setup-tss and --verify-only cannot be combined')
    root = runtime_root()
    child_env = os.environ.copy()
    child_env.setdefault('MPLCONFIGDIR', str(root / 'var/spinosarc/matplotlib'))
    for name, directory in (('nnUNet_raw', 'raw'), ('nnUNet_preprocessed', 'preprocessed'), ('nnUNet_results', 'results')):
        child_env.setdefault(name, str(root / 'var/totalspineseg/nnUNet' / directory))
    provider = root / '.venv-tss/bin/python'
    gui = root / '.venv-spinosarc/bin/python'
    if args.setup_tss:
        if not provider.is_file():
            if sys.version_info[:2] != (3, 12):
                raise SystemExit('Create this tested runtime using Python 3.12: python3.12 demo/scripts/setup_spinosarc.py --setup-tss')
            root.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([sys.executable, '-m', 'venv', str(provider.parent.parent)])
        subprocess.check_call([str(provider), '-m', 'pip', 'install', '-r', str(REPO_ROOT / 'demo/requirements-tss.lock')])
        subprocess.check_call([str(provider), '-c', 'from auglab.add_trainer import add_trainer; add_trainer("nnUNetTrainerDAExt")'], env=child_env)
    if not provider.is_file():
        raise SystemExit('TSS runtime is missing. Use Python 3.12 and --setup-tss; see demo/README.md.')
    probe = 'import json,sys,sysconfig; print(json.dumps([list(sys.version_info[:2]),sysconfig.get_paths()["purelib"]]))'
    provider_version, provider_site = json.loads(subprocess.check_output([str(provider), '-c', probe], text=True))
    if not gui.is_file():
        if args.verify_only:
            raise SystemExit('GUI environment is missing.')
        subprocess.check_call([str(provider), '-m', 'venv', str(gui.parent.parent)])
    gui_version, gui_site = json.loads(subprocess.check_output([str(gui), '-c', probe], text=True))
    if gui_version != provider_version:
        raise SystemExit('GUI and TSS must use the same Python major/minor version.')
    pth = Path(gui_site) / 'spinosarc_tss_runtime.pth'
    if args.verify_only:
        if not pth.is_file() or pth.read_text().strip() != provider_site:
            raise SystemExit('GUI runtime link is missing or points to another provider; rerun setup after moving directories.')
    else:
        pth.write_text(provider_site + '\n')
        subprocess.check_call([str(gui), '-m', 'pip', 'install', '--no-deps', '-r', str(REPO_ROOT / 'demo/requirements-spinosarc.lock')])
    subprocess.check_call([str(gui), '-c', 'import PyQt6.QtWidgets, nibabel, SimpleITK, scipy, pydicom, reportlab, openpyxl, monai'], env=child_env)
    subprocess.check_call([str(provider), '-c', 'from importlib.metadata import version; assert version("totalspineseg") == "20260730"; assert version("torch") == "2.5.1"; from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDAExt import nnUNetTrainerDAExtGPU'], env=child_env)
    if not args.verify_only and sys.platform == 'darwin':
        bundle = root / 'SpinoSarc Demo.app/Contents'
        (bundle / 'MacOS').mkdir(parents=True, exist_ok=True)
        info = dict(CFBundleDisplayName='SpinoSarc Demo', CFBundleName='SpinoSarc Demo',
                    CFBundleExecutable='SpinoSarc', CFBundleIdentifier='org.lumbar-mri.spinosarc-demo',
                    CFBundlePackageType='APPL', CFBundleVersion='0.2.0',
                    LSUIElement=False, NSHighResolutionCapable=True)
        with (bundle / 'Info.plist').open('wb') as stream:
            plistlib.dump(info, stream)
        launcher = bundle / 'MacOS/SpinoSarc'
        launcher.write_text('#!/bin/sh\nexec env ' + shlex.quote('SPINOSARC_RUNTIME_ROOT=' + str(root)) + ' ' +
                            shlex.quote(str(gui)) + ' ' + shlex.quote(str(REPO_ROOT / 'demo/scripts/run_spinosarc.py')) + ' --example\n')
        launcher.chmod(0o755)
    print('SpinoSarc GUI and TSS runtimes OK; weights and MRI are managed separately.')


if __name__ == '__main__':
    main()
