#!/usr/bin/env python3
"""Experimental Metal adapter for local, pinned TotalSpineSeg weights.

Uses the upstream inference function and original plans/checkpoints; no downloads,
reduced tiles, or installed-package edits. Unsupported Metal operators use CPU.
Pre/postprocessing and nnU-Net's logit accumulation still use CPU as upstream does.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import metadata, version
import json
import os
from pathlib import Path
import sys


PACKAGE_VERSION = '20260730'
TORCH_VERSION = '2.5.1'
WEIGHTS_RELEASE = 'r20260730'
TORCH_THREADS = 4
MODEL_DIRECTORY = 'nnUNetTrainerDAExtGPU__nnUNetPlans__3d_fullres'
DATASETS = ('Dataset101_TotalSpineSeg_step1', 'Dataset102_TotalSpineSeg_step2')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=('mps',), default='mps')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--step1', action='store_true')
    parser.add_argument('--iso', action='store_true')
    parser.add_argument('--max-workers', type=positive_int, default=1)
    parser.add_argument('--max-workers-nnunet', type=positive_int, default=1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--keep-only', nargs='+', default=[''])
    return parser.parse_args(argv)


def validate_assets(data_dir):
    """Reject mismatched/shadowing weights before importing the ML runtime."""
    package = metadata('totalspineseg')
    if package['Version'] != PACKAGE_VERSION:
        raise ValueError(f'Requires TotalSpineSeg {PACKAGE_VERSION}; found {package["Version"]}')
    urls = dict(item.split(', ', 1) for item in package.get_all('Project-URL', [])
                if item.startswith('Dataset'))
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('provider') != 'totalspineseg'
            or manifest.get('package_version') != PACKAGE_VERSION
            or manifest.get('weights_release') != WEIGHTS_RELEASE):
        raise ValueError('The TotalSpineSeg manifest does not match the pinned runtime/release')
    archives = {item['dataset']: item['url'] for item in manifest.get('source_archives', [])}
    files = manifest.get('files', {})
    results = data_dir / 'nnUNet' / 'results'
    checked = []
    for dataset in DATASETS:
        expected_url = (f'https://github.com/neuropoly/totalspineseg/releases/download/'
                        f'{WEIGHTS_RELEASE}/{dataset}_{WEIGHTS_RELEASE}.zip')
        if urls.get(dataset) != expected_url or archives.get(dataset) != expected_url:
            raise ValueError(f'Unexpected official weight URL for {dataset}')
        if (results / dataset).exists():
            raise ValueError(f'Unversioned model directory would shadow pinned weights: {results / dataset}')
        base = Path('nnUNet/results') / WEIGHTS_RELEASE / dataset / MODEL_DIRECTORY
        for filename in ('plans.json', 'dataset.json', 'fold_0/checkpoint_best.pth'):
            relative = (base / filename).as_posix()
            path = data_dir / relative
            expected_hash = files.get(relative)
            if not isinstance(expected_hash, str) or not path.is_file():
                raise ValueError(f'Missing recorded model asset: {relative}')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected_hash:
                raise ValueError(f'Model asset checksum does not match manifest: {relative}')
            checked.append({'path': relative, 'sha256': digest})
    return checked


def install_conv_transpose3d_cpu_shim(torch_module):
    """Keep model parameters on MPS; evaluate this unsupported op on CPU.

    Torch 2.5.1 raises inside the MPS implementation rather than using its
    dispatcher fallback for ConvTranspose3d. Match that version's forward and
    private output-padding signature, including callers that pass output_size.
    """
    if torch_module.__version__ != TORCH_VERSION:
        raise ValueError(f'ConvTranspose3d adapter requires Torch {TORCH_VERSION}')
    cls = torch_module.nn.ConvTranspose3d
    original_forward = cls.forward
    state = {'mps_calls': 0}

    def forward(self, input, output_size=None):
        if input.device.type != 'mps':
            return original_forward(self, input, output_size)
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation)
        result = torch_module.nn.functional.conv_transpose3d(
            input.cpu(), self.weight.cpu(), None if self.bias is None else self.bias.cpu(),
            self.stride, self.padding, output_padding, self.groups, self.dilation)
        state['mps_calls'] += 1
        return result.to(input.device)

    cls.forward = forward
    return state


def main(argv=None):
    args = parse_args(argv)  # --help exits before any torch/TSS import.
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve()
    if not args.input.exists():
        raise ValueError(f'Input does not exist: {args.input}')
    checked_assets = validate_assets(args.data_dir)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['nnUNet_def_n_proc'] = str(TORCH_THREADS)
    import torch
    if torch.__version__ != TORCH_VERSION:
        raise ValueError(f'Requires Torch {TORCH_VERSION}; found {torch.__version__}')
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; run with local Metal access.')
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    transpose_shim = install_conv_transpose3d_cpu_shim(torch)
    # Upstream add_trainer is non-overwriting, but require its existing file to
    # prevent this adapter from creating files inside the installed runtime.
    import nnunetv2
    trainer = (Path(nnunetv2.__file__).parent / 'training' / 'nnUNetTrainer'
               / 'nnUNetTrainerDAExt.py')
    if not trainer.is_file():
        raise RuntimeError('The pinned runtime must already contain nnUNetTrainerDAExt.py')
    tss = importlib.import_module('totalspineseg.inference')
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'mps-runtime.json'
    runtime = {
        'schema_version': 1, 'provider': 'totalspineseg',
        'runtime_mode': 'mps_original_context', 'device': 'mps',
        'inference_backend': 'mps_cpu_mixed',
        'experimental': True, 'validation_status': 'not_validated',
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'packages': {name: version(name) for name in
                     ('totalspineseg', 'torch', 'nnunetv2', 'dynamic_network_architectures')},
        'weights_release': WEIGHTS_RELEASE, 'verified_model_assets': checked_assets,
        'original_model_files_modified': False, 'original_plans_used': True,
        'sliding_window_context_changed': False, 'automatic_weight_downloads': False,
        'mps_operator_cpu_fallback': True,
        'operator_backends': {'Conv3d': 'mps', 'AvgPool3d': 'cpu_dispatcher_fallback',
                              'ConvTranspose3d': 'cpu_explicit_shim'},
        'explicit_cpu_conv_transpose3d_calls': 0,
        'cpu_preprocessing_and_logit_accumulation': True,
        'torch_threads': torch.get_num_threads(),
    }

    def save_runtime():
        marker.write_text(json.dumps(runtime, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    save_runtime()
    print('Experimental mixed MPS/CPU adapter: original plans, Torch threads 4; '
          'AvgPool3d uses CPU dispatcher fallback, ConvTranspose3d uses an explicit CPU shim.',
          flush=True)
    try:
        tss.inference(
            input_path=args.input, output_path=args.output, data_path=args.data_dir,
            default_release=WEIGHTS_RELEASE, device=torch.device('mps'),
            output_iso=args.iso, loc_path=None, suffix=[''], loc_suffix='',
            step1_only=args.step1, keep_only=args.keep_only,
            max_workers=args.max_workers,
            max_workers_nnunet=min(args.max_workers_nnunet, args.max_workers), quiet=args.quiet,
        )
        torch.mps.synchronize()
    except BaseException as error:
        runtime.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    else:
        runtime['status'] = 'completed'
    finally:
        runtime['explicit_cpu_conv_transpose3d_calls'] = transpose_shim['mps_calls']
        runtime['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_runtime()
    return 0


if __name__ == '__main__':
    sys.exit(main())
