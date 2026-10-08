#!/usr/bin/env python3
"""Experimental CPU wrapper for the pinned TotalSpineSeg CLI.

Usage: .venv-tss/bin/python demo/scripts/totalspineseg_cpu_demo.py INPUT OUTPUT
       --device cpu --step1 --data-dir var/totalspineseg [other TSS options]

Changes only the initialized predictor's in-memory sliding-window tile. Original
model plans/checkpoints are never written. Smaller context changes predictions;
this mode is not validated and is intended only for the local research demo.
"""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import json
import math
import os
import sys


DEMO_PATCH = (128, 128, 64)
TORCH_THREADS = 4
EXPECTED_PACKAGES = {
    'totalspineseg': '20260730',
    'nnunetv2': '2.6.2',
    'dynamic_network_architectures': '0.4.4',
    'torch': '2.5.1',
}
NETWORK_CLASS = 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'


def validate_demo_configuration(configuration, patch=DEMO_PATCH):
    """Allow only the audited fully convolutional 3D architecture/valid tiles."""
    if len(patch) != 3 or any(type(size) is not int or size <= 0 for size in patch):
        raise ValueError('Demo patch must contain three positive integer dimensions')
    architecture = configuration.get('architecture', {})
    kwargs = architecture.get('arch_kwargs', {})
    if architecture.get('network_class_name') != NETWORK_CLASS:
        raise ValueError('CPU demo only supports the audited ResidualEncoderUNet')
    if kwargs.get('conv_op') != 'torch.nn.modules.conv.Conv3d':
        raise ValueError('CPU demo requires the original 3D convolution architecture')
    if any(key in kwargs for key in ('patch_size', 'input_size', 'image_size', 'input_shape')):
        raise ValueError('Fixed spatial shape in architecture arguments is unsupported')
    strides = kwargs.get('strides')
    if not isinstance(strides, list) or len(strides) != kwargs.get('n_stages'):
        raise ValueError('Architecture strides must describe every encoder stage')
    if not strides or any(not isinstance(row, (list, tuple)) or len(row) != 3 or
                          any(type(value) is not int or value <= 0 for value in row)
                          for row in strides):
        raise ValueError('Expected three positive integer strides per stage')
    divisibility = tuple(math.prod(row[axis] for row in strides) for axis in range(3))
    if any(size % divisor for size, divisor in zip(patch, divisibility)):
        raise ValueError(f'Demo patch {patch} must be divisible by strides {divisibility}')
    original_patch = configuration.get('patch_size')
    if not isinstance(original_patch, (list, tuple)) or len(original_patch) != 3 or any(
            type(size) is not int or size <= 0 for size in original_patch):
        raise ValueError('The original model patch is invalid')
    if any(size > original for size, original in zip(patch, original_patch)):
        raise ValueError('Demo tile must not exceed the original model patch')
    return {'trained_patch_size_voxels': list(original_patch),
            'demo_patch_size_voxels': list(patch),
            'stride_divisibility_voxels': list(divisibility),
            'network_class': NETWORK_CLASS}


def apply_demo_configuration(predictor):
    """Detach the mutable configuration from loaded plans before changing it."""
    configuration = predictor.configuration_manager.configuration
    details = validate_demo_configuration(configuration)
    demo = deepcopy(configuration)
    demo['patch_size'] = list(DEMO_PATCH)
    predictor.configuration_manager.configuration = demo
    return details


def install_cpu_patch(predictor_class, torch_module, metadata_output, packages):
    original_initialize = predictor_class.initialize_from_trained_model_folder
    metadata = {
        'schema_version': 1,
        'provider': 'totalspineseg',
        'runtime_mode': 'cpu_demo_reduced_context',
        'experimental': True,
        'validation_status': 'not_validated',
        'context_change': 'Reduced sliding-window context can change segmentation and measurements.',
        'original_model_files_modified': False,
        'torch_threads': TORCH_THREADS,
        'nnUNet_def_n_proc': TORCH_THREADS,
        'requested_patch_size_voxels': list(DEMO_PATCH),
        'packages': packages,
        'models': [],
    }

    def initialize_with_demo_tile(self, *args, **kwargs):
        original_initialize(self, *args, **kwargs)
        if getattr(getattr(self, 'device', None), 'type', None) != 'cpu':
            raise ValueError('The reduced-context CPU demo requires --device cpu')
        details = apply_demo_configuration(self)
        torch_module.set_num_threads(TORCH_THREADS)
        details['actual_torch_threads'] = torch_module.get_num_threads()
        model_folder = args[0] if args else kwargs.get('model_training_output_dir')
        details['model_folder'] = str(model_folder)
        metadata['models'].append(details)
        if metadata_output is None:
            raise ValueError('CPU demo requires INPUT OUTPUT as the first CLI arguments')
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('Experimental CPU demo: tile 128×128×64, Torch threads 4; reduced context is not validated.',
              flush=True)

    predictor_class.initialize_from_trained_model_folder = initialize_with_demo_tile


def main():
    arguments = sys.argv[1:]
    help_requested = any(arg in ('--help', '-h') for arg in arguments)
    output = None
    if not help_requested:
        if len(arguments) < 2 or any(arg.startswith('-') for arg in arguments[:2]):
            raise SystemExit('Usage: totalspineseg_cpu_demo.py INPUT OUTPUT --device cpu [TSS options]')
        output = Path(arguments[1]).resolve() / 'cpu-demo-runtime.json'
        # Keep CPU behavior explicit even if the original CLI would choose CUDA.
        if '--device' not in arguments and not any(arg.startswith('--device=') for arg in arguments):
            sys.argv.extend(['--device', 'cpu'])
    packages = {name: version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise SystemExit(f'CPU demo requires the pinned runtime {EXPECTED_PACKAGES}; found {packages}')
    # Must be set before importing nnunetv2.configuration (which reads it once).
    os.environ['nnUNet_def_n_proc'] = str(TORCH_THREADS)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from totalspineseg.inference import main as totalspineseg_main
    torch.set_num_threads(TORCH_THREADS)
    install_cpu_patch(nnUNetPredictor, torch, output, packages)
    return totalspineseg_main()


if __name__ == '__main__':
    sys.exit(main())
