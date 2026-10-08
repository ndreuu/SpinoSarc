#!/usr/bin/env python3
"""Tiny real-Metal operator smoke check; no TSS weights, inference, or GUI.

Run with .venv-tss/bin/python demo/scripts/smoke_spinosarc_mps.py outside the sandbox
that hides local Metal devices. --output optionally saves the check as JSON.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    import torch
    from totalspineseg_mps import install_conv_transpose3d_cpu_shim
    if not torch.backends.mps.is_available():
        raise RuntimeError('Local Metal access is required; MPS is unavailable')
    torch.manual_seed(7)
    torch.set_num_threads(4)
    with torch.inference_mode():
        data = torch.randn(1, 2, 16, 16, 16, device='mps')
        layer = torch.nn.Conv3d(2, 4, 3, padding=1).to('mps')
        output = layer(data)
        torch.mps.synchronize()
        assert tuple(output.shape) == (1, 4, 16, 16, 16)
        assert output.device.type == 'mps'
        assert bool(torch.isfinite(output).all().item())
    shim = install_conv_transpose3d_cpu_shim(torch)
    cases = []
    for bias, groups in ((True, 1), (False, 2)):
        cpu_layer = torch.nn.ConvTranspose3d(
            2, 4, 3, stride=2, padding=1, bias=bias, groups=groups).eval()
        mps_layer = deepcopy(cpu_layer).to('mps')
        data = torch.randn(1, 2, 4, 5, 6)
        for output_size in (None, [1, 4, 8, 10, 12], [8, 10, 12]):
            with torch.inference_mode():
                expected = cpu_layer(data, output_size=output_size)
                actual = mps_layer(data.to('mps'), output_size=output_size)
                torch.mps.synchronize()
            assert actual.device.type == mps_layer.weight.device.type == 'mps'
            assert mps_layer.bias is None or mps_layer.bias.device.type == 'mps'
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
            cases.append({'bias': bias, 'groups': groups, 'output_size': output_size,
                          'output_shape': list(actual.shape),
                          'max_abs_error': (actual.cpu() - expected).abs().max().item(),
                          'output_device': str(actual.device), 'status': 'passed'})
    assert shim['mps_calls'] == 6
    result = {
        'schema_version': 1, 'scope': 'tiny_operator_fixtures_only',
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__},
        'conv3d': {'status': 'passed', 'output_shape': list(output.shape),
                   'output_device': str(output.device), 'all_values_finite': True,
                   'mps_synchronize_passed': True},
        'conv_transpose3d_explicit_cpu_shim': {'cases': cases, 'shim_calls': shim['mps_calls'],
                                             'model_parameters_remained_on_mps': True,
                                             'mps_synchronize_passed': True, 'status': 'passed'},
        'large_model_weights_loaded': False, 'full_tss_inference_run': False, 'gui_used': False,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
