#!/usr/bin/env python3
"""Run the official MuscleMap abdomen weights on a real native demo MR slice."""
from pathlib import Path
from _demo_paths import REPO_ROOT, runtime_root
import argparse
import json
import os
import sys
import time


def main():
    root = runtime_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', type=Path,
                        default=root / 'data/spinosarc-example/sudirman-0001/T2_TSE_TRA_384_0004')
    parser.add_argument('--frame-index', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, default=root / 'var/spinosarc/musclemap-smoke')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    model_root = root / 'var/spinosarc/musclemap'
    os.environ.setdefault('SPINOSARC_MUSCLEMAP_PATH', str(model_root / 'source/scripts'))
    os.environ.setdefault('SPINOSARC_MUSCLEMAP_MANIFEST', str(model_root / 'manifest.json'))
    os.environ.setdefault('SPINOSARC_MUSCLEMAP_MODEL_VERSION', '0.0')
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'var/spinosarc/matplotlib'))
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import nibabel as nib
    import torch
    from spinosarc_app.demo_io import prepare_volumes, write_native_slice_nifti
    from spinosarc_app.analyzer import SpinoSarcAnalyzer

    torch.set_num_threads(4)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_volumes([args.series], output_dir / 'prepared')
    frames = prepared['axial_slices']
    if not 0 <= args.frame_index < len(frames):
        raise ValueError('Requested frame is outside the native axial series.')
    frame = frames[args.frame_index]
    source = Path(write_native_slice_nifti(frame, output_dir / 'native_slice.nii.gz'))
    started = time.monotonic()
    analyzer = SpinoSarcAnalyzer(use_gpu=not args.cpu, canal_only=False)
    result = analyzer.analyze(str(source))
    elapsed = time.monotonic() - started
    segmentation = result.pop('segmentation_mask')
    result.pop('image_array')
    original = nib.load(source)
    mask_path = output_dir / 'native_slice_muscle_dseg.nii.gz'
    predicted = nib.Nifti1Image(segmentation[:, :, None].astype(np.int16), original.affine, original.header)
    nib.save(predicted, mask_path)
    native_mask = segmentation.T
    if native_mask.shape != frame['pixel_array'].shape or not result['muscles']:
        raise ValueError('Real native-slice muscle inference produced no aligned muscle measurements.')
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axis = plt.subplots(figsize=(8, 8))
    pixel_spacing = frame['pixel_spacing']
    axis.imshow(frame['pixel_array'], cmap='gray', aspect=pixel_spacing[0] / pixel_spacing[1])
    colors = ListedColormap(['#000000', '#e6194b', '#f58231', '#ffe119', '#bfef45',
                             '#3cb44b', '#42d4f4', '#4363d8', '#911eb4'])
    axis.imshow(np.ma.masked_where(native_mask == 0, native_mask), cmap=colors, vmin=0, vmax=8,
                alpha=.5, aspect=pixel_spacing[0] / pixel_spacing[1], interpolation='nearest')
    axis.set_title('Official MuscleMap abdomen v0.0 — native T2 MRI')
    axis.set_axis_off()
    fig.savefig(output_dir / 'native_slice_muscles.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    report = {
        'success': True, 'real_inference_run': True, 'elapsed_seconds': elapsed,
        'device': str(analyzer.engine.device), 'source_series': str(args.series.resolve()),
        'source_sop_instance_uid': frame['sop_instance_uid'], 'frame_index': args.frame_index,
        'native_pixel_shape': list(native_mask.shape),
        'native_pixel_spacing_mm': list(pixel_spacing),
        'source_nifti': str(source), 'predicted_mask': str(mask_path),
        'native_shape_and_affine_preserved': True, 'model': analyzer.engine.provenance,
        'labels_found': [int(label) for label in np.unique(segmentation) if label],
        'muscles': result['muscles'], 'asymmetry': result['asymmetry'],
        'sarcopenia': result['sarcopenia'], 'demographics_fabricated': False,
        'fat_fraction_method': result['fat_fraction_method'],
        'preview': str(output_dir / 'native_slice_muscles.png'),
    }
    artifact = root / 'artifacts/spinosarc-musclemap-smoke.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'success': True, 'device': report['device'], 'elapsed_seconds': elapsed,
                      'labels_found': report['labels_found'], 'artifact': str(artifact)}, indent=2))


if __name__ == '__main__':
    main()
