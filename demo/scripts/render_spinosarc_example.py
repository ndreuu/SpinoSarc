#!/usr/bin/env python3
"""Render real native axial MRI, predicted muscles and canal measurements."""
from pathlib import Path
from _demo_paths import REPO_ROOT, runtime_root
import argparse
import json
import os
import sys


def main():
    root = runtime_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--findings', type=Path)
    parser.add_argument('--output', type=Path, default=root / 'artifacts/spinosarc-open-example.png')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'var/spinosarc/matplotlib'))
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from spinosarc_app.demo_io import prepare_volumes
    from spinosarc_app.lumbar_demo import frame_identity
    from spinosarc_app.totalspineseg.canal_csa import resample_canal_to_axial_slice

    if args.findings is None:
        candidates = sorted((root / 'var/spinosarc/analyses').glob('*/findings.json'),
                            key=lambda path: path.stat().st_mtime_ns, reverse=True)
        args.findings = next(path for path in candidates if json.loads(path.read_text()).get('muscle_frames'))
    findings = json.loads(args.findings.read_text())
    prepared = prepare_volumes([source['path'] for source in findings['sources']], root / 'var/spinosarc/inputs')
    levels = [(name, info) for name, info in findings['multi_level_measurements']['levels'].items()
              if info.get('muscle_mask_path')]
    if not levels:
        raise ValueError('No real muscle predictions are available to render.')
    colors = ListedColormap(['#000000', '#e6194b', '#f58231', '#ffe119', '#bfef45',
                             '#3cb44b', '#42d4f4', '#4363d8', '#911eb4'])
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 8), facecolor='white', squeeze=False)
    fig.subplots_adjust(top=.82, bottom=.32, wspace=.04)
    rows = []
    for column, (name, info) in enumerate(levels):
        index = info['axial_slice_idx']
        frame = prepared['axial_slices'][index]
        saved = findings['muscle_frames'][str(index)]
        if json.dumps(saved['source_frame'], sort_keys=True) != json.dumps(frame_identity(frame), sort_keys=True):
            raise ValueError('Saved prediction belongs to another native frame.')
        mask = np.asarray(nib.load(saved['muscle_mask_path']).dataobj).squeeze().T
        if mask.shape != frame['pixel_array'].shape:
            raise ValueError('Prediction and native image shapes differ.')
        canal = resample_canal_to_axial_slice(findings['canal_nifti_path'], frame)
        pixels = frame['pixel_array']
        axis = axes[0, column]
        aspect = frame['pixel_spacing'][0] / frame['pixel_spacing'][1]
        lo, hi = np.percentile(pixels, [1, 99.5])
        axis.imshow(pixels, cmap='gray', vmin=lo, vmax=hi, aspect=aspect)
        axis.imshow(np.ma.masked_where(mask == 0, mask), cmap=colors, vmin=0, vmax=8,
                    alpha=.42, interpolation='nearest', aspect=aspect)
        if canal is not None and canal.any():
            axis.contour(canal, levels=[.5], colors=['#00e5ff'], linewidths=1.5)
        axis.set_title(f"{name}\nКанал: {info['canal_csa_mm2']:.1f} мм² · мышц: {len(info['muscles'])}", fontsize=13)
        axis.set_axis_off()
        by_name = {muscle['name']: muscle for muscle in info['muscles']}
        summary = [name, f"{info['canal_csa_mm2']:.1f}"]
        for group in ('multifidus', 'erector', 'psoas', 'QL'):
            pair = [by_name.get(f'{group}_{side}') for side in ('R', 'L')]
            summary.append(' / '.join(f"{muscle['csa_mm2']:.0f}" if muscle else '—' for muscle in pair))
        rows.append(summary)
    fig.suptitle('Поясничное МРТ · открытый пример Sudirman 0001', fontsize=20, fontweight='bold', y=.97)
    fig.text(.5, .90, 'Исходные аксиальные T2-срезы · цвет: MuscleMap · голубой контур: канал TotalSpineSeg',
             ha='center', fontsize=12)
    table_axis = fig.add_axes([.06, .12, .88, .14]); table_axis.set_axis_off()
    table = table_axis.table(cellText=rows,
                            colLabels=['Уровень*', 'Канал, мм²', 'Multifidus R / L', 'Erector R / L', 'Psoas R / L', 'QL R / L'],
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d0d7df')
        cell.set_facecolor('#eaf1f8' if row == 0 else 'white')
    fig.text(.5, .067, 'Площади мышц в мм²; R / L — справа / слева. «—» — класс не найден моделью.', ha='center', fontsize=10)
    fig.text(.5, .036, '*Нумерация и совмещение серий требуют проверки. L1–L3 и тело L3 не покрыты; PMI не оценён.', ha='center', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor='white'); plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
