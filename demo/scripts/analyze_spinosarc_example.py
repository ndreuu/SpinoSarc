#!/usr/bin/env python3
"""Prepare canal/muscle results for the open example without opening a window.

Requires a completed local TSS run for the exact sagittal example. Uses the
same native-frame QThread as the desktop app; no patient demographics invented.
"""
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
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--manifest', type=Path, default=root / 'data/spinosarc-example/manifest.json')
    args = parser.parse_args()
    model = root / 'var/spinosarc/musclemap'
    os.environ.update(SPINOSARC_MUSCLEMAP_PATH=str(model / 'source/scripts'),
                     SPINOSARC_MUSCLEMAP_MANIFEST=str(model / 'manifest.json'),
                     SPINOSARC_MUSCLEMAP_MODEL_VERSION='0.0',
                     MPLCONFIGDIR=str(root / 'var/spinosarc/matplotlib'),
                     OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                     PYTORCH_ENABLE_MPS_FALLBACK='1')
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from PyQt6.QtCore import QCoreApplication
    from spinosarc_app.analyzer import SpinoSarcAnalyzer
    from spinosarc_app.demo_io import prepare_volumes
    from spinosarc_app.lumbar_demo import MuscleAnalysisWorker, serializable_result, restrict_l3_indices
    from spinosarc_app.totalspineseg.level_mapper import LevelMapper
    from spinosarc_app.totalspineseg.multi_level_analyzer import MultiLevelAnalyzer

    torch.set_num_threads(4)
    work = root / 'var/spinosarc'
    example = json.loads(args.manifest.read_text())
    volumes = prepare_volumes(example['demo_paths'], work / 'inputs')
    source = volumes['sagittal_nifti_path']
    candidates = sorted((work / 'analyses').glob('*/process-result.json'),
                        key=lambda path: path.stat().st_mtime_ns, reverse=True)
    chosen = None
    for path in candidates:
        result = json.loads(path.read_text())
        if (result.get('success') and source in result.get('command', [])
                and Path(result['output_dir']).resolve() == path.parent.resolve()):
            chosen = path.parent, result
            break
    if chosen is None:
        raise SystemExit('Run Detect Levels on the open axial+sagittal example first; no matching completed TSS run exists.')
    directory, segmentation = chosen
    mapper = LevelMapper()
    levels = mapper.map_to_axial(mapper.parse(str(directory), source), volumes['axial_slices'])
    canals = list((directory / 'step1_canal').glob('*.nii.gz'))
    if len(canals) != 1:
        raise ValueError('Expected one completed canal mask.')
    measurements = MultiLevelAnalyzer(SpinoSarcAnalyzer(canal_only=True), volumes['axial_slices'], str(canals[0])).analyze_all(
        levels, slice_nifti_producer=lambda _: None)
    indices = [info['axial_slice_idx'] for info in levels.values() if info.get('axial_slice_idx') is not None]
    if not indices:
        raise ValueError('No detected levels are covered by native axial frames.')
    started = time.monotonic()
    analyzer = SpinoSarcAnalyzer(use_gpu=not args.cpu)
    app = QCoreApplication([])
    worker = MuscleAnalysisWorker(analyzer, volumes['axial_slices'], indices, None, directory / 'muscles', {})
    output = []
    worker.progress.connect(lambda message: print(message, flush=True))
    worker.result_ready.connect(lambda result: (output.append(result), app.quit()))
    worker.start()
    app.exec()
    if not worker.wait(30_000) or len(output) != 1:
        raise RuntimeError('Muscle worker did not finish cleanly.')
    inferred = output[0]
    if inferred['errors'] or not inferred['frames']:
        raise RuntimeError(f'Actual muscle inference failed: {inferred["errors"]}')
    l3_index = levels.get('L3_body', {}).get('axial_slice_idx')
    for index, result in inferred['frames'].items():
        restrict_l3_indices(result, index == l3_index)
    for entry in measurements['levels'].values():
        result = inferred['frames'].get(entry.get('axial_slice_idx'))
        if result:
            entry.update(muscles=result['muscles'], asymmetry=result['asymmetry'],
                         muscle_mask_path=result['muscle_mask_path'], muscle_assessment_status='research_only')
    l3 = inferred['frames'].get(l3_index)
    measurements.update(analysis_mode='canal_and_muscles', muscle_assessment_status='research_only',
                        sarcopenia={'level_used': 'L3_body', 'axial_slice_idx': l3_index,
                                   'result': l3.get('sarcopenia') if l3 else None,
                                   'note': 'No nearest-slice substitution; missing L3 coverage/height remain unassessed.'})
    findings = dict(analysis_mode='canal_and_muscles', numbering_status='needs_confirmation',
                    diagnostic_assessment_status='not_assessed', sources=volumes['sources'],
                    detected_levels=levels, segmentation=segmentation, canal_nifti_path=str(canals[0]),
                    dural_sac_assessment_status='not_assessed', canal_area_assessment_status='research_only',
                    registration_status=volumes.get('registration_status', 'not_assessed'),
                    geometry_warnings=volumes.get('geometry_warnings', []),
                    multi_level_measurements=measurements, muscle_assessment_status='research_only',
                    muscle_frames={str(index): serializable_result(result) for index, result in inferred['frames'].items()},
                    muscle_errors={}, sarcopenia_diagnosis_status='not_assessed')
    (directory / 'findings.json').write_text(json.dumps(findings, indent=2, allow_nan=False) + '\n')
    report = dict(success=True, real_inference_run=True, native_window_opened=False,
                  native_qthread_inference=True, worker_stack_bytes=worker.stackSize(),
                  device=str(analyzer.engine.device), muscle_duration_sec=time.monotonic() - started,
                  tss_duration_sec=segmentation['duration_sec'], findings_path=str(directory / 'findings.json'),
                  source=example['source'], license=example['license'],
                  levels={key: {'canal_area_mm2': value.get('canal_csa_mm2'),
                                'muscle_count': len(value.get('muscles', [])),
                                'axial_slice_idx': value.get('axial_slice_idx')}
                          for key, value in measurements['levels'].items()},
                  sarcopenia=measurements['sarcopenia'], registration_status=findings['registration_status'],
                  model=analyzer.engine.provenance)
    artifact = root / 'artifacts/spinosarc-full-example-smoke.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
