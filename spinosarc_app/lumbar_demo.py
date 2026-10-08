"""Local spine demo built on SpinoSarc's existing desktop viewer."""
from pathlib import Path
import json
import os
import sys
import time
import uuid
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from PyQt6.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt6.QtWidgets import (QApplication, QFileDialog, QLabel, QMessageBox, QPushButton,
                            QListWidgetItem, QTableWidgetItem, QScrollArea, QSplitter,
                            QLayout, QFrame)
from .gui import SpinoSarcWindow, SUCCESS, PRIMARY, EngineLoaderThread
from .analyzer import SpinoSarcAnalyzer
from .demo_io import prepare_volumes, write_native_slice_nifti, native_slice_affine
from .totalspineseg.runner import TotalSpineSegRunner
from .totalspineseg.level_mapper import LevelMapper


class SegmentationWorker(QThread):
    result_ready = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, source, output, parent=None):
        super().__init__(parent)
        self.source, self.output = source, output
        self.runner = TotalSpineSegRunner()

    def run(self):
        try:
            result = self.runner.run(self.source, self.output, device=os.environ.get('SPINE_TSS_DEVICE', 'cpu'),
                                     step1_only=True, iso=True,
                                     progress_callback=self.progress.emit)
        except Exception as error:
            result = {'success': False, 'error': str(error)}
        self.result_ready.emit(result)


def serializable_result(result):
    """Keep metrics and provenance in JSON; image arrays live in NIfTI files."""
    return {key: value for key, value in result.items()
            if key not in ('image_array', 'segmentation_mask')}


def frame_identity(frame):
    return {key: frame[key] for key in (
        'source_kind', 'source_volume', 'source_path', 'native_frame_index',
        'native_frame_axis', 'sop_instance_uid', 'series_instance_uid',
        'image_position', 'image_orientation', 'pixel_spacing') if key in frame}


def restrict_l3_indices(result, is_l3):
    """A psoas area can be reported anywhere; the L3 PMI cannot."""
    if not is_l3:
        sarc = dict(result.get('sarcopenia') or {})
        sarc.update(pmi_cm2_per_m2=None, thresholds={}, risk_category='Unknown')
        sarc['notes'] = list(sarc.get('notes', [])) + ['This frame is not the matched L3 body reference slice; PMI is not assessed.']
        result['sarcopenia'] = sarc
        result['sarcopenia_assessment_status'] = 'not_l3_reference_slice'
    return result


class MuscleAnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def __init__(self, analyzer, frames, indices, demographics, directory, cache, parent=None):
        super().__init__(parent)
        # macOS Qt's ~512 KiB stack is too small for OpenBLAS/MONAI affine
        # inversion. A native worker needs enough stack for the original model.
        self.setStackSize(16 * 1024 * 1024)
        self.analyzer, self.frames, self.indices = analyzer, frames, sorted(set(indices))
        self.demographics, self.directory, self.cache = demographics, Path(directory), cache

    def run(self):
        results, errors = {}, {}
        self.directory.mkdir(parents=True, exist_ok=True)
        for index in self.indices:
            if self.isInterruptionRequested():
                break
            self.progress.emit(f'Analyzing muscles: axial slice {index + 1} ({len(results) + 1}/{len(self.indices)})...')
            try:
                if index in self.cache:
                    result = dict(self.cache[index])
                    # Demographics may have changed since the cached segmentation.
                    muscles = self.analyzer._compute_muscle_metrics(
                        result['image_array'], result['segmentation_mask'], result['pixel_area_mm2'])
                    from dataclasses import asdict
                    result['sarcopenia'] = asdict(self.analyzer._compute_sarcopenia(muscles, self.demographics))
                    result['demographics'] = asdict(self.demographics) if self.demographics else None
                else:
                    source = write_native_slice_nifti(self.frames[index], self.directory / f'frame-{index:03d}.nii.gz')
                    result = self.analyzer.analyze(str(source), self.demographics)
                    image = nib.load(source)
                    mask = np.asarray(result['segmentation_mask'])
                    if mask.ndim == 2:
                        mask = mask[:, :, None]
                    if mask.shape != image.shape:
                        raise ValueError('Muscle prediction does not match the native input grid.')
                    mask_path = self.directory / f'frame-{index:03d}_muscles.nii.gz'
                    saved = nib.Nifti1Image(mask.astype(np.int16), image.affine, image.header)
                    saved.header.set_data_dtype(np.int16); nib.save(saved, mask_path)
                    result['muscle_mask_path'] = str(mask_path)
                frame = self.frames[index]
                result['source_frame'] = frame_identity(frame)
                result['assessment_status'] = 'research_only'
                result['fat_fraction_method'] = 'intensity_otsu_estimate_not_dixon'
                results[index] = result
            except Exception as error:
                errors[index] = str(error)
        self.result_ready.emit({'frames': results, 'errors': errors,
                                'cancelled': self.isInterruptionRequested()})


class SpinoSarcDemoWindow(SpinoSarcWindow):
    def __init__(self):
        self.work_dir = Path(os.environ['SPINOSARC_WORK_DIR'])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.worker = None
        self.muscle_worker = None
        self.segmentation_pending = False
        self.muscle_pending = False
        self.muscle_loader = None
        self.muscle_analyzer = None
        self.muscle_results = {}
        self.muscles_requested = os.environ.get('SPINOSARC_ENABLE_MUSCLES') == '1'
        self.sources = []
        self.registration_status = 'not_assessed'
        self.geometry_warnings = []
        self.sagittal_canal_mask = None
        super().__init__()
        self.setWindowTitle('SpinoSarc — Lumbar MRI Demo')
        self.setMinimumSize(1050, 680); self.resize(1280, 800)
        for widget in (self.risk_label, self.pmi_label, self.muscle_table):
            widget.setVisible(self.muscles_requested)
        self.muscle_table.setHorizontalHeaderLabels(['Muscle', 'Area mm²', 'Intensity FF* %'])
        self.muscle_table.setToolTip('FF is an intensity-based Otsu estimate, not a quantitative Dixon fat fraction.')
        self._make_sidebar_scrollable()
        for label in self.findChildren(QLabel):
            if label.text() == 'Paraspinal Muscle & Sarcopenia Analyzer':
                label.setText('Lumbar anatomy, canal and muscles · research demo')
            elif label.text() == 'AXIAL':
                label.setText('NATIVE AXIAL · optional')
        self.results_group.setTitle('Spine / canal results')
        self.save_pdf_btn.setText('Save annotated PNG')
        self.export_excel_btn.setText('Export findings JSON')
        self.analyze_btn.setText('Analyze current axial slice' if self.muscles_requested else 'Measure current axial slice')
        self.analyze_all_btn.setText('Analyze all covered levels' if self.muscles_requested else 'Measure all covered levels')
        self.analyze_all_btn.setToolTip('Measure canal area on covered native axial slices. Numbering needs review.')
        self.levels_status_label.setText('Load sagittal T2, then detect the lumbar levels.')
        self.csa_label.setText('Canal area: not assessed')
        self.status_label.setText('Ready — load sagittal T2; native axial T2 is optional.')
        example_manifest = os.environ.get('SPINOSARC_AXIAL_EXAMPLE_MANIFEST')
        button = QPushButton('Load open axial + sagittal example' if example_manifest and Path(example_manifest).is_file() else 'Load open SPIDER example')
        button.clicked.connect(self.load_open_example)
        self.centralWidget().layout().insertWidget(2, button)

    def _make_sidebar_scrollable(self):
        """Keep controls readable when the demo runs on a laptop display."""
        splitter = self.centralWidget().findChild(QSplitter)
        sidebar = splitter.widget(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(420)
        splitter.replaceWidget(1, scroll)
        sidebar.setMinimumWidth(0)
        sidebar.setMaximumWidth(16777215)
        sidebar.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.levels_list.setMinimumHeight(140)
        self.muscle_table.setMinimumHeight(250)
        self.risk_label.setWordWrap(True)
        for field in (self.patient_id_input, self.age_input, self.sex_input,
                      self.height_input, self.weight_input):
            field.setMinimumHeight(field.sizeHint().height())
        scroll.setWidget(sidebar)
        self.sidebar_scroll = scroll

    def load_open_example(self):
        if self._analysis_running():
            self.status_label.setText('Stop segmentation before reloading the example.')
            return
        manifest = os.environ.get('SPINOSARC_AXIAL_EXAMPLE_MANIFEST')
        if manifest and Path(manifest).is_file():
            example = json.loads(Path(manifest).read_text())
            self.load_files(example['demo_paths'])
        else:
            self.load_files([os.environ['SPINOSARC_DEMO_MRI']])
        self.restore_latest_segmentation()

    def restore_latest_segmentation(self):
        """Reuse a completed run for this exact prepared input, without inference."""
        source = getattr(self, 'sagittal_nifti_path', None)
        if not source:
            return False
        paths = (self.work_dir / 'analyses').glob('*/process-result.json')
        for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True):
            if path.stat().st_size > 300_000:
                continue
            try:
                result = json.loads(path.read_text())
                if (not result.get('success') or str(Path(source).resolve()) not in result.get('command', [])
                        or Path(result['output_dir']).resolve() != path.parent.resolve()):
                    continue
                self.analysis_output = path.parent
                findings_path = path.parent / 'findings.json'
                saved = json.loads(findings_path.read_text()) if findings_path.is_file() else {}
                self.last_result = None
                self._segmentation_finished(result)
                if self.last_result is not None:
                    if saved.get('sources') == self.sources:
                        self._restore_muscle_results(saved)
                    self.status_label.setText('Loaded saved TSS segmentation for this example. Numbering needs review.')
                    return True
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return False

    def _on_new_case(self):
        if self._analysis_running():
            self.status_label.setText('Stop segmentation before starting a new case.')
            return
        super()._on_new_case()
        self.sources = []
        self.detected_levels = {}
        self.canal_nifti_path = None
        self.sagittal_nifti_path = None
        self.sagittal_canal_mask = None
        self.multi_level_result = None
        self.muscle_results = {}
        self.registration_status = 'not_assessed'
        self.geometry_warnings = []
        self.axial_display.clear(); self.sagittal_display.clear()
        self.axial_display.clear_level_lines(); self.sagittal_display.clear_level_lines()
        self.levels_list.clear()
        self.slice_slider.setEnabled(False); self.sag_slider.setEnabled(False)
        self.slice_label.setText('Axial: -'); self.sag_slice_label.setText('Sagittal: -')
        self.detect_levels_btn.setEnabled(False); self.analyze_all_btn.setEnabled(False)
        self.levels_status_label.setText('Load sagittal T2, then detect the lumbar levels.')
        self.status_label.setText('Ready — drop MHA or NIfTI volumes to begin.')

    def _load_engine_async(self):
        self.analyzer = SpinoSarcAnalyzer(canal_only=True)
        device = os.environ.get('SPINE_TSS_DEVICE', 'cpu')
        self.engine_status.setText('Canal demo · TSS / ' + ('Apple GPU (MPS)' if device == 'mps' else 'experimental CPU tiles'))
        self.engine_status.setStyleSheet(f'color: {SUCCESS}; font-size: 11px;')
        if self.muscles_requested:
            self.engine_status.setText('Loading MuscleMap model...')
            self.muscle_loader = EngineLoaderThread()
            self.muscle_loader.finished.connect(self._muscle_engine_ready)
            self.muscle_loader.error.connect(self._muscle_engine_error)
            self.muscle_loader.start()

    def _muscle_engine_ready(self, analyzer):
        self.muscle_analyzer = analyzer
        self.engine_status.setText(f'TSS + MuscleMap ready · {analyzer.engine.device}')
        self._enable_analysis_buttons()

    def _muscle_engine_error(self, error):
        self.engine_status.setText('MuscleMap unavailable; canal measurements remain available')
        self.status_label.setText(f'Muscle model failed: {error}')
        self.muscles_requested = False
        self._enable_analysis_buttons()

    def _analysis_running(self):
        return bool(self.segmentation_pending or self.muscle_pending or
                    (self.worker and self.worker.isRunning()) or
                    (self.muscle_worker and self.muscle_worker.isRunning()))

    def _enable_analysis_buttons(self):
        ready = bool(self.axial_slices and getattr(self, 'canal_nifti_path', None)
                     and not self._analysis_running()
                     and (not self.muscles_requested or self.muscle_analyzer is not None))
        self.analyze_btn.setEnabled(ready); self.analyze_all_btn.setEnabled(ready)

    def load_files(self, paths):
        if self._analysis_running():
            self.status_label.setText('Wait for segmentation before loading another case.')
            return
        try:
            volumes = prepare_volumes(paths, self.work_dir / 'inputs')
        except Exception as error:
            QMessageBox.warning(self, 'Load MRI', str(error)); return
        self._on_new_case()
        self.detected_levels = {}; self.canal_nifti_path = None
        self.sagittal_canal_mask = None; self.multi_level_result = None
        self.sources = volumes['sources']
        self.registration_status = volumes.get('registration_status', 'not_assessed')
        self.geometry_warnings = volumes.get('geometry_warnings', [])
        self.axial_slices = volumes['axial_slices']
        self.sagittal_data = volumes['sagittal_data']
        self.sagittal_affine = volumes['sagittal_affine']
        self.sagittal_nifti_path = volumes['sagittal_nifti_path']
        self.drop_zone.hide(); self.viewer_widget.show(); self.new_case_btn.setEnabled(True)
        if self.sagittal_data is not None:
            zoom = np.linalg.norm(self.sagittal_affine[:3, :3], axis=0)
            self.sagittal_display._physical_spacing = (float(zoom[0]), float(zoom[1]))
            n = self.sagittal_data.shape[2]
            self.current_sag_idx = n // 2
            self.sag_slider.blockSignals(True); self.sag_slider.setRange(0, n - 1)
            self.sag_slider.setValue(self.current_sag_idx); self.sag_slider.blockSignals(False)
            self.sag_slider.setEnabled(True); self._render_sagittal()
        if self.axial_slices:
            self.axial_display._physical_spacing = self.axial_slices[0]['pixel_spacing']
            self.current_slice_idx = len(self.axial_slices) // 2
            self.slice_slider.setRange(0, len(self.axial_slices) - 1)
            self.slice_slider.setValue(self.current_slice_idx); self.slice_slider.setEnabled(True)
            self._on_slider(self.current_slice_idx)
        else:
            self.axial_display.setText('No native axial series\nCanal area is not assessed')
            self.slice_slider.setEnabled(False)
        self.detect_levels_btn.setEnabled(self.sagittal_data is not None)
        self.levels_status_label.setText('Ready to detect levels.' if self.sagittal_data is not None else 'Sagittal T2 is required for level detection.')
        self.analyze_btn.setEnabled(False); self.analyze_all_btn.setEnabled(False)
        self.file_info.setText(' | '.join(f"{Path(s['path']).name}: {s['plane']}" for s in self.sources))
        self.status_label.setText('MRI loaded. Click Detect Levels to run TotalSpineSeg.')

    def _render_sagittal(self):
        if self.sagittal_data is None:
            return
        mask = None if self.sagittal_canal_mask is None else self.sagittal_canal_mask[:, :, self.current_sag_idx]
        self.sagittal_display.set_image(self.sagittal_data[:, :, self.current_sag_idx],
                                       rotation=1, canal_overlay=mask)
        self.sag_slice_label.setText(f'Sagittal: {self.current_sag_idx + 1} / {self.sagittal_data.shape[2]}')
        self._update_sagittal_level_overlay()

    def _update_sagittal_level_overlay(self):
        """Project RAS level markers into the prepared ASR sagittal voxel grid."""
        if self.sagittal_data is None or self.sagittal_affine is None:
            return
        n_si = self.sagittal_data.shape[1]
        if n_si <= 1:
            return
        colors = {
            'L1-L2': (31, 119, 180),
            'L2-L3': (44, 160, 44),
            'L3-L4': (255, 127, 14),
            'L4-L5': (214, 39, 40),
            'L5-S': (148, 103, 189),
        }
        inverse = np.linalg.inv(self.sagittal_affine)
        lines = []
        for name, info in getattr(self, 'detected_levels', {}).items():
            if name not in colors:
                continue
            try:
                world = np.asarray(info['world_xyz'], dtype=float)
                if world.shape != (3,) or not np.isfinite(world).all():
                    continue
                voxel = inverse @ np.r_[world, 1.]
                y_frac = float(np.clip(1. - voxel[1] / (n_si - 1), 0., 1.))
            except (KeyError, ValueError, TypeError):
                continue
            lines.append({'y_frac': y_frac, 'color_rgb': colors[name],
                          'label': name, 'dashed': info.get('axial_slice_idx') is None})
        if self.sagittal_display._img is not None:
            self.sagittal_display._level_lines = lines
            self.sagittal_display._refresh()

    def _on_sag_slider(self, value):
        self.current_sag_idx = value; self._render_sagittal()

    def _update_locators(self):
        if self.axial_slices:
            self._refresh_axial_with_canal_overlay()
        self._render_sagittal()

    def _refresh_axial_with_canal_overlay(self):
        if not self.axial_slices:
            return
        frame = self.axial_slices[self.current_slice_idx]
        result = self.muscle_results.get(self.current_slice_idx)
        muscle_mask = None if result is None else np.asarray(result['segmentation_mask']).squeeze().T
        if muscle_mask is not None and muscle_mask.shape != frame['pixel_array'].shape:
            raise ValueError('Muscle mask and native axial image have different grids.')
        self.axial_display._physical_spacing = frame['pixel_spacing']
        canal_mask = self._compute_canal_overlay_for_axial_slice(self.current_slice_idx)
        self.axial_display.set_image(frame['pixel_array'], overlay_mask=muscle_mask,
                                     rotation=self.axial_rotation,
                                     canal_overlay=canal_mask)
        if canal_mask is not None:
            area = float(canal_mask.sum() * np.prod(frame['pixel_spacing']))
            self.csa_label.setText(f'Canal area: {area:.1f} mm² · research measurement')
        if result is not None:
            self._show_muscle_metrics(result)
        elif self.muscles_requested:
            self.muscle_table.setRowCount(0)
            self.pmi_label.setText('Muscles: this slice has not been analyzed')
            self.risk_label.setText('L3 indices: not assessed')

    def _reset_csa_display(self):
        self.csa_label.setText('Canal area: not assessed')

    def _on_detect_levels(self):
        if self.muscle_worker and self.muscle_worker.isRunning():
            self.status_label.setText('Wait for muscle analysis before running segmentation.'); return
        if self.worker and self.worker.isRunning():
            if hasattr(self.worker.runner, 'cancel'):
                self.worker.runner.cancel()
            return
        source = getattr(self, 'sagittal_nifti_path', None)
        if not source:
            QMessageBox.warning(self, 'Detect Levels', 'Load a sagittal T2 volume first.'); return
        output = self.work_dir / 'analyses' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        self.analysis_output = output
        self.worker = SegmentationWorker(source, str(output), self)
        self.segmentation_pending = True
        self.worker.progress.connect(self.status_label.setText)
        self.worker.result_ready.connect(self._segmentation_finished)
        self.worker.finished.connect(self._enable_analysis_buttons)
        self.detect_levels_btn.setText('Stop segmentation')
        self.analyze_btn.setEnabled(False); self.analyze_all_btn.setEnabled(False)
        device = os.environ.get('SPINE_TSS_DEVICE', 'cpu')
        self.status_label.setText('Running TotalSpineSeg on ' + ('Apple GPU (MPS).' if device == 'mps' else 'CPU with experimental tiles.') + ' First scan may take several minutes.')
        self.worker.start()

    def _segmentation_finished(self, result):
        self.segmentation_pending = False
        self.detect_levels_btn.setText('Detect Levels')
        self.analysis_output.mkdir(parents=True, exist_ok=True)
        (self.analysis_output / 'process-result.json').write_text(json.dumps(result, indent=2) + '\n')
        if not result.get('success'):
            self.status_label.setText(result.get('error', 'Segmentation failed') + f' Log: {self.analysis_output / "process-result.json"}')
            return
        try:
            mapper = LevelMapper()
            levels = mapper.parse(result['output_dir'], self.sagittal_nifti_path)
            self.detected_levels = mapper.map_to_axial(levels, self.axial_slices)
            canals = list((Path(result['output_dir']) / 'step1_canal').glob('*.nii.gz'))
            if len(canals) != 1:
                raise ValueError('Expected one canal output for this scan.')
            self.canal_nifti_path = str(canals[0])
            target = (self.sagittal_data.shape, self.sagittal_affine)
            self.sagittal_canal_mask = np.asarray(resample_from_to(nib.load(canals[0]), target, order=1).dataobj) > .5
            self.last_result = {'analysis_mode': 'canal_only', 'numbering_status': 'needs_confirmation',
                                'diagnostic_assessment_status': 'not_assessed', 'sources': self.sources,
                                'detected_levels': self.detected_levels, 'segmentation': result,
                                'canal_nifti_path': self.canal_nifti_path,
                                'registration_status': self.registration_status,
                                'geometry_warnings': self.geometry_warnings,
                                'dural_sac_assessment_status': 'not_assessed',
                                'canal_area_assessment_status': 'not_assessed' if not self.axial_slices else 'available_for_research_measurement'}
            self.levels_list.clear()
            for name, info in self.detected_levels.items():
                if info['type'] != 'IVD':
                    continue
                item = QListWidgetItem(f"{name} · numbering unconfirmed")
                item.setData(Qt.ItemDataRole.UserRole, info.get('axial_slice_idx'))
                self.levels_list.addItem(item)
            self._render_sagittal()
            self._refresh_axial_with_canal_overlay()
            self.save_pdf_btn.setEnabled(True); self.export_excel_btn.setEnabled(True)
            self._enable_analysis_buttons()
            self.levels_status_label.setText('Predicted lumbar levels — confirm numbering.')
            self._persist_findings()
            self.status_label.setText(f"Anatomy complete in {result['duration_sec']:.0f}s. Cyan: canal; level numbering needs review.")
        except Exception as error:
            self.status_label.setText(f'Output parsing failed: {error}')

    def _on_analyze(self):
        if self._analysis_running():
            return
        if not self.axial_slices or not getattr(self, 'canal_nifti_path', None):
            return
        mask = self._compute_canal_overlay_for_axial_slice(self.current_slice_idx)
        if mask is None:
            self.status_label.setText('Canal mask unavailable on this plane.'); return
        frame = self.axial_slices[self.current_slice_idx]
        area = float(mask.sum() * np.prod(frame['pixel_spacing']))
        self.csa_label.setText(f'Canal area: {area:.1f} mm² · research measurement')
        self.last_result['current_axial_measurement'] = {'area_mm2': area, 'slice_index': self.current_slice_idx,
                                                       'image_position_lps': frame['image_position'],
                                                       'pixel_spacing_mm': frame['pixel_spacing'],
                                                       'source_volume': frame['source_volume'],
                                                       'native_frame_index': frame['native_frame_index'],
                                                       'native_frame_axis': frame['native_frame_axis'],
                                                       'assessment_status': 'research_only',
                                                       'registration_status': self.registration_status,
                                                       'source_path': frame.get('source_path'),
                                                       'sop_instance_uid': frame.get('sop_instance_uid'),
                                                       'method': 'tss_mask_on_native_plane_pixel_count_times_pixel_area'}
        self._persist_findings()
        self._refresh_axial_with_canal_overlay()
        if self.muscles_requested and self.muscle_analyzer is not None:
            self._start_muscle_analysis([self.current_slice_idx])

    def _on_analyze_all_levels(self):
        from .totalspineseg.multi_level_analyzer import MultiLevelAnalyzer
        if self._analysis_running() or not self.axial_slices or not getattr(self, 'detected_levels', None):
            return
        result = MultiLevelAnalyzer(self.analyzer, self.axial_slices, self.canal_nifti_path).analyze_all(
            self.detected_levels, slice_nifti_producer=lambda _: None)
        # Canal measurements do not establish dural-sac stenosis grades.
        self.multi_level_result = result
        self.last_result['multi_level_measurements'] = result
        self.last_result['canal_area_assessment_status'] = 'research_only'
        self._persist_findings()
        self._show_level_measurements()
        if self.muscles_requested and self.muscle_analyzer is not None:
            indices = [info['axial_slice_idx'] for info in self.detected_levels.values()
                       if info.get('axial_slice_idx') is not None]
            self._start_muscle_analysis(indices)
        else:
            self.status_label.setText('Canal measurements complete. Diagnostic stenosis grades are not assessed.')

    def _show_level_measurements(self):
        result = self.multi_level_result
        if result is None:
            return
        self.levels_list.clear()
        for name, info in result['levels'].items():
            area = info.get('canal_csa_mm2')
            text = f'{name}: {area:.1f} mm²' if area is not None else f'{name}: not assessed'
            if info.get('muscles'):
                text += f" · {len(info['muscles'])} muscles"
            item = QListWidgetItem(text + ' · numbering unconfirmed')
            item.setData(Qt.ItemDataRole.UserRole, info.get('axial_slice_idx'))
            self.levels_list.addItem(item)

    def _start_muscle_analysis(self, indices):
        if not indices:
            self.status_label.setText('No detected levels are covered by the native axial series.'); return
        self.muscle_worker = MuscleAnalysisWorker(
            self.muscle_analyzer, self.axial_slices, indices, self._get_demographics(),
            self.analysis_output / 'muscles', dict(self.muscle_results), self)
        self.muscle_pending = True
        self.muscle_worker.progress.connect(self.status_label.setText)
        self.muscle_worker.result_ready.connect(self._muscle_analysis_finished)
        self.muscle_worker.finished.connect(self._enable_analysis_buttons)
        self.analyze_btn.setEnabled(False); self.analyze_all_btn.setEnabled(False)
        self.muscle_worker.start()

    def _muscle_analysis_finished(self, output):
        self.muscle_pending = False
        l3_index = self.detected_levels.get('L3_body', {}).get('axial_slice_idx')
        for index, result in output['frames'].items():
            restrict_l3_indices(result, index == l3_index)
        self.muscle_results.update(output['frames'])
        self.last_result['analysis_mode'] = 'canal_and_muscles'
        self.last_result['muscle_assessment_status'] = 'research_only' if self.muscle_results else 'failed'
        self.last_result['muscle_frames'] = {str(index): serializable_result(result)
                                            for index, result in self.muscle_results.items()}
        self.last_result['muscle_errors'] = {str(index): value for index, value in output['errors'].items()}
        self.last_result['sarcopenia_diagnosis_status'] = 'not_assessed'
        if self.multi_level_result:
            self.multi_level_result['analysis_mode'] = 'canal_and_muscles'
            self.multi_level_result['muscle_assessment_status'] = 'research_only'
            for level in self.multi_level_result['levels'].values():
                muscle = self.muscle_results.get(level.get('axial_slice_idx'))
                if muscle:
                    level.update(muscles=muscle['muscles'], asymmetry=muscle['asymmetry'],
                                 muscle_mask_path=muscle['muscle_mask_path'],
                                 muscle_assessment_status='research_only')
            l3_index = self.detected_levels.get('L3_body', {}).get('axial_slice_idx')
            l3 = self.muscle_results.get(l3_index)
            self.multi_level_result['sarcopenia'] = {
                'level_used': 'L3_body', 'axial_slice_idx': l3_index,
                'result': l3.get('sarcopenia') if l3 else None,
                'note': 'Research indices; no nearest-slice substitution. L3 numbering needs review.' if l3 else 'L3 body is not covered by native axial slices; indices not assessed.'}
            self._show_level_measurements()
        if output['frames'] and self.current_slice_idx not in self.muscle_results:
            self.slice_slider.setValue(next(iter(output['frames'])))
        self._refresh_axial_with_canal_overlay()
        self._persist_findings()
        errors = '; '.join(output['errors'].values())
        self.status_label.setText(f'Muscle analysis completed on {len(output["frames"])} native slices.' +
                                  (f' Errors: {errors}' if errors else ' Areas and intensity estimates are shown; numbering/alignment need review.'))

    def _show_muscle_metrics(self, result):
        muscles = result.get('muscles', [])
        self.muscle_table.setRowCount(len(muscles))
        for row, muscle in enumerate(muscles):
            for col, value in enumerate((muscle['name'], f"{muscle['csa_mm2']:.1f}",
                                         f"{muscle['fat_fraction'] * 100:.1f}")):
                self.muscle_table.setItem(row, col, QTableWidgetItem(value))
        sarc = result.get('sarcopenia') or {}
        asymmetry = ', '.join(f"{key.removesuffix('_asymmetry_pct')}: {value:.1f}%"
                              for key, value in result.get('asymmetry', {}).items())
        self.pmi_label.setText(f"Psoas area: {sarc.get('total_psoas_area_cm2', 0):.1f} cm²\nL/R asymmetry: {asymmetry or 'not assessed'}")
        self.pmi_label.setWordWrap(True)
        l3_index = self.detected_levels.get('L3_body', {}).get('axial_slice_idx')
        pmi = sarc.get('pmi_cm2_per_m2') if self.current_slice_idx == l3_index else None
        self.risk_label.setText(f'L3 PMI: {pmi:.2f} cm²/m² · research index' if pmi is not None
                               else 'L3 PMI: not assessed · requires L3 coverage and height')

    def _restore_muscle_results(self, saved):
        restored = {}
        for key, value in saved.get('muscle_frames', {}).items():
            try:
                index = int(key)
                if not 0 <= index < len(self.axial_slices):
                    continue
                frame = self.axial_slices[index]
                if json.dumps(value.get('source_frame'), sort_keys=True) != json.dumps(frame_identity(frame), sort_keys=True):
                    continue
                source = nib.load(value['slice_path']); mask = nib.load(value['muscle_mask_path'])
                expected = np.asarray(frame['pixel_array'], dtype=np.float32).T[:, :, None]
                if (source.shape != expected.shape or source.shape != mask.shape
                        or not np.allclose(source.affine, native_slice_affine(frame), atol=1e-4)
                        or not np.allclose(source.affine, mask.affine, atol=1e-4)
                        or not np.array_equal(np.asarray(source.dataobj), expected)):
                    continue
                labels = np.asarray(mask.dataobj)
                if not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)) or not set(np.unique(labels)).issubset(set(range(9))):
                    continue
                restored[index] = restrict_l3_indices(
                    dict(value, image_array=np.asarray(source.dataobj).squeeze(), segmentation_mask=labels.squeeze()),
                    index == self.detected_levels.get('L3_body', {}).get('axial_slice_idx'))
            except (OSError, ValueError, KeyError, TypeError, nib.filebasedimages.ImageFileError):
                continue
        self.muscle_results = restored
        self.last_result['muscle_frames'] = {str(index): serializable_result(value)
                                            for index, value in restored.items()}
        self.last_result['muscle_errors'] = {}
        self.last_result['muscle_assessment_status'] = 'research_only' if restored else 'not_assessed'
        self.last_result['sarcopenia_diagnosis_status'] = 'not_assessed'
        self.last_result['analysis_mode'] = 'canal_and_muscles' if restored else 'canal_only'
        if saved.get('multi_level_measurements'):
            self.multi_level_result = json.loads(json.dumps(saved['multi_level_measurements']))
            for level in self.multi_level_result.get('levels', {}).values():
                muscle = restored.get(level.get('axial_slice_idx'))
                level.update(muscles=muscle.get('muscles', []) if muscle else [],
                             asymmetry=muscle.get('asymmetry', {}) if muscle else {},
                             muscle_assessment_status='research_only' if muscle else 'not_assessed')
                level.pop('muscle_mask_path', None)
                if muscle:
                    level['muscle_mask_path'] = muscle['muscle_mask_path']
            l3_index = self.detected_levels.get('L3_body', {}).get('axial_slice_idx')
            l3 = restored.get(l3_index)
            self.multi_level_result['sarcopenia'] = {
                'level_used': 'L3_body', 'axial_slice_idx': l3_index,
                'result': l3.get('sarcopenia') if l3 else None,
                'note': 'Cached research indices from validated native frame.' if l3 else 'L3 indices not assessed.'}
            self.multi_level_result['analysis_mode'] = 'canal_and_muscles' if restored else 'canal_only'
            self.multi_level_result['muscle_assessment_status'] = 'research_only' if restored else 'not_assessed'
            self.last_result['multi_level_measurements'] = self.multi_level_result
            self._show_level_measurements()
        if restored:
            self.slice_slider.setValue(next(iter(restored)))
            self._refresh_axial_with_canal_overlay()
        self._persist_findings()

    def _on_save_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save annotated screenshot', str(self.work_dir / 'spinosarc-demo.png'), 'PNG (*.png)')
        if filename:
            self.grab().save(filename); self.status_label.setText(f'PNG saved: {filename}')

    def _persist_findings(self):
        (self.analysis_output / 'findings.json').write_text(json.dumps(self.last_result, indent=2, allow_nan=False) + '\n')

    def _on_export_excel(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Export findings', str(self.work_dir / 'spinosarc-findings.json'), 'JSON (*.json)')
        if filename:
            Path(filename).write_text(json.dumps(self.last_result, indent=2, allow_nan=False) + '\n')
            self.status_label.setText(f'JSON saved: {filename}')

    def closeEvent(self, event):
        if self.muscle_worker and self.muscle_worker.isRunning():
            self.muscle_worker.requestInterruption()
            self.status_label.setText('Finishing the current muscle slice; close again after it stops.')
            event.ignore(); return
        if self.muscle_loader and self.muscle_loader.isRunning():
            self.status_label.setText('Wait for the muscle model to finish loading before closing.')
            event.ignore(); return
        if self.worker and self.worker.isRunning():
            if hasattr(self.worker.runner, 'cancel'):
                self.worker.runner.cancel()
            self.status_label.setText('Stopping segmentation; close again after it stops.')
            event.ignore(); return
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = SpinoSarcDemoWindow()
    window.show(); window.raise_(); window.activateWindow()
    if '--example' in sys.argv:
        QTimer.singleShot(0, window.load_open_example)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
