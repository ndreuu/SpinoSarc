#!/usr/bin/env python3
"""Headless geometry/state smoke; run with .venv-spinosarc/bin/python.

Synthetic volumes live in TemporaryDirectory. No native window, model download,
or inference is started. If available, the open SPIDER sample is also checked.
"""
from pathlib import Path
from _demo_paths import REPO_ROOT, runtime_root
from tempfile import TemporaryDirectory
import argparse
import io
import json
import os
import sys
import unittest
from unittest.mock import patch
from dataclasses import asdict


ROOT = runtime_root()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import QApplication
from spinosarc_app.analyzer import SpinoSarcAnalyzer, Demographics
from spinosarc_app.demo_io import prepare_volumes, write_native_slice_nifti
from spinosarc_app.lumbar_demo import SpinoSarcDemoWindow, MuscleAnalysisWorker, serializable_result
from spinosarc_app.totalspineseg.canal_csa import resample_canal_to_axial_slice
from spinosarc_app.totalspineseg.multi_level_analyzer import MultiLevelAnalyzer
from spinosarc_app.totalspineseg.level_mapper import LevelMapper


class SpinoSarcSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['spinosarc-headless-smoke'])

    def setUp(self):
        self.temp = TemporaryDirectory(prefix='spinosarc-smoke-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        environment = patch.dict(os.environ, {'SPINOSARC_ENABLE_MUSCLES': '0',
                                               'SPINOSARC_AXIAL_EXAMPLE_MANIFEST': ''})
        environment.start()
        self.addCleanup(environment.stop)
        os.environ['SPINOSARC_WORK_DIR'] = str(self.directory / 'gui')
        # GUI tests always use a temporary sagittal fixture, independent of
        # downloaded cases. One separate test covers SPIDER when available.
        sagittal = self.save('sag_demo.nii.gz', np.zeros((5, 14, 12), np.float32),
                             np.diag([5., .8, 1.2, 1.]))
        os.environ['SPINOSARC_DEMO_MRI'] = str(sagittal)

    def save(self, name, array, affine):
        path = self.directory / name
        image = nib.Nifti1Image(array, affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, path)
        return path

    def axial_fixture(self):
        data = np.arange(8 * 10 * 4, dtype=np.float32).reshape(8, 10, 4)
        affine = np.diag([2., 1.5, -5., 1.])
        affine[:3, 3] = [11., -30., 45.]
        path = self.save('ax_native.nii.gz', data, affine)
        return data, affine, prepare_volumes([path], self.directory / 'axial')

    def test_native_axial_physical_landmarks_and_original_frames(self):
        data, affine, loaded = self.axial_fixture()
        frames = loaded['axial_slices']
        self.assertEqual([frame['native_frame_index'] for frame in frames], [3, 2, 1, 0])
        for frame in frames:
            self.assertEqual(frame['native_frame_axis'], 2)
            ps = np.asarray(frame['pixel_spacing'])
            iop = np.asarray(frame['image_orientation'])
            ipp = np.asarray(frame['image_position'])
            for row, col in [(0, 0), (3, 4), (9, 7)]:
                world_lps = ipp + col * ps[1] * iop[:3] + row * ps[0] * iop[3:]
                world_ras = np.diag([-1., -1., 1.]) @ world_lps
                source = (np.linalg.inv(affine) @ np.r_[world_ras, 1])[:3]
                voxel = np.rint(source).astype(int)
                np.testing.assert_allclose(source, voxel, atol=1e-5)
                self.assertEqual(frame['pixel_array'][row, col], data[tuple(voxel)])
                self.assertEqual(voxel[frame['native_frame_axis']], frame['native_frame_index'])

    def test_permuted_source_frame_axis(self):
        data, _, loaded = self.axial_fixture()
        affine = np.array([[0, 2, 0, 11], [0, 0, 1.5, -30],
                           [-5, 0, 0, 45], [0, 0, 0, 1.]])
        path = self.save('ax_permuted.nii.gz', data.transpose(2, 0, 1), affine)
        frames = prepare_volumes([path], self.directory / 'permuted')['axial_slices']
        self.assertEqual([frame['native_frame_index'] for frame in frames], [3, 2, 1, 0])
        self.assertTrue(all(frame['native_frame_axis'] == 0 for frame in frames))
        for original, permuted in zip(loaded['axial_slices'], frames):
            np.testing.assert_array_equal(original['pixel_array'], permuted['pixel_array'])
            np.testing.assert_allclose(original['image_position'], permuted['image_position'])

    def test_canal_area_36_mm2_without_muscle_inference(self):
        data, affine, loaded = self.axial_fixture()
        frame = loaded['axial_slices'][1]
        expected = np.zeros(frame['pixel_array'].shape, dtype=bool)
        expected[2:5, 3:7] = True
        canal = np.zeros(data.shape, dtype=np.float32)
        ps = np.asarray(frame['pixel_spacing'])
        iop = np.asarray(frame['image_orientation'])
        ipp = np.asarray(frame['image_position'])
        for row, col in np.argwhere(expected):
            lps = ipp + col * ps[1] * iop[:3] + row * ps[0] * iop[3:]
            ras = np.diag([-1., -1., 1.]) @ lps
            voxel = np.rint((np.linalg.inv(affine) @ np.r_[ras, 1])[:3]).astype(int)
            canal[tuple(voxel)] = 1
        path = self.save('canal.nii.gz', canal, affine)
        np.testing.assert_array_equal(resample_canal_to_axial_slice(str(path), frame), expected)
        analyzer = SpinoSarcAnalyzer(canal_only=True)
        self.assertIsNone(analyzer.engine)
        self.assertNotIn('spinosarc_app.inference_engine', sys.modules)

        def forbidden_muscle_callback(_):
            self.fail('Canal-only analysis called the muscle inference callback')

        levels = {'L4-L5': {'axial_slice_idx': 1, 'type': 'IVD',
                            'world_xyz': (0, 0, frame['image_position'][2])}}
        result = MultiLevelAnalyzer(analyzer, loaded['axial_slices'], str(path)).analyze_all(
            levels, forbidden_muscle_callback)
        self.assertEqual(result['levels']['L4-L5']['canal_csa_mm2'], 36.)
        self.assertEqual(result['levels']['L4-L5']['muscles'], [])
        self.assertIsNone(result['levels']['L4-L5']['stenosis'])
        self.assertEqual(result['muscle_assessment_status'], 'not_assessed')
        self.assertIsNone(result['sarcopenia'])
        # No nearby-slice substitution for an uncovered level in canal-only mode.
        gap = {'L4-L5': {'axial_slice_idx': None, 'type': 'IVD', 'world_xyz': (0, 0, 500)}}
        missing = MultiLevelAnalyzer(analyzer, loaded['axial_slices'], str(path)).analyze_all(
            gap, forbidden_muscle_callback)
        self.assertIsNone(missing['levels']['L4-L5']['canal_csa_mm2'])
        single = self.save('slice.nii.gz', np.ones((4, 5, 1), np.float32),
                           np.diag([2, 1.5, 1, 1]))
        summary = analyzer.analyze(str(single))
        self.assertEqual(summary['muscles'], [])
        self.assertIsNone(summary['sarcopenia'])
        self.assertFalse(summary['segmentation_mask'].any())
        self.assertEqual(summary['segmentation_mask_kind'], 'empty_muscle_display_mask')

    def test_sagittal_physical_reorientation_without_interpolation(self):
        data = np.arange(9 * 3 * 7, dtype=np.float32).reshape(9, 3, 7)
        affine = np.array([[0, 4, 0, 12], [0, 0, -.8, 9],
                           [1.2, 0, 0, 20], [0, 0, 0, 1.]])
        path = self.save('sag_native.nii.gz', data, affine)
        loaded = prepare_volumes([path], self.directory / 'sagittal')
        self.assertEqual(nib.aff2axcodes(loaded['sagittal_affine']), ('A', 'S', 'R'))
        self.assertEqual(loaded['axial_slices'], [])
        self.assertEqual(loaded['sources'][0]['interpolation'], 'none')
        for index in [(0, 0, 0), (6, 8, 2), (2, 4, 1)]:
            world = loaded['sagittal_affine'] @ np.r_[index, 1]
            source = np.rint((np.linalg.inv(affine) @ world)[:3]).astype(int)
            self.assertEqual(loaded['sagittal_data'][index], data[tuple(source)])

    def test_oblique_axial_preserves_native_planes_and_inplane_rotation(self):
        theta = np.deg2rad(15)
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([1, 1, 5])
        data = np.zeros((16, 64, 4), np.float32)
        path = self.save('ax_oblique.nii.gz', data, affine)
        loaded = prepare_volumes([path], self.directory / 'oblique')
        self.assertEqual(len(loaded['axial_slices']), 4)
        frame = loaded['axial_slices'][1]
        native = nib.load(write_native_slice_nifti(frame, self.directory / 'one_plane.nii.gz'))
        np.testing.assert_array_equal(native.get_fdata()[:, :, 0].T, frame['pixel_array'])
        for row, col in [(0, 0), (10, 5), (63, 15)]:
            world = native.affine @ [col, row, 0, 1]
            source = np.rint((np.linalg.inv(affine) @ world)[:3]).astype(int)
            self.assertEqual(data[tuple(source)], frame['pixel_array'][row, col])
        affine[:3, :3] = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ np.diag([1, 1, 5])
        path = self.save('ax_inplane.nii.gz', data, affine)
        self.assertEqual(len(prepare_volumes([path], self.directory / 'inplane')['axial_slices']), 4)

    def dicom_fixture(self, folder, positions, orientations, study_uid=None, frame_uid=None):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
        folder.mkdir()
        series_uid = generate_uid()
        study_uid = study_uid or generate_uid()
        frame_uid = frame_uid or generate_uid()
        records = []
        for index, (position, orientation) in enumerate(zip(positions, orientations)):
            filename = folder / f'{index:03d}.ima'
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(filename), {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Modality = 'MR'
            ds.SeriesInstanceUID = series_uid
            ds.StudyInstanceUID = study_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.InstanceNumber = index + 1
            ds.ImagePositionPatient = list(map(float, position))
            ds.ImageOrientationPatient = list(map(float, orientation))
            ds.PixelSpacing = [1.5, 2.]
            ds.SliceThickness = 3.
            ds.Rows, ds.Columns = 10, 8
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            array = (np.arange(80).reshape(10, 8) + index * 100).astype('<i2')
            ds.PixelData = array.tobytes()
            ds.save_as(filename, enforce_file_format=True)
            records.append((ds, array))
        return records

    def test_classic_dicom_variable_oblique_planes_and_exact_slice_nifti(self):
        theta = np.deg2rad(20.)
        row_dir = [0., np.cos(theta), np.sin(theta)]
        orientation = [1., 0., 0.] + row_dir
        normal = np.cross(orientation[:3], orientation[3:])
        origin = np.array([10., -20., 35.])
        positions = [origin + normal * gap for gap in (0., 3.6, 40.)]
        orientations = [orientation, orientation, [1., 0., 0., 0., 1., 0.]]
        folder = self.directory / 'T2_TRA'
        records = self.dicom_fixture(folder, positions, orientations)
        loaded = prepare_volumes([folder], self.directory / 'dicom_work')
        frames = loaded['axial_slices']
        self.assertIsNone(loaded['sagittal_nifti_path'])
        self.assertEqual(len(frames), 3)
        self.assertEqual(loaded['sources'][0]['interpolation'], 'none')
        for frame, (ds, pixels) in zip(frames, records):
            self.assertEqual(frame['sop_instance_uid'], str(ds.SOPInstanceUID))
            self.assertEqual(frame['native_frame_index'], 0)
            self.assertIsNone(frame['native_frame_axis'])
            np.testing.assert_array_equal(frame['pixel_array'], pixels)
            native = nib.load(write_native_slice_nifti(frame, self.directory / f'{frame["instance_number"]}.nii.gz'))
            np.testing.assert_array_equal(native.get_fdata()[:, :, 0].T, pixels)
            for row, col in [(0, 0), (3, 4), (9, 7)]:
                iop = np.asarray(frame['image_orientation'])
                lps = np.asarray(frame['image_position']) + col * 2. * iop[:3] + row * 1.5 * iop[3:]
                np.testing.assert_allclose(native.affine @ [col, row, 0, 1], np.r_[lps * [-1., -1., 1.], 1.], atol=1e-5)

    def test_native_plane_level_mapping_ras_lps_gaps_and_fov(self):
        theta = np.deg2rad(20.)
        col_dir = np.array([1., 0., 0.])
        row_dir = np.array([0., np.cos(theta), np.sin(theta)])
        normal = np.cross(col_dir, row_dir)
        origin = np.array([10., -20., 35.])
        frames = [dict(pixel_array=np.zeros((40, 40)), image_position=origin + normal * gap,
                       image_orientation=np.r_[col_dir, row_dir], pixel_spacing=(1., 1.), slice_thickness=3.)
                  for gap in (0., 3.6, 7.2, 40., 43.6)]
        point = origin + 5. * col_dir + 20. * row_dir
        # The point's Z almost matches frame 2's corner, yet lies in frame 0.
        self.assertEqual(int(np.argmin([abs(frame['image_position'][2] - point[2]) for frame in frames])), 2)
        levels = {'correct': {'world_xyz': point * [-1., -1., 1.]},
                  'lps': {'world_xyz': point, 'world_coordinate_system': 'LPS'},
                  'gap': {'world_xyz': (point + 20. * normal) * [-1., -1., 1.]},
                  'outside': {'world_xyz': (point + 100. * col_dir) * [-1., -1., 1.]}}
        mapped = LevelMapper().map_to_axial(levels, frames)
        self.assertEqual(mapped['correct']['axial_slice_idx'], 0)
        self.assertEqual(mapped['lps']['axial_slice_idx'], 0)
        self.assertAlmostEqual(mapped['correct']['axial_distance_mm'], 0.)
        np.testing.assert_allclose(mapped['correct']['axial_pixel_row_col'], [20., 5.])
        self.assertIsNone(mapped['gap']['axial_slice_idx'])
        self.assertIn('gap', mapped['gap']['out_of_range_reason'])
        self.assertIsNone(mapped['outside']['axial_slice_idx'])
        self.assertIn('field of view', mapped['outside']['out_of_range_reason'])

    def test_dicom_sagittal_native_stack_and_different_frame_uid_warning(self):
        from pydicom.uid import generate_uid
        study = generate_uid()
        orientation = [0., 1., 0., 0., 0., -1.]
        origin = np.array([100., 20., 30.])
        normal = np.cross(orientation[:3], orientation[3:])
        sag = self.directory / 'T2_SAG'
        records = self.dicom_fixture(sag, [origin + normal * index * 4.8 for index in range(3)],
                                     [orientation] * 3, study_uid=study)
        axial = self.directory / 'T2_AX'
        self.dicom_fixture(axial, [[10., 20., 30.]], [[1., 0., 0., 0., 1., 0.]], study_uid=study)
        loaded = prepare_volumes([sag, axial], self.directory / 'paired')
        self.assertEqual(nib.aff2axcodes(loaded['sagittal_affine']), ('A', 'S', 'R'))
        self.assertEqual(loaded['registration_status'], 'needs_review')
        self.assertEqual(len(loaded['geometry_warnings']), 1)
        native = nib.load(loaded['sagittal_nifti_path'])
        for ds, pixels in records:
            for row, col in [(0, 0), (3, 4), (9, 7)]:
                lps = np.asarray(ds.ImagePositionPatient) + col * 2. * np.asarray(orientation[:3]) + row * 1.5 * np.asarray(orientation[3:])
                voxel = np.rint((np.linalg.inv(native.affine) @ np.r_[lps * [-1., -1., 1.], 1.])[:3]).astype(int)
                self.assertEqual(native.get_fdata()[tuple(voxel)], pixels[row, col])

    def test_open_spider_is_native_sagittal_without_axial_fabrication(self):
        source = ROOT / 'data/spider/images/246_t2.mha'
        if not source.is_file():
            self.skipTest('Optional downloaded SPIDER example is unavailable; synthetic geometry tests still run.')
        loaded = prepare_volumes([source], self.directory / 'spider')
        self.assertEqual(loaded['sagittal_data'].shape, (384, 277, 18))
        self.assertEqual(loaded['axial_slices'], [])
        self.assertEqual(loaded['sources'][0]['plane'], 'sagittal')

    def test_oblique_sagittal_level_line_projects_full_world_marker(self):
        theta = np.deg2rad(20)
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])
        affine = np.array([[0, 4, 0, 12], [0, 0, -.8, 9],
                           [1.2, 0, 0, 20], [0, 0, 0, 1.]])
        affine[:3, :3] = rotation @ affine[:3, :3]
        path = self.save('sag_oblique.nii.gz', np.zeros((9, 3, 7), np.float32), affine)
        window = SpinoSarcDemoWindow()
        self.addCleanup(window.close)
        window.load_files([path])
        self.assertEqual(nib.aff2axcodes(window.sagittal_affine), ('A', 'S', 'R'))
        marker_voxel = np.array([4., 3., 1., 1.])
        world = (window.sagittal_affine @ marker_voxel)[:3]
        window.detected_levels = {
            'L4-L5': {'world_xyz': tuple(world), 'type': 'IVD', 'axial_slice_idx': None},
            'L3_body': {'world_xyz': tuple(world), 'type': 'VB', 'axial_slice_idx': 1},
        }
        window._update_sagittal_level_overlay()
        lines = window.sagittal_display._level_lines
        self.assertEqual(len(lines), 1)  # Only the five lumbar IVD labels are drawn.
        self.assertEqual(lines[0]['label'], 'L4-L5')
        expected = 1. - marker_voxel[1] / (window.sagittal_data.shape[1] - 1)
        self.assertAlmostEqual(lines[0]['y_frac'], expected, places=10)
        self.assertEqual(lines[0]['color_rgb'], (214, 39, 40))
        self.assertTrue(lines[0]['dashed'])
        # This fixture must expose the previous world-Z-only formula's error.
        old_si = (world[2] - window.sagittal_affine[2, 3]) / window.sagittal_affine[2, 1]
        self.assertGreater(abs(old_si - marker_voxel[1]), .5)
        window.detected_levels['L4-L5']['axial_slice_idx'] = 1
        window._update_sagittal_level_overlay()
        self.assertFalse(window.sagittal_display._level_lines[0]['dashed'])

    def test_gui_new_case_clears_previous_case_and_blocks_reset_during_worker(self):
        window = SpinoSarcDemoWindow()
        self.addCleanup(window.close)
        window.load_files([os.environ['SPINOSARC_DEMO_MRI']])
        self.assertTrue(window.detect_levels_btn.isEnabled())
        self.assertFalse(window.analyze_btn.isEnabled())
        self.assertFalse(window.analyze_all_btn.isEnabled())
        window.detected_levels = {'L4-L5': {'type': 'IVD'}}
        window.canal_nifti_path = 'previous-canal.nii.gz'
        window.sagittal_canal_mask = np.ones(window.sagittal_data.shape, bool)
        window.multi_level_result = {'previous': True}
        window.last_result = {'previous': True}
        window.levels_list.addItem('Previous level')
        for button in (window.analyze_btn, window.analyze_all_btn,
                       window.save_pdf_btn, window.export_excel_btn):
            button.setEnabled(True)
        # A running worker must retain its original case until it completes.
        class RunningWorker:
            def isRunning(self):
                return True
        window.worker = RunningWorker()
        window._on_new_case()
        self.assertEqual(window.canal_nifti_path, 'previous-canal.nii.gz')
        self.assertIn('Stop segmentation', window.status_label.text())
        window.worker = None
        # Results queued after QThread returns must keep the source case alive.
        for pending_field in ('segmentation_pending', 'muscle_pending'):
            setattr(window, pending_field, True)
            window._on_new_case()
            self.assertEqual(window.canal_nifti_path, 'previous-canal.nii.gz')
            self.assertTrue(window._analysis_running())
            setattr(window, pending_field, False)
        window.registration_status = 'needs_review'
        window.geometry_warnings = ['Previous case alignment warning']
        window._on_new_case()
        self.assertEqual(window.sources, [])
        self.assertEqual(window.axial_slices, [])
        self.assertEqual(window.detected_levels, {})
        self.assertEqual(window.registration_status, 'not_assessed')
        self.assertEqual(window.geometry_warnings, [])
        for field in ('canal_nifti_path', 'sagittal_nifti_path', 'sagittal_canal_mask',
                      'sagittal_data', 'sagittal_affine', 'axial_data', 'axial_affine',
                      'multi_level_result', 'last_result'):
            self.assertIsNone(getattr(window, field), field)
        self.assertEqual(window.levels_list.count(), 0)
        self.assertEqual(window.axial_display._level_lines, [])
        self.assertEqual(window.sagittal_display._level_lines, [])
        self.assertIsNone(window.axial_display._img)
        self.assertIsNone(window.sagittal_display._img)
        for widget in (window.detect_levels_btn, window.analyze_btn, window.analyze_all_btn,
                       window.save_pdf_btn, window.export_excel_btn, window.new_case_btn,
                       window.slice_slider, window.sag_slider):
            self.assertFalse(widget.isEnabled())
        self.assertNotIn('spinosarc_app.inference_engine', sys.modules)

    def muscle_worker_fixture(self, frames, indices, demographics=None, cache=None):
        """Exercise real worker/native NIfTI I/O with a deterministic fake model."""
        analyzer = SpinoSarcAnalyzer(canal_only=True)
        calls = []

        def predict(source, demo):
            calls.append(source)
            image = nib.load(source)
            pixels = image.get_fdata()[:, :, 0]
            # Each of eight labels has ten pixels on the asymmetric 8x10 grid.
            mask = (np.arange(pixels.size).reshape(pixels.shape) // 10 + 1).astype(np.int16)
            pixel_area = float(np.prod(image.header.get_zooms()[:2]))
            muscles = analyzer._compute_muscle_metrics(pixels, mask, pixel_area)
            return dict(slice_path=source, image_array=pixels, segmentation_mask=mask,
                        pixel_spacing_mm=list(map(float, image.header.get_zooms()[:2])),
                        pixel_area_mm2=pixel_area, muscles=[muscle.to_dict() for muscle in muscles],
                        asymmetry=analyzer._compute_asymmetry(muscles),
                        demographics=asdict(demo) if demo else None,
                        sarcopenia=asdict(analyzer._compute_sarcopenia(muscles, demo)))

        analyzer.analyze = predict
        worker = MuscleAnalysisWorker(analyzer, frames, indices, demographics,
                                      self.directory / 'muscles', cache or {})
        output = []
        worker.result_ready.connect(output.append)
        worker.run()  # No thread/model/GPU process; signals deliver synchronously.
        self.assertEqual(len(output), 1)
        return worker, output[0], calls

    def test_muscle_worker_native_mask_alignment_and_demographic_cache_reuse(self):
        _, _, loaded = self.axial_fixture()
        frames = loaded['axial_slices']
        worker, output, calls = self.muscle_worker_fixture(frames, [1, 1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(output['errors'], {})
        self.assertFalse(output['cancelled'])
        self.assertEqual(set(output['frames']), {1})
        result = output['frames'][1]
        np.testing.assert_array_equal(result['image_array'].T, frames[1]['pixel_array'])
        source = nib.load(result['slice_path'])
        mask = nib.load(result['muscle_mask_path'])
        self.assertEqual(source.shape, mask.shape)
        np.testing.assert_allclose(source.affine, mask.affine)
        np.testing.assert_array_equal(mask.get_fdata()[:, :, 0], result['segmentation_mask'])
        self.assertEqual(result['source_frame']['native_frame_index'], frames[1]['native_frame_index'])
        self.assertEqual(len(result['muscles']), 8)
        self.assertTrue(all(muscle['csa_mm2'] == 30. for muscle in result['muscles']))
        self.assertIsNone(result['sarcopenia']['pmi_cm2_per_m2'])
        changed_demographics = Demographics(sex='M', height_cm=170.)
        _, reused, new_calls = self.muscle_worker_fixture(frames, [1], changed_demographics, output['frames'])
        self.assertEqual(new_calls, [])
        self.assertEqual(reused['errors'], {})
        self.assertEqual(reused['frames'][1]['demographics']['height_cm'], 170.)
        self.assertAlmostEqual(reused['frames'][1]['sarcopenia']['pmi_cm2_per_m2'], round(.6 / 1.7 ** 2, 2))
        self.assertIsNone(result['sarcopenia']['pmi_cm2_per_m2'])  # Original cache remains unchanged.

    def test_muscle_results_native_overlay_and_no_uncovered_l3_substitution(self):
        _, _, loaded = self.axial_fixture()
        _, output, _ = self.muscle_worker_fixture(loaded['axial_slices'], [1], Demographics(sex='M', height_cm=170.))
        window = SpinoSarcDemoWindow()
        self.addCleanup(window.close)
        window.load_files([loaded['sources'][0]['path']])
        window.muscles_requested = True  # Use supplied fake results, never load weights.
        window.analysis_output = self.directory / 'analysis'
        window.analysis_output.mkdir()
        window.last_result = {'sources': window.sources}
        window.detected_levels = {'L4-L5': {'axial_slice_idx': 1}, 'L3_body': {'axial_slice_idx': None}}
        window.multi_level_result = {'levels': {
            'L4-L5': {'covered': True, 'axial_slice_idx': 1, 'muscles': [], 'canal_csa_mm2': None},
            'L5-S': {'covered': False, 'axial_slice_idx': None, 'muscles': [], 'canal_csa_mm2': None}}}
        window._muscle_analysis_finished(output)
        self.assertEqual(window.current_slice_idx, 1)
        np.testing.assert_array_equal(window.axial_display._img, loaded['axial_slices'][1]['pixel_array'])
        np.testing.assert_array_equal(window.axial_display._mask, output['frames'][1]['segmentation_mask'].T)
        self.assertEqual(window.muscle_table.rowCount(), 8)
        self.assertEqual(len(window.multi_level_result['levels']['L4-L5']['muscles']), 8)
        self.assertEqual(window.multi_level_result['levels']['L5-S']['muscles'], [])
        self.assertIsNone(window.multi_level_result['sarcopenia']['result'])
        self.assertIsNone(window.multi_level_result['sarcopenia']['axial_slice_idx'])
        self.assertIn('not covered', window.multi_level_result['sarcopenia']['note'])
        self.assertIn('not assessed', window.risk_label.text())
        window.slice_slider.setValue(0)
        self.assertIsNone(window.axial_display._mask)
        self.assertEqual(window.muscle_table.rowCount(), 0)
        window._on_new_case()
        self.assertEqual(window.muscle_results, {})
        self.assertIsNone(window.last_result)

    def test_muscle_cache_rejects_foreign_native_grid_and_clears_stale_level_metrics(self):
        _, _, loaded = self.axial_fixture()
        _, output, _ = self.muscle_worker_fixture(loaded['axial_slices'], [1])
        good = serializable_result(output['frames'][1])
        # Both foreign files agree with each other but neither belongs to frame2.
        foreign = dict(good, source_frame=dict(good['source_frame']))
        foreign['source_frame']['native_frame_index'] = loaded['axial_slices'][2]['native_frame_index']
        window = SpinoSarcDemoWindow()
        self.addCleanup(window.close)
        window.load_files([loaded['sources'][0]['path']])
        window.analysis_output = self.directory / 'cache-analysis'
        window.analysis_output.mkdir()
        window.last_result = {'sources': window.sources}
        window.detected_levels = {'L4-L5': {'axial_slice_idx': 1},
                                  'L5-S': {'axial_slice_idx': 2}, 'L3_body': {'axial_slice_idx': None}}
        saved = {'muscle_frames': {'1': good, '2': foreign, '99': good},
                 'muscle_assessment_status': 'research_only',
                 'multi_level_measurements': {'levels': {
                     'L4-L5': {'axial_slice_idx': 1, 'muscles': good['muscles'], 'canal_csa_mm2': None},
                     'L5-S': {'axial_slice_idx': 2, 'muscles': foreign['muscles'], 'canal_csa_mm2': None}},
                     'sarcopenia': {'axial_slice_idx': 2, 'result': {'pmi_cm2_per_m2': 999.}}}}
        window._restore_muscle_results(saved)
        self.assertEqual(set(window.muscle_results), {1})
        self.assertEqual(set(window.last_result.get('muscle_frames', {})), {'1'})
        self.assertEqual(len(window.multi_level_result['levels']['L4-L5']['muscles']), 8)
        self.assertEqual(window.multi_level_result['levels']['L5-S'].get('muscles', []), [])
        self.assertIsNone(window.multi_level_result.get('sarcopenia', {}).get('result'))
        # If all masks are missing or foreign, cached muscle results vanish.
        window._restore_muscle_results(dict(saved, muscle_frames={'2': foreign}))
        self.assertEqual(window.muscle_results, {})
        self.assertEqual(window.last_result.get('muscle_frames', {}), {})
        self.assertTrue(all(not level.get('muscles') for level in window.multi_level_result['levels'].values()))
        truncated = self.directory / 'truncated-cache.nii.gz'
        truncated.write_bytes(b'not a NIfTI image')
        corrupt = dict(good, slice_path=str(truncated))
        window._restore_muscle_results(dict(saved, muscle_frames={'1': corrupt}))
        self.assertEqual(window.muscle_results, {})
        self.assertEqual(window.last_result.get('muscle_frames', {}), {})

    def test_saved_segmentation_matches_exact_case_and_restores_without_worker(self):
        path = self.save('sag_cache_case.nii.gz', np.zeros((4, 12, 10), np.float32),
                         np.diag([-4., -1., 1.5, 1.]))
        os.environ['SPINOSARC_DEMO_MRI'] = str(path)
        window = SpinoSarcDemoWindow()
        self.addCleanup(window.close)
        window.load_open_example()
        self.assertIsNone(window.worker)
        self.assertIsNone(window.last_result)
        source = str(Path(window.sagittal_nifti_path).resolve())

        def save_result(folder, command, output=None):
            folder.mkdir(parents=True, exist_ok=True)
            result = {'success': True, 'output_dir': str(output or folder),
                      'duration_sec': .25, 'command': command}
            (folder / 'process-result.json').write_text(json.dumps(result) + '\n')
            return result

        analyses = window.work_dir / 'analyses'
        wrong_case = analyses / '002-wrong-case'
        save_result(wrong_case, ['totalspineseg', str(self.directory / 'different-sag.nii.gz'),
                                 str(wrong_case)])
        self.assertFalse(window.restore_latest_segmentation())
        self.assertIsNone(window.last_result)
        self.assertIsNone(window.sagittal_canal_mask)
        self.assertFalse(window.export_excel_btn.isEnabled())
        # Even an exact input cannot authorize outputs belonging to another run.
        wrong_output = analyses / '003-wrong-output'
        save_result(wrong_output, ['totalspineseg', source, str(wrong_output)],
                    output=analyses / 'unrelated-output')
        self.assertFalse(window.restore_latest_segmentation())
        self.assertIsNone(window.last_result)

        correct = analyses / '004-correct-case'
        for name in ('step1_levels', 'step1_canal'):
            (correct / name).mkdir(parents=True)
        compact = np.zeros(window.sagittal_data.shape, np.uint8)
        compact[6, 4, window.current_sag_idx] = 24  # TSS compact L4-L5 marker.
        canal = np.zeros(window.sagittal_data.shape, np.float32)
        canal[4:8, 2:7, :] = 1  # Known ROI, including the displayed sagittal plane.
        basename = Path(source).name
        self.save(correct / 'step1_levels' / basename, compact, window.sagittal_affine)
        self.save(correct / 'step1_canal' / basename, canal, window.sagittal_affine)
        save_result(correct, ['totalspineseg', source, str(correct), '--step1'])
        self.assertTrue(window.restore_latest_segmentation())
        self.assertIsNone(window.worker)
        self.assertEqual(window.analysis_output, correct)
        self.assertEqual(window.last_result['numbering_status'], 'needs_confirmation')
        self.assertEqual(window.last_result['diagnostic_assessment_status'], 'not_assessed')
        self.assertEqual(window.last_result['canal_area_assessment_status'], 'not_assessed')
        self.assertEqual(set(window.detected_levels), {'L4-L5'})
        np.testing.assert_array_equal(window.sagittal_canal_mask, canal > .5)
        np.testing.assert_array_equal(window.sagittal_display._canal_overlay,
                                      canal[:, :, window.current_sag_idx] > .5)
        self.assertEqual(window.sagittal_display._level_lines[0]['label'], 'L4-L5')
        self.assertTrue(window.sagittal_display._level_lines[0]['dashed'])
        self.assertTrue(window.save_pdf_btn.isEnabled())
        self.assertTrue(window.export_excel_btn.isEnabled())
        self.assertFalse(window.analyze_btn.isEnabled())
        self.assertFalse(window.analyze_all_btn.isEnabled())
        self.assertIn('numbering unconfirmed', window.levels_list.item(0).text())
        self.assertTrue((correct / 'findings.json').is_file())
        # The example button should restore the same valid cache after resetting.
        window.load_open_example()
        self.assertIsNone(window.worker)
        self.assertEqual(window.last_result['numbering_status'], 'needs_confirmation')
        self.assertTrue(window.export_excel_btn.isEnabled())
        self.assertIn('Loaded saved TSS segmentation', window.status_label.text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/spinosarc-gui-smoke.json')
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpinoSarcSmoke)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        'success': result.wasSuccessful(),
        'mode': 'headless_offscreen',
        'real_inference_run': False,
        'native_gui_opened': False,
        'tests_run': result.testsRun,
        'failures': [{'test': test.id(), 'error': error} for test, error in result.failures],
        'errors': [{'test': test.id(), 'error': error} for test, error in result.errors],
        'skipped': [{'test': test.id(), 'reason': reason} for test, reason in result.skipped],
        'checks': [name.removeprefix('test_') for name in sorted(dir(SpinoSarcSmoke))
                   if name.startswith('test_')],
        'known_roi_area_mm2': 36,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(stream.getvalue(), end='')
    print(f'Report: {args.output}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
