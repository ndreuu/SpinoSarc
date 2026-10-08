"""Native MRI loading with physical geometry; no invented axial series.

Classic DICOM axial slices remain independent, including variable oblique
orientations. Only a regular sagittal stack is assembled for TotalSpineSeg.
"""
from pathlib import Path
import hashlib
import numpy as np
import nibabel as nib
import SimpleITK as sitk


LPS_TO_RAS = np.diag([-1., -1., 1., 1.])


def _frame_geometry(frame):
    """Validate DICOM-style native plane geometry; return unit directions."""
    ipp = np.asarray(frame['image_position'], dtype=float)
    iop = np.asarray(frame['image_orientation'], dtype=float)
    ps = np.asarray(frame['pixel_spacing'], dtype=float)
    if (ipp.shape != (3,) or iop.shape != (6,) or ps.shape != (2,)
            or not np.isfinite(np.r_[ipp, iop, ps]).all() or not (ps > 0).all()):
        raise ValueError('Invalid native slice position, orientation or pixel spacing.')
    col_dir, row_dir = iop[:3], iop[3:]
    if (not np.allclose([np.linalg.norm(col_dir), np.linalg.norm(row_dir)], 1., atol=1e-3)
            or abs(np.dot(col_dir, row_dir)) > 1e-3):
        raise ValueError('Native slice directions must be orthonormal.')
    col_dir = col_dir / np.linalg.norm(col_dir)
    row_dir = row_dir / np.linalg.norm(row_dir)
    normal = np.cross(col_dir, row_dir)
    thickness = float(frame.get('slice_thickness', 0.))
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('Native slice thickness must be positive and finite.')
    return ipp, col_dir, row_dir, normal, ps, thickness


def native_slice_affine(frame):
    """Exact RAS affine for pixels indexed as (column, row, plane)."""
    ipp, col_dir, row_dir, normal, ps, thickness = _frame_geometry(frame)
    lps = np.eye(4)
    lps[:3, 0] = col_dir * ps[1]
    lps[:3, 1] = row_dir * ps[0]
    lps[:3, 2] = normal * thickness
    lps[:3, 3] = ipp
    return LPS_TO_RAS @ lps


def write_native_slice_nifti(frame, path):
    """Write one measured native plane for MuscleMap, without interpolation.

    The source row/column pixel array is transposed to NIfTI column/row axes.
    Analyzer masks must be transposed back when displayed on the native frame.
    A single-frame DICOM remains a single slice, never an invented volume.
    """
    pixels = np.asarray(frame['pixel_array'], dtype=np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError('Expected finite scalar native 2D pixels.')
    image = nib.Nifti1Image(pixels.T[:, :, None], native_slice_affine(frame))
    image.header.set_xyzt_units('mm')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    return str(path)


def _load_dicom_series(path):
    """Read exactly one selected classic MR series, preserving native frames."""
    import pydicom
    candidates = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if len(candidates) > 10000:
        raise ValueError('Select a single DICOM series folder below 10000 files.')
    headers = []
    for filename in candidates:
        try:
            ds = pydicom.dcmread(str(filename), stop_before_pixels=True)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue
        if getattr(ds, 'Modality', None) != 'MR' or not hasattr(ds, 'Rows'):
            continue
        if int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Enhanced/multiframe DICOM is not supported in this demo; use classic single-frame MR.')
        if not getattr(ds, 'SeriesInstanceUID', None):
            raise ValueError('DICOM SeriesInstanceUID is required.')
        headers.append((filename, ds))
    if not headers:
        raise ValueError('No classic MR DICOM slices found in the selected series folder.')
    uids = {str(ds.SeriesInstanceUID) for _, ds in headers}
    if len(uids) != 1:
        raise ValueError('Select one DICOM series folder; multiple SeriesInstanceUID values were found.')
    voxel_count = sum(int(ds.Rows) * int(ds.Columns) for _, ds in headers)
    if voxel_count > 100_000_000:
        raise ValueError('Expected a scalar MR series below 100M pixels.')
    frames = []
    for filename, header in headers:
        if int(getattr(header, 'SamplesPerPixel', 1)) != 1:
            raise ValueError('Only scalar grayscale MR DICOM is supported.')
        try:
            frame = {
                'image_position': list(map(float, header.ImagePositionPatient)),
                'image_orientation': list(map(float, header.ImageOrientationPatient)),
                'pixel_spacing': tuple(map(float, header.PixelSpacing)),
                'slice_thickness': float(header.SliceThickness),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f'DICOM slice lacks native physical geometry: {filename.name}') from exc
        _frame_geometry(frame)
        ds = pydicom.dcmread(str(filename))
        pixels = np.asarray(ds.pixel_array)
        if pixels.ndim != 2 or pixels.shape != (int(header.Rows), int(header.Columns)):
            raise ValueError('Expected one scalar 2D DICOM frame per file.')
        pixels = pixels.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1.)) + float(getattr(ds, 'RescaleIntercept', 0.))
        if not np.isfinite(pixels).all():
            raise ValueError('DICOM slice contains invalid rescaled intensities.')
        frame.update(pixel_array=pixels, source_kind='dicom', source_path=str(filename.resolve()),
                     source_file=filename.name, source_volume=str(path.resolve()),
                     native_frame_index=0, native_frame_axis=None,
                     instance_number=int(getattr(header, 'InstanceNumber', 0)),
                     sop_instance_uid=str(getattr(header, 'SOPInstanceUID', '')),
                     series_instance_uid=str(header.SeriesInstanceUID),
                     study_instance_uid=str(getattr(header, 'StudyInstanceUID', '')),
                     frame_of_reference_uid=str(getattr(header, 'FrameOfReferenceUID', '')),
                     z_world_mm=frame['image_position'][2], rows=pixels.shape[0], cols=pixels.shape[1])
        frames.append(frame)
    if len({f['sop_instance_uid'] for f in frames}) != len(frames) or any(not f['sop_instance_uid'] for f in frames):
        raise ValueError('DICOM slices require unique SOPInstanceUID values.')
    frames.sort(key=lambda f: (f['instance_number'], f['source_path']))
    planes = set()
    for frame in frames:
        normal = _frame_geometry(frame)[3]
        planes.add({0: 'sagittal', 1: 'coronal', 2: 'axial'}[int(np.argmax(abs(normal)))])
    if len(planes) != 1 or 'coronal' in planes:
        raise ValueError('Select one sagittal or native axial DICOM series.')
    plane = planes.pop()
    provenance = dict(path=str(path.resolve()), source_kind='dicom', plane=plane,
                      interpolation='none', native_frame_count=len(frames),
                      series_instance_uid=frames[0]['series_instance_uid'],
                      study_instance_uid=frames[0]['study_instance_uid'],
                      frame_of_reference_uid=frames[0]['frame_of_reference_uid'],
                      native_sop_instance_uids=[f['sop_instance_uid'] for f in frames],
                      native_source_files=[dict(path=f['source_path'], sop_instance_uid=f['sop_instance_uid'],
                                                size_bytes=Path(f['source_path']).stat().st_size,
                                                mtime_ns=Path(f['source_path']).stat().st_mtime_ns)
                                           for f in frames])
    if plane == 'axial':
        return None, None, frames, provenance
    # A sagittal series must form a regular physical grid for TSS. Preserve
    # pixels exactly; reject varying angulation and nonuniform slice positions.
    if len(frames) < 2:
        raise ValueError('Sagittal DICOM needs at least two slices to form a native volume.')
    ipp, col_dir, row_dir, normal, ps, _ = _frame_geometry(frames[0])
    for frame in frames:
        _, cd, rd, _, spacing, _ = _frame_geometry(frame)
        if (not np.allclose(cd, col_dir, atol=1e-4) or not np.allclose(rd, row_dir, atol=1e-4)
                or not np.allclose(spacing, ps, atol=1e-4)
                or frame['pixel_array'].shape != frames[0]['pixel_array'].shape):
            raise ValueError('Sagittal DICOM must have a uniform native orientation, matrix and pixel spacing.')
    frames.sort(key=lambda f: float(np.dot(f['image_position'], normal)))
    positions = np.array([f['image_position'] for f in frames])
    distances = positions @ normal
    gaps = np.diff(distances)
    gap = float(np.median(gaps))
    if gap <= 0 or not np.allclose(gaps, gap, atol=max(.01, gap * .01), rtol=0):
        raise ValueError('Sagittal DICOM must have regular, distinct native slice positions.')
    expected = positions[0] + np.arange(len(frames))[:, None] * normal * gap
    if not np.allclose(positions, expected, atol=.02, rtol=0):
        raise ValueError('Sagittal DICOM slice positions do not form an orthogonal native stack.')
    lps = np.eye(4)
    lps[:3, 0], lps[:3, 1], lps[:3, 2] = col_dir * ps[1], row_dir * ps[0], normal * gap
    lps[:3, 3] = positions[0]
    array = np.stack([f['pixel_array'].T for f in frames], axis=2)
    provenance['native_spacing_mm'] = [float(ps[1]), float(ps[0]), gap]
    provenance['native_stack_position_tolerance_mm'] = .02
    provenance['native_sop_instance_uids'] = [f['sop_instance_uid'] for f in frames]
    return array, LPS_TO_RAS @ lps, [], provenance


def prepare_volumes(paths, work_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    result = dict(sagittal_nifti_path=None, sagittal_data=None,
                  sagittal_affine=None, axial_slices=[], sources=[],
                  registration_status='not_assessed', geometry_warnings=[])
    for filename in paths:
        path = Path(filename).resolve()
        if not path.exists():
            raise ValueError("Select a local MHA/NIfTI volume or one DICOM series folder.")
        dicom_provenance = None
        if path.is_dir() or path.suffix.lower() in ('.dcm', '.ima'):
            array, affine, frames, dicom_provenance = _load_dicom_series(path)
            if frames:
                if result['axial_slices']:
                    raise ValueError('Select only one axial series.')
                result['axial_slices'] = frames
                result['sources'].append(dicom_provenance)
                continue
        elif path.name.lower().endswith('.mha'):
            # Only embedded MHA; never follow a header's external data path.
            with path.open('rb') as stream:
                header = stream.read(16384).split(b'ElementDataFile', 1)
            if len(header) != 2 or header[1].splitlines()[0].strip() != b'= LOCAL':
                raise ValueError('MHA must contain embedded pixel data.')
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path)); reader.ReadImageInformation()
            if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1 or np.prod(reader.GetSize()) > 100_000_000:
                raise ValueError('Expected a scalar 3D volume below 100M voxels.')
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image).transpose(2, 1, 0).astype(np.float32)
            lps = np.eye(4)
            lps[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
            lps[:3, 3] = image.GetOrigin()
            affine = np.diag([-1, -1, 1, 1]) @ lps
        elif path.name.lower().endswith(('.nii', '.nii.gz')):
            image = nib.load(path)
            if len(image.shape) != 3 or np.prod(image.shape) > 100_000_000:
                raise ValueError('Expected a scalar 3D volume below 100M voxels.')
            unit = image.header.get_xyzt_units()[0]
            if unit not in ('mm', 'meter', 'micron'):
                raise ValueError('NIfTI spatial units must be known for measurements.')
            array = image.get_fdata(dtype=np.float32)
            affine = image.affine.copy()
            affine[:3, :] *= {'mm': 1, 'meter': 1000, 'micron': .001}[unit]
        else:
            raise ValueError('Supported volumes: .mha, .nii, .nii.gz')
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.isfinite(array).all() or not np.isfinite(affine).all() or not np.all(spacing > 0):
            raise ValueError('Volume contains invalid intensities or geometry.')
        direction = affine[:3, :3] / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-3):
            raise ValueError('Non-orthogonal source geometry is unsupported.')
        native_axis = int(np.argmax(spacing))
        if spacing.max() / np.median(spacing) >= 1.3:
            plane = {0: 'sagittal', 1: 'coronal', 2: 'axial'}[int(np.argmax(abs(direction[:, native_axis])))]
        elif 'sag' in path.name.lower():
            plane = 'sagittal'
        elif 'ax' in path.name.lower():
            plane = 'axial'
        else:
            raise ValueError('Ambiguous acquisition plane; use a sag/ax filename hint.')
        if plane == 'coronal':
            raise ValueError('Load sagittal T2 and optionally native axial T2.')
        current = nib.orientations.io_orientation(affine)
        target = nib.orientations.axcodes2ornt(('A', 'S', 'R') if plane == 'sagittal' else ('L', 'P', 'S'))
        transform = nib.orientations.ornt_transform(current, target)
        oriented = nib.orientations.apply_orientation(array, transform)
        source_voxel_transform = nib.orientations.inv_ornt_aff(transform, array.shape)
        new_affine = affine @ source_voxel_transform
        provenance = dicom_provenance or {'path': str(path), 'source_kind': 'mha' if path.suffix.lower() == '.mha' else 'nifti',
                                        'plane': plane, 'native_spacing_mm': spacing.tolist(), 'interpolation': 'none'}
        result['sources'].append(provenance)
        if plane == 'sagittal':
            if result['sagittal_nifti_path']:
                raise ValueError('Select only one sagittal series.')
            identity = str(path).encode() + str(path.stat().st_mtime_ns).encode()
            if dicom_provenance:
                identity += ('|'.join(dicom_provenance['native_sop_instance_uids'])).encode()
                for entry in dicom_provenance['native_source_files']:
                    identity += f"{entry['path']}|{entry['size_bytes']}|{entry['mtime_ns']}".encode()
            digest = hashlib.sha256(identity).hexdigest()[:12]
            prepared = work_dir / f'sagittal_{digest}.nii.gz'
            volume = nib.Nifti1Image(oriented, new_affine)
            volume.header.set_xyzt_units('mm'); nib.save(volume, prepared)
            result.update(sagittal_nifti_path=str(prepared), sagittal_data=oriented,
                          sagittal_affine=new_affine)
        else:
            if result['axial_slices']:
                raise ValueError('Select only one axial series.')
            lps = np.diag([-1, -1, 1, 1]) @ new_affine
            zoom = np.linalg.norm(lps[:3, :3], axis=0)
            source_frame_axis = int(np.argmax(abs(source_voxel_transform[:3, 2])))
            provenance['native_frame_axis'] = source_frame_axis
            for index in range(oriented.shape[2]):
                position = (lps @ np.array([0, 0, index, 1]))[:3]
                source_voxel = source_voxel_transform @ np.array([0, 0, index, 1])
                source_frame_index = int(round(float(source_voxel[source_frame_axis])))
                result['axial_slices'].append({
                    'pixel_array': oriented[:, :, index].T,
                    'image_position': position.tolist(),
                    'image_orientation': np.concatenate([lps[:3, 0] / zoom[0], lps[:3, 1] / zoom[1]]).tolist(),
                    'pixel_spacing': (float(zoom[1]), float(zoom[0])),
                    'slice_thickness': float(zoom[2]), 'z_world_mm': float(position[2]),
                    'source_kind': provenance['source_kind'], 'source_volume': str(path),
                    'native_frame_axis': source_frame_axis,
                    'native_frame_index': source_frame_index,
                })
    dicom_sources = [source for source in result['sources'] if source['source_kind'] == 'dicom']
    studies = {source['study_instance_uid'] for source in dicom_sources if source.get('study_instance_uid')}
    if len(studies) > 1:
        raise ValueError('Selected DICOM series do not share the same StudyInstanceUID.')
    if result['sagittal_nifti_path'] and result['axial_slices']:
        frames = {source['frame_of_reference_uid'] for source in dicom_sources if source.get('frame_of_reference_uid')}
        if len(dicom_sources) == 2 and len(frames) == 1 and all(source.get('frame_of_reference_uid') for source in dicom_sources):
            result['registration_status'] = 'shared_dicom_frame_of_reference'
        else:
            result['registration_status'] = 'needs_review'
            warning = 'Sagittal and axial series lack a shared DICOM FrameOfReferenceUID; verify physical alignment visually before accepting measurements.'
            result['geometry_warnings'].append(warning)
            for source in result['sources']:
                source['registration_status'] = 'needs_review'
                source['geometry_warning'] = warning
    return result
