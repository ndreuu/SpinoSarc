"""
LevelMapper: parse TotalSpineSeg's `step1_levels` output and map each
disc level to (a) world coordinates and (b) the nearest axial DICOM slice.

Single responsibility: turn TotalSpineSeg's raw NIfTI label map into a
clean Python dict keyed by anatomical level name, and map each level to
the nearest axial slice index in SpinoSarc's existing slice list.

Does NOT:
- Run TotalSpineSeg (runner.py's job)
- Run MuscleMap inference (analyzer.py's job)
- Touch the GUI

Assumes runner.py has already produced a step1_levels NIfTI in the
TotalSpineSeg output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Constants from TotalSpineSeg's levels_maps.json
# (resources/labels_maps/levels_maps.json)
# These are TotalSpineSeg's compact level codes used in step1_levels NIfTI.
# ---------------------------------------------------------------------------

LEVELS_MAP = {
    1: "C1",
    2: "C1-C2",
    3: "C2-C3",
    4: "C3-C4",
    5: "C4-C5",
    6: "C5-C6",
    7: "C6-C7",
    8: "C7-T1",
    9: "T1-T2",
    10: "T2-T3",
    11: "T3-T4",
    12: "T4-T5",
    13: "T5-T6",
    14: "T6-T7",
    15: "T7-T8",
    16: "T8-T9",
    17: "T9-T10",
    18: "T10-T11",
    19: "T11-T12",
    20: "T12-L1",
    21: "L1-L2",
    22: "L2-L3",
    23: "L3-L4",
    24: "L4-L5",
    25: "L5-S",
}

# SpinoSarc-specific 6 target levels we want to analyze.
# 5 IVDs for stenosis assessment + L3 vertebral body for sarcopenia.
TARGET_IVDS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S"]
TARGET_LEVELS = TARGET_IVDS + ["L3_body"]


class LevelMapper:
    """Parse TotalSpineSeg level output and resolve axial slice indices."""

    LEVELS_MAP = LEVELS_MAP
    TARGET_IVDS = TARGET_IVDS
    TARGET_LEVELS = TARGET_LEVELS

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        totalspineseg_output_dir: str,
        sagittal_nifti_path: str,
    ) -> dict:
        """Read the step1_levels NIfTI, return a level -> info dict.

        Parameters
        ----------
        totalspineseg_output_dir
            Directory produced by `runner.run()`. We expect a
            subdirectory `step1_levels/` containing exactly one NIfTI.
        sagittal_nifti_path
            Path to the original sagittal NIfTI that TotalSpineSeg was run
            on. Used to recover the corresponding levels file name (output
            files in step1_levels carry the same basename).

        Returns
        -------
        dict
            Keys are level names from TARGET_LEVELS that were actually
            found (IVDs not visible in the sagittal volume will be
            absent). Each value:
                {
                    "world_xyz": (x, y, z),       # mm in NIfTI RAS coordinates
                    "world_coordinate_system": "RAS",
                    "voxel_xyz": (i, j, k) | None,  # voxel index in the sagittal NIfTI
                    "type": "IVD" | "VB",
                    "source_label": <int> | "computed",
                }

            L3_body is added only if BOTH L2-L3 and L3-L4 were found
            (arithmetic midpoint).

        Raises
        ------
        FileNotFoundError
            If no step1_levels NIfTI is found.
        ValueError
            If the NIfTI is empty (no non-zero labels at all).
        """
        out = Path(totalspineseg_output_dir)
        levels_dir = out / "step1_levels"
        if not levels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected step1_levels directory under "
                f"{totalspineseg_output_dir}, but it does not exist."
            )

        # TotalSpineSeg names the output after the input basename.
        # E.g. input '2_t2_tse_sag_384_L.nii' -> '2_t2_tse_sag_384_L.nii.gz'.
        sag_path = Path(sagittal_nifti_path)
        sag_base = sag_path.name
        # Strip .nii / .nii.gz
        if sag_base.endswith(".nii.gz"):
            sag_stem = sag_base[: -len(".nii.gz")]
        elif sag_base.endswith(".nii"):
            sag_stem = sag_base[: -len(".nii")]
        else:
            sag_stem = sag_path.stem

        # Try both .nii.gz and .nii
        candidates = [
            levels_dir / f"{sag_stem}.nii.gz",
            levels_dir / f"{sag_stem}.nii",
        ]
        # Fallback: any single nifti in the folder
        levels_file: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                levels_file = c
                break
        if levels_file is None:
            any_nii = list(levels_dir.glob("*.nii*"))
            if len(any_nii) == 1:
                levels_file = any_nii[0]
            else:
                raise FileNotFoundError(
                    f"Could not find levels NIfTI in {levels_dir}. "
                    f"Looked for {[c.name for c in candidates]}, "
                    f"found {[p.name for p in any_nii]}."
                )

        # Load
        img = nib.load(str(levels_file))
        data = np.asarray(img.get_fdata())
        affine = np.asarray(img.affine)

        unique_labels = [int(v) for v in np.unique(data) if int(v) != 0]
        if not unique_labels:
            raise ValueError(
                f"Levels NIfTI {levels_file} contains no non-zero labels."
            )

        # Build a level_name -> info dict for whatever labels are present.
        found: dict = {}
        for lbl in unique_labels:
            name = LEVELS_MAP.get(lbl)
            if name is None:
                # Unknown label code; skip silently rather than crash.
                continue
            mask = (data == lbl)
            if not mask.any():
                continue
            coords = np.argwhere(mask)
            # step1_levels stores single-voxel markers, but be defensive
            # and use the centroid in case of more than one voxel.
            vox = coords.mean(axis=0)  # (i, j, k) in float
            world_hom = affine @ np.array([vox[0], vox[1], vox[2], 1.0])
            world = world_hom[:3]
            found[name] = {
                "world_xyz": (float(world[0]), float(world[1]), float(world[2])),
                "world_coordinate_system": "RAS",
                "voxel_xyz": (float(vox[0]), float(vox[1]), float(vox[2])),
                "type": "IVD",
                "source_label": lbl,
            }

        # Keep only the IVDs we actually care about.
        result: dict = {name: found[name] for name in TARGET_IVDS if name in found}

        # ------- L3 body (arithmetic midpoint of L2-L3 and L3-L4) -------
        if "L2-L3" in result and "L3-L4" in result:
            wx = (result["L2-L3"]["world_xyz"][0] + result["L3-L4"]["world_xyz"][0]) / 2.0
            wy = (result["L2-L3"]["world_xyz"][1] + result["L3-L4"]["world_xyz"][1]) / 2.0
            wz = (result["L2-L3"]["world_xyz"][2] + result["L3-L4"]["world_xyz"][2]) / 2.0
            result["L3_body"] = {
                "world_xyz": (wx, wy, wz),
                "world_coordinate_system": "RAS",
                "voxel_xyz": None,
                "type": "VB",
                "source_label": "computed",
            }

        return result

    # ------------------------------------------------------------------
    # Axial mapping
    # ------------------------------------------------------------------

    def map_to_axial(
        self,
        levels: dict,
        axial_slice_metadata: list,
    ) -> dict:
        """Match RAS level markers to independent native LPS axial planes.

        Coverage needs perpendicular distance inside a physical slice slab
        and the projected point inside the native pixel field of view. The
        explicit world_coordinate_system='LPS' convention is also supported;
        parser output and older parser dictionaries use NIfTI RAS.
        """
        planes = []
        for index, frame in enumerate(axial_slice_metadata):
            try:
                position = np.asarray(frame['image_position'], dtype=float)
                orientation = np.asarray(frame['image_orientation'], dtype=float)
                spacing = np.asarray(frame['pixel_spacing'], dtype=float)
                shape = np.asarray(frame['pixel_array']).shape
                thickness = float(frame['slice_thickness'])
                if (position.shape != (3,) or orientation.shape != (6,) or spacing.shape != (2,)
                        or len(shape) != 2 or min(shape) < 1 or thickness <= 0
                        or not np.isfinite(np.r_[position, orientation, spacing, thickness]).all()
                        or not (spacing > 0).all()):
                    continue
                col_dir, row_dir = orientation[:3], orientation[3:]
                if (not np.allclose([np.linalg.norm(col_dir), np.linalg.norm(row_dir)], 1., atol=1e-3)
                        or abs(np.dot(col_dir, row_dir)) > 1e-3):
                    continue
                col_dir = col_dir / np.linalg.norm(col_dir)
                row_dir = row_dir / np.linalg.norm(row_dir)
                normal = np.cross(col_dir, row_dir)
                planes.append(dict(index=index, position=position, col_dir=col_dir,
                                   row_dir=row_dir, normal=normal, spacing=spacing,
                                   shape=shape, thickness=thickness))
            except (KeyError, ValueError, TypeError):
                continue

        # Derive local spacing only from almost parallel neighbors. Cap it
        # to prevent a large gap between IVD groups from inventing coverage.
        for plane in planes:
            gaps = []
            for other in planes:
                if plane is other or abs(np.dot(plane['normal'], other['normal'])) < np.cos(np.deg2rad(.5)):
                    continue
                gap = abs(float(np.dot(other['position'] - plane['position'], plane['normal'])))
                if gap > 1e-3:
                    gaps.append(gap)
            nearest_gap = min(gaps) if gaps else plane['thickness']
            effective_spacing = max(plane['thickness'], min(nearest_gap, 2 * plane['thickness']))
            plane['tolerance'] = effective_spacing / 2 + .25

        for name, info in levels.items():
            info.update(axial_slice_idx=None, out_of_range=True,
                        axial_mapping_method='native_plane_distance_and_fov')
            for stale_key in ('axial_distance_mm', 'axial_pixel_row_col', 'axial_coverage_tolerance_mm'):
                info.pop(stale_key, None)
            if not planes:
                info['out_of_range_reason'] = 'no valid native axial plane geometry'
                continue
            try:
                point = np.asarray(info['world_xyz'], dtype=float)
                convention = info.get('world_coordinate_system', 'RAS').upper()
                if point.shape != (3,) or not np.isfinite(point).all() or convention not in ('RAS', 'LPS'):
                    raise ValueError('invalid world point')
                if convention == 'RAS':
                    point = point * [-1., -1., 1.]
            except (KeyError, TypeError, ValueError, AttributeError):
                info['out_of_range_reason'] = 'invalid level world coordinates'
                continue
            candidates = []
            for plane in planes:
                offset = point - plane['position']
                distance = abs(float(np.dot(offset, plane['normal'])))
                col = float(np.dot(offset, plane['col_dir']) / plane['spacing'][1])
                row = float(np.dot(offset, plane['row_dir']) / plane['spacing'][0])
                nrow, ncol = plane['shape']
                if -.5 <= row <= nrow - .5 and -.5 <= col <= ncol - .5:
                    candidates.append((distance, plane['index'], row, col, plane['tolerance']))
            if not candidates:
                info['out_of_range_reason'] = 'level projects outside native axial field of view'
                continue
            distance, index, row, col, tolerance = min(candidates, key=lambda entry: (entry[0], entry[1]))
            if distance > tolerance:
                info['out_of_range_reason'] = (
                    f'nearest native plane is {distance:.2f} mm away '
                    f'(coverage tolerance {tolerance:.2f} mm); level is outside slice coverage or in an axial gap'
                )
                continue
            info.update(axial_slice_idx=index, out_of_range=False,
                        axial_distance_mm=distance, axial_pixel_row_col=[row, col],
                        axial_coverage_tolerance_mm=tolerance)
            info.pop('out_of_range_reason', None)
        return levels
