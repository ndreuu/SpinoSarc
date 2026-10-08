"""
MuscleMap inference engine — in-process wrapper.

Setup once, segment many times. Model stays in RAM.
"""
import os
import sys
import logging
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd, EnsureTyped,
    Invertd, AsDiscreted,
)
from monai.networks.nets import UNet
from monai.inferers import SliceInferer
from monai.networks.layers.factories import Norm


def _resolve_musclemap_path():
    """MuscleMap scripts klasörünü bul. PyInstaller bundle veya dev mode.

    Sıra:
    1. SPINOSARC_MUSCLEMAP_PATH env var (manuel override)
    2. PyInstaller bundle: sys._MEIPASS/musclemap_scripts
    3. Dev mode: ~/SpinoSarc/MuscleMap/scripts (Berkay'in makinesi)
    """
    # 1) Env var override
    env_path = os.environ.get('SPINOSARC_MUSCLEMAP_PATH')
    if env_path and Path(env_path).is_dir():
        return Path(env_path)

    # 2) PyInstaller frozen bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_path = Path(sys._MEIPASS) / 'musclemap_scripts'
        if bundle_path.is_dir():
            return bundle_path

    # 3) Dev mode fallback
    dev_path = Path.home() / 'SpinoSarc' / 'MuscleMap' / 'scripts'
    return dev_path


_MM_PATH = _resolve_musclemap_path()
if str(_MM_PATH) not in sys.path:
    sys.path.insert(0, str(_MM_PATH))

from mm_util import (
    get_model_and_config_paths, load_model_config,
    SqueezeTransform, RemapLabels, run_inference,
    ensure_model_downloaded,
)


log = logging.getLogger(__name__)


class MuscleMapEngine:
    """
    MuscleMap'i Python class olarak kullan. Subprocess yok.
    Constructor model + transformlari yukler (~5-10 sn).
    segment() her cagrida 1-2 sn (M4 Pro MPS).
    """

    NORM_MAP = {'instance': Norm.INSTANCE}

    def __init__(self, region: str = 'abdomen',
                 model_version: str = None,
                 use_gpu: bool = True,
                 chunk_size = 'auto',
                 overlap_percent: float = 25.0):
        self.region = region
        self.chunk_size = chunk_size
        model_version = model_version or os.environ.get(
            'SPINOSARC_MUSCLEMAP_MODEL_VERSION', '0.0' if region == 'abdomen' else 'latest'
        )
        self.model_version = str(model_version)

        # 1) Device sec: cuda > mps > cpu
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif use_gpu and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        log.info(f"Device: {self.device}")

        # 2) AMP context (cuda-only)
        if self.device.type == 'cuda':
            self.amp_context = torch.amp.autocast('cuda')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = True
        else:
            self.amp_context = nullcontext()

        # 3) Model + config yolu - cache yoksa Zenodo'dan indir
        try:
            self.model_path, self.config_path = get_model_and_config_paths(
                region, None, self.model_version
            )
        except Exception:
            # Cache yoksa indir
            log.info(f"Downloading model: {region} v{model_version}")
            ensure_model_downloaded(region, self.model_version)
            self.model_path, self.config_path = get_model_and_config_paths(
                region, None, self.model_version
            )

        cfg = load_model_config(self.config_path)
        self.config = cfg
        self.provenance = self._verify_model_provenance()
        log.info(f"Loaded config: {region} v{cfg.get('model', {}).get('version', '?')}")

        # 4) Parametreler config'den
        self.roi_size                = tuple(cfg['parameters']['roi_size'])
        self.sw_batch_size           = cfg['parameters']['spatial_window_batch_size']
        self.pix_dim                 = tuple(cfg['parameters']['pix_dim'])
        spatial_dims                 = cfg['model']['spatial_dims']
        in_channels                  = cfg['model']['in_channels']
        self.out_channels            = cfg['model']['out_channels']
        channels                     = cfg['model']['channels']
        act                          = cfg['model']['act']
        strides                      = cfg['model']['strides']
        num_res_units                = cfg['model']['num_res_units']
        norm_str                     = cfg['model']['norm']
        label_entries                = cfg['labels']
        anatomy_names = {
            'multifidus': 'multifidus', 'erector spinae': 'erector',
            'psoas major': 'psoas', 'quadratus lumborum': 'QL',
        }
        self.muscle_labels = {}
        if region == 'abdomen':
            for entry in label_entries:
                anatomy = anatomy_names.get(entry['anatomy'].lower())
                side = {'right': 'R', 'left': 'L'}.get(entry['side'].lower())
                if anatomy is None or side is None:
                    raise ValueError('Unexpected MuscleMap abdomen anatomy or side.')
                label = int(entry['value'])
                if label in self.muscle_labels or f'{anatomy}_{side}' in self.muscle_labels.values():
                    raise ValueError('Duplicate MuscleMap abdomen label.')
                self.muscle_labels[label] = f'{anatomy}_{side}'
            expected = {f'{name}_{side}' for name in anatomy_names.values() for side in ('R', 'L')}
            if set(self.muscle_labels.values()) != expected or self.out_channels != 9:
                raise ValueError('The selected model does not contain the expected eight abdomen muscles.')
        self.provenance['labels'] = self.muscle_labels

        # Label mapping (wholebody icin gerekli)
        labels = sorted({e['value'] for e in label_entries})
        self.id_map = {0: 0}
        for new_id, orig in enumerate(labels, start=1):
            self.id_map[orig] = new_id
        self.inv_id_map = {new: orig for orig, new in self.id_map.items()}

        norm = self.NORM_MAP[norm_str]

        # 5) Pre-transforms
        self.pre_transforms = Compose([
            LoadImaged(keys=['image'], image_only=False),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=self.pix_dim, mode='bilinear'),
            NormalizeIntensityd(keys=['image'], nonzero=True),
            CropForegroundd(keys=['image'], source_key='image', margin=20),
            SpatialPadd(keys=['image'], spatial_size=(256, 256, 1),
                         method='end', mode='constant'),
            EnsureTyped(keys=['image']),
        ])

        # 6) Post-transforms (CPU'da; MPS uyumsuzluk fix'i icin)
        post_device = torch.device('cpu')
        post_list = [
            # Invertd's device argument moves the result only after inversion.
            # MONAI computes float64 affine grids while inverting Spacingd;
            # move logits first because Metal has no float64 tensor support.
            EnsureTyped(keys=['pred'], device=post_device),
            Invertd(
                keys='pred', transform=self.pre_transforms, orig_keys='image',
                meta_keys='pred_meta_dict', orig_meta_keys='image_meta_dict',
                meta_key_postfix='meta_dict', nearest_interp=False,
                to_tensor=True, device=post_device,
            ),
            AsDiscreted(keys='pred', argmax=True),
            SqueezeTransform(keys=['pred']),
        ]
        if region == 'wholebody':
            post_list.append(RemapLabels(keys=['pred'], id_map=self.inv_id_map))
        self.post_transforms = Compose(post_list)

        # 7) Model yukle
        log.info(f"Loading model weights from {self.model_path}")
        state = torch.load(self.model_path, map_location='cpu', weights_only=True)
        self.model = UNet(
            spatial_dims=spatial_dims, in_channels=in_channels,
            out_channels=self.out_channels, channels=channels,
            act=act, strides=strides, num_res_units=num_res_units, norm=norm,
        )
        self.model.load_state_dict(state)
        del state
        self.model = self.model.to(self.device)
        self.model.eval()

        # 8) Inferer
        overlap = overlap_percent / 100.0
        self.inferer = SliceInferer(
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            spatial_dim=2, mode='gaussian', overlap=overlap,
        )

        log.info(f"MuscleMapEngine ready: region={region}, device={self.device}")

    def _verify_model_provenance(self):
        """Verify a configured local manifest without changing model files."""
        provenance = {
            'repository': 'https://github.com/MuscleMap/MuscleMap',
            'region': self.region, 'model_version': self.model_version,
            'device': str(self.device), 'checksums_verified': False,
            'preprocessing_device': 'cpu', 'postprocessing_device': 'cpu',
        }
        manifest_value = os.environ.get('SPINOSARC_MUSCLEMAP_MANIFEST')
        if not manifest_value:
            return provenance
        manifest_path = Path(manifest_value).resolve()
        manifest = json.loads(manifest_path.read_text())
        if self.region != 'abdomen' or str(manifest.get('version')) != self.model_version:
            raise ValueError('MuscleMap manifest does not match the selected region/version.')
        for actual_path in (self.model_path, self.config_path):
            actual_path = Path(actual_path).resolve()
            entry = next((entry for entry in manifest['files']
                          if (manifest_path.parent / entry['path']).resolve() == actual_path), None)
            if entry is None or not actual_path.is_file():
                raise ValueError('MuscleMap model/config is not present in the configured manifest.')
            digest = hashlib.sha256()
            with actual_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if actual_path.stat().st_size != entry['size'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('MuscleMap model/config checksum mismatch.')
        provenance.update({
            'checksums_verified': True, 'manifest_path': str(manifest_path),
            'record_id': manifest['record_id'], 'source': manifest['source'],
            'license': manifest['license'], 'source_revision': manifest.get('source_revision'),
        })
        return provenance

    def segment(self, image_path: str, output_dir: str = None) -> np.ndarray:
        """
        Segment a NIfTI 2D slice or 3D volume.
        Returns the segmentation array (same shape as input).
        Optionally writes _dseg.nii.gz to output_dir.
        """
        image_path = str(image_path)
        write_disk = output_dir is not None
        if not write_disk:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix='mm_inference_')

        out_path = run_inference(
            image_path,
            output_dir,
            self.pre_transforms,
            self.post_transforms,
            self.amp_context,
            self.chunk_size,
            self.device,
            self.inferer,
            self.model,
            out_channels=self.out_channels,
            target_pixdim=self.pix_dim,
        )

        # Load segmentation back as numpy
        predicted = nib.load(out_path)
        original = nib.load(image_path)
        if predicted.shape != original.shape or not np.allclose(predicted.affine, original.affine, atol=1e-4):
            raise ValueError('MuscleMap prediction is not aligned with the original native NIfTI grid.')
        raw_seg = np.asarray(predicted.dataobj)
        if not np.isfinite(raw_seg).all() or not np.equal(raw_seg, np.round(raw_seg)).all():
            raise ValueError('MuscleMap prediction contains invalid label values.')
        seg = raw_seg.astype(np.int16)
        if self.region == 'abdomen' and not set(np.unique(seg)).issubset({0, *self.muscle_labels}):
            raise ValueError('MuscleMap prediction contains labels absent from its configuration.')

        if not write_disk:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

        return seg
