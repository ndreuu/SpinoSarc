"""
TotalSpineSeg subprocess runner.

Calls an existing TotalSpineSeg executable or a separate conda environment
to avoid dependency conflicts with SpinoSarc's MuscleMap environment.

Single responsibility: launch subprocess, manage errors, report progress.
NIfTI parsing and axial slice mapping happens elsewhere (level_mapper.py).
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
import threading
from pathlib import Path
from typing import Callable, Optional


class TotalSpineSegRunner:
    """Wrap the local provider, with the original conda path as fallback."""

    def __init__(self, conda_env_name: str = "totalspineseg"):
        self.conda_env_name = conda_env_name
        self._conda_exe: Optional[str] = None
        # The local demo reuses the separately installed, verified provider.
        # No conda environment or MPS patch is needed for this path.
        self.local_command = os.environ.get("SPINE_TSS_COMMAND")
        self.local_data_dir = os.environ.get("SPINE_TSS_DATA_DIR")
        self._process = None
        self._cancelled = threading.Event()
        self._manual_cancelled = threading.Event()

    def cancel(self, *, runtime_triggered=False):
        """Cancel only this runner's process group without blocking Qt."""
        if not runtime_triggered:
            self._manual_cancelled.set()
        self._cancelled.set()
        process = self._process
        if process is None:
            return
        try:
            if os.name == 'posix':
                # The leader may have exited while children still hold its
                # stdout/stderr pipes. Its own process group still needs stopping.
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            return
        def force_stop():
            if self._process is process:
                try:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.poll() is None:
                        process.kill()
                except ProcessLookupError:
                    pass
        timer = threading.Timer(3, force_stop)
        timer.daemon = True
        timer.start()

    @staticmethod
    def _read_runtime_marker(output_dir, baseline, status):
        for name in ('mps-runtime.json', 'cpu-demo-runtime.json'):
            path = output_dir / name
            try:
                if path.is_symlink():
                    continue
                stat = path.stat()
                if stat.st_size > 1024 * 1024 or stat.st_mtime_ns == baseline[name]:
                    continue
                payload = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(payload, dict) or payload.get('status') != status:
                    continue
            except (OSError, ValueError, UnicodeError):
                # A writer may briefly leave incomplete JSON; retry next tick.
                continue
            error = payload.get('error')
            if status == 'failed' and (not isinstance(error, str) or not error.strip()):
                error = 'Inference adapter reported a failure without an error message'
            return {'error': error[:4000] if isinstance(error, str) else None, 'marker': str(path)}
        return None

    def _watch_runtime_failure(self, process, output_dir, baseline, stop_event,
                               failure, recovery, output_files, progress_callback):
        """Read only this run's changed runtime markers; never drain its pipes."""
        completed_at = None
        completed_marker = None
        while not stop_event.wait(1):
            if self._process is not process:
                return
            detected = self._read_runtime_marker(output_dir, baseline, 'failed')
            if detected:
                failure.update(detected)
                progress_callback('TotalSpineSeg runtime failed; stopping inference workers...')
                self.cancel(runtime_triggered=True)
                return
            if self._manual_cancelled.is_set():
                return
            completed = self._read_runtime_marker(output_dir, baseline, 'completed')
            try:
                outputs_ready = all(path.is_file() and path.stat().st_size for path in output_files)
            except OSError:
                outputs_ready = False
            if not completed or not outputs_ready:
                completed_at = None
                completed_marker = None
                continue
            if completed_marker != completed['marker']:
                completed_marker = completed['marker']
                completed_at = time.monotonic()
            if time.monotonic() - completed_at >= 2:
                # A completed adapter can hang in Python's multiprocessing
                # shutdown. Preserve its outputs, but close our own workers.
                recovery.update(runtime_metadata_path=completed_marker)
                progress_callback('Inference complete; closing background workers...')
                self.cancel(runtime_triggered=True)
                return

    def _local_prefix(self) -> list[str]:
        """Validate the local provider without importing torch or downloading.

        TotalSpineSeg normally auto-installs missing weights. Check its actual
        default release and both installed datasets before allowing inference.
        The launcher supplies the same manifest prepared by our downloader.
        """
        executable = shutil.which(self.local_command or "")
        if not executable:
            raise ValueError("SPINE_TSS_COMMAND is not an executable file")
        if not self.local_data_dir:
            raise ValueError("SPINE_TSS_DATA_DIR is required for the local provider")
        command = Path(executable).absolute()
        data_dir = Path(self.local_data_dir).resolve()
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("A local TotalSpineSeg weights manifest is required")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("provider") != "totalspineseg" or manifest.get("schema_version") != 1:
            raise ValueError("Invalid local TotalSpineSeg weights manifest")

        interpreter = os.environ.get("SPINE_TSS_PYTHON")
        if not interpreter:
            sibling = command.parent / "python"
            if not sibling.is_file():
                raise ValueError("Set SPINE_TSS_PYTHON to the provider's interpreter")
            interpreter = str(sibling)
        program = (
            "import json; from importlib.metadata import metadata; "
            "m=metadata('totalspineseg'); "
            "print(json.dumps({'version':m['Version'], "
            "'urls':dict(x.split(', ',1) for x in m.get_all('Project-URL',[]) "
            "if x.startswith('Dataset'))}))"
        )
        probe = subprocess.run(
            [interpreter, "-c", program], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=15,
        )
        if probe.returncode or len(probe.stdout) > 65536:
            raise ValueError("Cannot read the local TotalSpineSeg package metadata")
        package = json.loads(probe.stdout)
        if not isinstance(package, dict) or package.get("version") != manifest.get("package_version"):
            raise ValueError("Local TotalSpineSeg package does not match its manifest")
        datasets = ("Dataset101_TotalSpineSeg_step1", "Dataset102_TotalSpineSeg_step2")
        urls = package.get("urls", {})
        release = manifest.get("weights_release")
        if not isinstance(release, str) or not release or Path(release).name != release or not isinstance(urls, dict) or set(urls) != set(datasets):
            raise ValueError("Invalid local TotalSpineSeg release metadata")
        results = data_dir / "nnUNet" / "results"
        if all((results / dataset).is_dir() for dataset in datasets):
            raise ValueError("Unversioned TotalSpineSeg models shadow the verified release")
        recorded = manifest.get("files")
        if not isinstance(recorded, dict):
            raise ValueError("Local weights manifest contains no model files")
        for dataset in datasets:
            expected_url = f"https://github.com/neuropoly/totalspineseg/releases/download/{release}/{dataset}_{release}.zip"
            if urls.get(dataset) != expected_url:
                raise ValueError("Provider would use a different weights release")
            folds = list((results / release / dataset).glob("*/fold_0"))
            if len(folds) != 1:
                raise ValueError(f"Installed local model is missing: {dataset}")
            for model_file in (folds[0] / "checkpoint_best.pth", folds[0].parent / "dataset.json", folds[0].parent / "plans.json"):
                if not model_file.is_file() or not model_file.stat().st_size or model_file.relative_to(data_dir).as_posix() not in recorded:
                    raise ValueError(f"Local model file is missing or unrecorded: {model_file.name}")
        return [str(command)]

    @staticmethod
    def _stop_process_group(process: subprocess.Popen) -> tuple[str, str]:
        """Stop the inference process and its nnU-Net worker processes."""
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            return process.communicate()

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def _find_conda(self) -> Optional[str]:
        """Locate the conda executable.

        Returns absolute path to conda, or None if not found.
        Caches the result.
        """
        if self._conda_exe is not None:
            return self._conda_exe

        # 1) PATH lookup
        conda = shutil.which("conda")
        if conda:
            self._conda_exe = conda
            return conda

        # 2) Common install paths on macOS
        for candidate in (
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/miniconda3/bin/conda"),
        ):
            if os.path.isfile(candidate):
                self._conda_exe = candidate
                return candidate

        return None

    def is_available(self) -> bool:
        """Check the local provider's metadata/models, or the conda executable."""
        if self.local_command:
            try:
                self._local_prefix()
                return True
            except (OSError, ValueError, subprocess.SubprocessError):
                return False

        conda = self._find_conda()
        if conda is None:
            return False

        try:
            r = subprocess.run(
                [conda, "run", "-n", self.conda_env_name,
                 "totalspineseg", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

        return r.returncode == 0

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        sagittal_nifti_path: str,
        output_dir: str,
        device: str = "mps",
        step1_only: bool = True,
        iso: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run TotalSpineSeg on a sagittal NIfTI volume.

        Parameters
        ----------
        sagittal_nifti_path
            Absolute path to a sagittal NIfTI (`.nii` or `.nii.gz`).
            Must be a single file; TotalSpineSeg's input folder will be
            created internally.
        output_dir
            Absolute path where TotalSpineSeg will write its outputs.
            Will be created if missing. Existing contents are NOT erased.
        device
            "mps" (Apple Silicon), "cuda" (NVIDIA), or "cpu".
            Defaults to "mps". The TotalSpineSeg installation must have
            our MPS support patch applied for "mps" to work.
            With SPINE_TSS_COMMAND, SPINE_TSS_DEVICE defaults to "cpu" and
            overrides this argument. MPS requires SPINE_TSS_MPS_WRAPPER;
            the pinned official CLI itself supports only CPU/CUDA.
        step1_only
            If True, pass `--step1` (faster, vertebra-level identification only).
            If False, runs the full two-step pipeline (slower, individual labels).
        iso
            If True, pass `--iso` (1mm isotropic output, easier downstream
            processing). Defaults to True.
        progress_callback
            Optional callable taking a single status string. Called at
            start, intermediate milestones (if available), and end.

        Returns
        -------
        dict
            On success:
                {
                    "success": True,
                    "output_dir": <absolute path>,
                    "duration_sec": <float>,
                    "command": <list of strings>,
                }
            On failure:
                {
                    "success": False,
                    "error": <human-readable string>,
                    "stdout": <last 2000 chars>,
                    "stderr": <last 2000 chars>,
                    "command": <list of strings>,
                    "duration_sec": <float>,
                }
        """
        def _emit(msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    # progress callback errors should not break inference
                    pass

        # ------ validate input ------
        sag = Path(sagittal_nifti_path)
        if not sag.is_file():
            return {
                "success": False,
                "error": f"Sagittal NIfTI not found: {sagittal_nifti_path}",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }
        if sag.suffix not in (".nii",) and "".join(sag.suffixes[-2:]) != ".nii.gz":
            return {
                "success": False,
                "error": f"Input is not a NIfTI file: {sagittal_nifti_path}",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }

        # ------ resolve provider ------
        try:
            if self.local_command:
                prefix = self._local_prefix()
                # MPS uses our adapter around the same upstream inference function.
                device = os.environ.get("SPINE_TSS_DEVICE", "cpu")
                if device not in {"cpu", "cuda", "mps"}:
                    raise ValueError("Unknown inference device")
                interpreter = os.environ.get("SPINE_TSS_PYTHON", str(Path(prefix[0]).parent / "python"))
                if device == "mps":
                    mps_wrapper = os.environ.get("SPINE_TSS_MPS_WRAPPER")
                    if not mps_wrapper or not Path(mps_wrapper).is_file():
                        raise ValueError("MPS requires the local adapter; the official CLI supports cpu/cuda")
                    prefix = [interpreter, str(Path(mps_wrapper).resolve())]
                wrapper = os.environ.get("SPINE_TSS_CPU_DEMO_WRAPPER")
                if wrapper and device == "cpu":
                    if not Path(wrapper).is_file():
                        raise ValueError("CPU demo wrapper is missing")
                    prefix = [interpreter, str(Path(wrapper).resolve())]
            else:
                conda = self._find_conda()
                if conda is None:
                    raise ValueError("conda executable not found")
                prefix = [conda, "run", "-n", self.conda_env_name, "totalspineseg"]
            timeout_seconds = int(os.environ.get("SPINE_TSS_TIMEOUT_SECONDS", "1800" if self.local_command else "600"))
            if not 1 <= timeout_seconds <= 86400:
                raise ValueError("TotalSpineSeg timeout must be between 1 and 86400 seconds")
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            return {
                "success": False,
                "error": f"TotalSpineSeg provider is unavailable: {e}",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }

        # ------ prepare a single-file input folder ------
        # TotalSpineSeg accepts either a single file OR a folder. To keep
        # the output deterministic (named by input filename), we always
        # pass the file directly. But we still need an output folder.
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # ------ build command ------
        cmd = [
            *prefix,
            str(sag.resolve()),
            str(out),
            "--device", device,
        ]
        if step1_only:
            cmd.append("--step1")
        if iso:
            cmd.append("--iso")
        required_outputs = ["step1_levels", "step1_canal"]
        if not step1_only:
            required_outputs.append("step2_output")
        if self.local_command:
            cmd.extend([
                "--data-dir", str(Path(self.local_data_dir).resolve()),
                "--max-workers", "1", "--max-workers-nnunet", "1", "--quiet",
                "--keep-only", *required_outputs,
            ])
        stem = sag.name[:-7] if sag.name.endswith('.nii.gz') else sag.stem
        expected_files = [out / folder / f'{stem}.nii.gz' for folder in required_outputs]

        _emit("Starting TotalSpineSeg...")

        # ------ run ------
        t0 = time.time()
        process = None
        watcher = None
        watcher_stop = threading.Event()
        runtime_failure = {}
        runtime_recovery = {}
        # Capture before Popen: a reused output directory must not let a stale
        # failed marker terminate a new run unless the marker is written again.
        marker_baseline = {}
        for name in ('mps-runtime.json', 'cpu-demo-runtime.json'):
            try:
                marker_baseline[name] = (out / name).stat().st_mtime_ns
            except OSError:
                marker_baseline[name] = None

        def runtime_failure_response(stdout, stderr, duration):
            return {
                'success': False,
                'runtime_failed': True,
                'error': 'TotalSpineSeg runtime failed: ' + runtime_failure['error'],
                'runtime_error': runtime_failure['error'],
                'runtime_marker': runtime_failure['marker'],
                'stdout': (stdout or '')[-2000:],
                'stderr': (stderr or '')[-2000:],
                'command': cmd,
                'duration_sec': duration,
            }

        try:
            if self._cancelled.is_set():
                return {'success': False, 'cancelled': True, 'error': 'Segmentation cancelled', 'duration_sec': 0}
            env = dict(os.environ)
            if self.local_command:
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    env[name] = "1"
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
            self._process = process
            if self._cancelled.is_set():
                self.cancel()
            if self.local_command:
                watcher = threading.Thread(
                    target=self._watch_runtime_failure,
                    args=(process, out, marker_baseline, watcher_stop, runtime_failure,
                          runtime_recovery, expected_files, _emit),
                    name='tss-runtime-watchdog', daemon=True,
                )
                watcher.start()
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._stop_process_group(process)
            duration = time.time() - t0
            if runtime_failure:
                return runtime_failure_response(stdout, stderr, duration)
            if not runtime_recovery:
                _emit("TotalSpineSeg timed out")
                return {
                    "success": False,
                    "error": (
                        f"TotalSpineSeg timed out after {duration:.0f} seconds "
                        f"on {device}; all inference workers were stopped."
                    ),
                    "stdout": (stdout or "")[-2000:],
                    "stderr": (stderr or "")[-2000:],
                    "command": cmd,
                    "duration_sec": duration,
                }
            # Completion was already confirmed before the timeout; continue to
            # failure/manual-cancel checks and exact output validation below.
        except (FileNotFoundError, OSError) as e:
            if process is not None and process.poll() is None:
                self._stop_process_group(process)
            duration = time.time() - t0
            return {
                "success": False,
                "error": f"Could not execute subprocess: {e}",
                "stdout": "",
                "stderr": "",
                "command": cmd,
                "duration_sec": duration,
            }
        except BaseException:
            if process is not None and process.poll() is None:
                self._stop_process_group(process)
            raise
        finally:
            watcher_stop.set()
            if watcher is not None:
                watcher.join(timeout=1)
            self._process = None

        duration = time.time() - t0

        if self.local_command and not runtime_failure:
            # A fast failure can exit before the watchdog's first one-second tick.
            runtime_failure.update(self._read_runtime_marker(out, marker_baseline, 'failed') or {})
        if runtime_failure:
            return runtime_failure_response(stdout, stderr, duration)

        if self._cancelled.is_set() and (not runtime_recovery or self._manual_cancelled.is_set()):
            return {'success': False, 'cancelled': True, 'error': 'Segmentation cancelled',
                    'stdout': stdout[-2000:], 'stderr': stderr[-2000:], 'command': cmd,
                    'duration_sec': duration}

        if process.returncode != 0 and not runtime_recovery:
            _emit("TotalSpineSeg failed")
            return {
                "success": False,
                "error": (
                    f"TotalSpineSeg exited with code {process.returncode}. "
                    "See stdout/stderr below for details."
                ),
                "stdout": stdout[-2000:] if stdout else "",
                "stderr": stderr[-2000:] if stderr else "",
                "command": cmd,
                "duration_sec": duration,
            }

        # ------ verify expected output exists ------
        # Mapper needs compact markers, while CSA needs the separate soft mask.
        # A final anatomical step2 label map cannot replace either output.
        missing = [folder for folder in required_outputs if not (out / folder / f"{stem}.nii.gz").is_file() or not (out / folder / f"{stem}.nii.gz").stat().st_size]
        if missing:
            _emit("TotalSpineSeg finished but expected output is missing")
            return {
                "success": False,
                "error": (
                    "TotalSpineSeg exited successfully but required output "
                    f"files are missing for {stem}: {', '.join(missing)}"
                ),
                "stdout": stdout[-2000:] if stdout else "",
                "stderr": stderr[-2000:] if stderr else "",
                "command": cmd,
                "duration_sec": duration,
            }

        _emit(f"TotalSpineSeg completed in {duration:.1f}s")
        result = {
            "success": True,
            "output_dir": str(out),
            "duration_sec": duration,
            "command": cmd,
            "inference_profile": "mps_adapter_original_plans" if device == "mps" else ("experimental_cpu_tiles" if os.environ.get("SPINE_TSS_CPU_DEMO_WRAPPER") and device == "cpu" else "official_default"),
        }
        if runtime_recovery:
            result.update(teardown_recovered=True,
                          actual_process_returncode=process.returncode,
                          runtime_metadata_path=runtime_recovery['runtime_metadata_path'])
        return result
