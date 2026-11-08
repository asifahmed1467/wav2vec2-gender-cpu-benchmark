"""Gender detection benchmark for wav2vec2 (CPU only): PyTorch, ONNX Runtime from Python, ONNX Runtime from C++; 32 / 16 / 8 bit.

Per variant:
  1. 20 ms sweep: the first 20, 40 ... 2000 ms of every clip are classified (accuracy and confidence at every step).
  2. "stable from" = first audio length from which the accuracy on ALL clips never falls below its value at 2000 ms.
  3. batch time: all clips are cut at that length; WARMUP_RUNS untimed runs, then TIMED_RUNS timed run(s), for batch sizes 1, 2, 4, 8, 16.
     Only the model call is timed (cutting, normalising and decoding are outside the timer).
Results appear live, chunk by chunk, on a web page (full page and a read-only view for a supervisor).
"""
from __future__ import annotations

import contextlib
import copy
import csv
import gc
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np

# --------------------------------------------------------------------------- settings
MODEL_ID = os.environ.get("HB_MODEL", "prithivMLmods/Common-Voice-Gender-Detection")
SAMPLE_RATE = 16_000
SAMPLES_PER_MS = SAMPLE_RATE // 1000  # 16
SWEEP_MS = list(range(20, 2001, 20))  # 100 steps: 20, 40 ... 2000 ms
MIN_SAMPLES = 400  # the wav2vec2 conv stack needs at least 400 samples (25 ms) to give one output frame
BATCH_SIZES = (1, 2, 4, 8, 16)
SWEEP_BATCH = 16  # clips per model call in the sweep (same length, so no padding)
WARMUP_RUNS = 10  # untimed passes over all batches before the timed pass
TIMED_RUNS = 1  # timed passes
MODELS_DIR = Path(os.environ.get("HB_MODELS_DIR", "/models"))  # converted / quantised models are saved here and reused
RESULTS_DIR = Path(os.environ.get("HB_RESULTS_DIR", "/results"))
CLIPS_DIR = Path(os.environ.get("HB_CLIPS_DIR", "/clips"))
WORK_DIR = Path(os.environ.get("HB_WORK_DIR", "/tmp/gb_work"))
CPP_BIN = os.environ.get("HB_AUDIO", "/usr/local/bin/hb_audio")


# --------------------------------------------------------------------------- machine
def cpu_summary() -> dict:
    """CPU name and thread counts (logical, physical, visible to this container)."""
    name = platform.processor()
    phys: set = set()
    try:
        cur: dict = {}
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and name in ("", "x86_64"):
                name = line.split(":", 1)[1].strip()
            if ":" in line:
                k, v = [x.strip() for x in line.split(":", 1)]
                cur[k] = v
            elif cur:
                phys.add((cur.get("physical id"), cur.get("core id")))
                cur = {}
        if cur:
            phys.add((cur.get("physical id"), cur.get("core id")))
    except OSError:
        pass
    logical = os.cpu_count() or 1
    physical = len(phys) if phys and (None, None) not in phys else logical
    return {"name": name, "logical": logical, "physical": physical, "affinity": len(os.sched_getaffinity(0))}


def choose_threads(cpu: dict, setting: str) -> int:
    """HB_THREADS: 'max' = every logical thread, a number, or empty = physical cores. The same count is used by every row."""
    setting = setting.strip().lower()
    if setting == "max":
        return min(cpu["logical"], cpu["affinity"])
    if setting.isdigit() and int(setting) >= 1:
        return min(int(setting), cpu["logical"], cpu["affinity"])
    return min(cpu["physical"], cpu["affinity"])


# --------------------------------------------------------------------------- audio
@dataclass(frozen=True)
class Clip:
    name: str
    label: int  # index into the model's labels
    samples: np.ndarray  # float32, exactly SWEEP_MS[-1] ms (the first 2 s of the clip)


LABEL_ALIASES = {"f": "female", "female": "female", "woman": "female", "w": "female", "m": "male", "male": "male", "man": "male"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}
HEADER_CELLS = {"filename", "file", "name", "path", "gender", "label", "sex", "class"}


class ClipLoader:
    """Reads labelled clips from a folder: labels.csv (filename,gender) or sub-folders named after the labels. Clips shorter than 2 s are skipped."""

    NEEDED = SWEEP_MS[-1] * SAMPLES_PER_MS

    def __init__(self, folder: Path, labels: Sequence[str]) -> None:
        self.folder, self.labels = Path(folder), [str(x).lower() for x in labels]

    def load(self) -> tuple[list[Clip], list[str]]:
        """-> (clips, messages about skipped files)."""
        clips: list[Clip] = []
        skipped: list[str] = []
        for path, label_text in self._entries():
            idx = self._index(label_text, path)
            try:
                x = self._read(path)
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
                skipped.append(f"{path.name}: cannot be read ({type(e).__name__}: {str(e)[:80]})")
                continue
            if len(x) < self.NEEDED:
                skipped.append(f"{path.name}: {len(x) / SAMPLE_RATE:.2f} s is shorter than 2 s")
                continue
            clips.append(Clip(path.name, idx, np.ascontiguousarray(x[: self.NEEDED], dtype=np.float32)))
        return clips, skipped

    def _index(self, text: str, path: Path) -> int:
        name = LABEL_ALIASES.get(text.strip().lower(), text.strip().lower())
        if name not in self.labels:
            raise ValueError(f"{path.name}: unknown gender '{text}' (the model knows {self.labels})")
        return self.labels.index(name)

    def _entries(self) -> list[tuple[Path, str]]:
        csv_path = self.folder / "labels.csv"
        if csv_path.exists():
            out = []
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if len(row) < 2 or not row[0].strip():
                        continue
                    if not out and (row[0].strip().lower() in HEADER_CELLS or row[1].strip().lower() in HEADER_CELLS):
                        continue  # header line
                    p = self.folder / row[0].strip()
                    if not p.exists():
                        raise FileNotFoundError(f"labels.csv lists {row[0].strip()} but the file is not in {self.folder}")
                    out.append((p, row[1]))
            if not out:
                raise ValueError(f"{csv_path} has no usable rows (expected lines like: clip1.wav,female)")
            return out
        out = [(p, label) for label in self.labels for p in sorted((self.folder / label).glob("*")) if p.suffix.lower() in AUDIO_EXT]
        if not out:
            raise FileNotFoundError(f"no clips found: put audio files + labels.csv (filename,gender) in {self.folder}, or use sub-folders {self.labels}")
        return out

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        """Mono, 16 kHz, float32; only the first ~2.1 s are kept."""
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except ImportError:  # plain PCM wav only
            import wave

            with wave.open(str(path), "rb") as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                if width != 2:
                    raise ValueError("only 16-bit wav without soundfile")
                data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32).reshape(-1, ch) / 32768.0
        x = data.mean(axis=1)[: int(sr * 2.1)]
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            g = math.gcd(SAMPLE_RATE, int(sr))
            x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
        return x


class Preprocessor:
    """Same as the model's Wav2Vec2FeatureExtractor with do_normalize: every row is made zero-mean, unit-variance on its own."""

    def __init__(self, do_normalize: bool = True) -> None:
        self.do_normalize = do_normalize

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: [B, T] float32 -> [B, T] float32."""
        x = np.asarray(x, dtype=np.float32)
        if not self.do_normalize:
            return x
        mean = x.mean(axis=1, keepdims=True, dtype=np.float64)
        var = x.var(axis=1, keepdims=True, dtype=np.float64)
        return ((x - mean) / np.sqrt(var + 1e-7)).astype(np.float32)


# --------------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    name: str  # row name, e.g. python_onnx_int8_pc
    runtime: str  # "torch" | "ort_py" | "ort_cpp"
    precision: str  # "fp32" | "fp16" | "int8" | "int8_pc" | "int8_ffn_pc"
    info: str  # data type text (shown when you hover over the model name)


ONNX_PRECISIONS = [
    ("fp32", "32-bit float"),
    ("int8", "8-bit int, MatMul, per-tensor weights (convolutions stay float)"),
    ("int8_pc", "8-bit int, MatMul, per-channel weights, 7-bit weight range (convolutions stay float)"),
    ("int8_ffn_pc", "8-bit int, feed-forward MatMul only, per-channel weights, 7-bit weight range"),
    ("fp16", "16-bit float weights and ops"),
]


def build_variants() -> list[Variant]:
    """The 14 rows: 4 PyTorch, then each ONNX file through Python and through C++."""
    out = [Variant("python_fp32", "torch", "fp32", "32-bit float (PyTorch)"),
           Variant("python_fp16", "torch", "fp16", "16-bit float (PyTorch)"),
           Variant("python_int8", "torch", "int8", "8-bit int, per-tensor weights, Linear layers (PyTorch)"),
           Variant("python_int8_pc", "torch", "int8_pc", "8-bit int, per-channel weights, Linear layers (PyTorch)")]
    for prec, text in ONNX_PRECISIONS:
        out.append(Variant(f"python_onnx_{prec}", "ort_py", prec, f"{text} (ONNX Runtime, Python)"))
        out.append(Variant(f"cpp_onnx_{prec}", "ort_cpp", prec, f"{text} (ONNX Runtime, C++)"))
    return out


# --------------------------------------------------------------------------- engines
class Engine(ABC):
    """One model in one runtime. prepare() and to_logits() are outside the timer; only run() is timed."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def prepare(self, batch: np.ndarray) -> Any:
        """[B, T] float32 array -> whatever run() needs."""

    @abstractmethod
    def run(self, prepared: Any) -> Any:
        """ONE model call on ONE prepared batch (this is what is timed)."""

    @abstractmethod
    def to_logits(self, raw: Any) -> np.ndarray:
        """Output of run() -> float32 array [B, classes]."""

    @abstractmethod
    def size_mb(self) -> float:
        """Size of the stored model in MB."""

    def context(self) -> Any:
        return contextlib.nullcontext()

    def close(self) -> None:
        """Release resources."""

    def logits(self, batches: Sequence[np.ndarray]) -> np.ndarray:
        """Logits of all batches, in order (used by the sweep; not timed)."""
        prepared = [self.prepare(b) for b in batches]
        with self.context():
            return np.concatenate([self.to_logits(self.run(p)) for p in prepared])

    def timed(self, batches: Sequence[np.ndarray], warmup: int, repeat: int) -> tuple[np.ndarray, list[list[float]]]:
        """`warmup` untimed passes over all batches (the first one also returns the logits), then `repeat` timed passes.
        -> (logits of the first warm-up pass, [[ms of every batch call] for every timed pass])."""
        if warmup < 1:
            raise ValueError("at least one warm-up pass is needed")
        prepared = [self.prepare(b) for b in batches]
        first = None
        times: list[list[float]] = []
        gc.collect()
        gc.disable()  # no garbage-collection pause inside a timed call
        try:
            with self.context():
                for w in range(warmup):
                    outs = [self.run(p) for p in prepared]
                    if w == 0:
                        first = np.concatenate([self.to_logits(o) for o in outs])
                for _ in range(repeat):
                    one = []
                    for p in prepared:
                        t0 = time.perf_counter()
                        self.run(p)
                        one.append((time.perf_counter() - t0) * 1000.0)
                    times.append(one)
        finally:
            gc.enable()
        return first, times


class TorchEngine(Engine):
    """PyTorch model (FP32, FP16 or dynamically quantised INT8)."""

    def __init__(self, name: str, model: Any, dtype: Any, size_mb: float) -> None:
        super().__init__(name)
        self.model, self.dtype, self._size = model, dtype, size_mb

    def prepare(self, batch: np.ndarray) -> Any:
        import torch

        return torch.from_numpy(np.ascontiguousarray(batch)).to(self.dtype)

    def run(self, prepared: Any) -> Any:
        return self.model(input_values=prepared).logits

    def to_logits(self, raw: Any) -> np.ndarray:
        return raw.float().numpy()

    def context(self) -> Any:
        import torch

        return torch.inference_mode()

    def size_mb(self) -> float:
        return self._size


class OrtEngine(Engine):
    """ONNX Runtime from Python."""

    def __init__(self, name: str, path: str, threads: int) -> None:
        super().__init__(name)
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self._size = os.path.getsize(path) / 1e6

    def prepare(self, batch: np.ndarray) -> Any:
        return {self.input_name: np.ascontiguousarray(batch, dtype=np.float32)}

    def run(self, prepared: Any) -> Any:
        return self.sess.run(None, prepared)[0]

    def to_logits(self, raw: Any) -> np.ndarray:
        return np.asarray(raw, dtype=np.float32)

    def size_mb(self) -> float:
        return self._size

    def close(self) -> None:
        self.sess = None


class CppEngine(Engine):
    """ONNX Runtime from C++ (hb_audio worker: loads the model once, then runs jobs sent over stdin)."""

    def __init__(self, name: str, path: str, threads: int, binary: str = CPP_BIN) -> None:
        super().__init__(name)
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self._job_path = WORK_DIR / f"job_{name}.bin"
        self._err = open(WORK_DIR / f"{name}.err", "w+")
        self._size = os.path.getsize(path) / 1e6
        self.proc = subprocess.Popen([binary, str(path), str(threads)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, text=True, bufsize=1)
        first = self.proc.stdout.readline().strip()
        if first != "READY":
            raise RuntimeError(f"C++ worker did not start: {self._tail(first)}")

    def _tail(self, extra: str = "") -> str:
        self._err.flush()
        self._err.seek(0)
        return (extra + " " + self._err.read()[-300:]).strip()

    def _job(self, batches: Sequence[np.ndarray], warmup: int, repeat: int, print_logits: bool) -> tuple[Optional[np.ndarray], list[list[float]]]:
        with open(self._job_path, "wb") as f:
            np.array([len(batches)], dtype="<i4").tofile(f)
            for b in batches:
                b = np.ascontiguousarray(b, dtype="<f4")
                np.array(b.shape, dtype="<i4").tofile(f)
                b.tofile(f)
        self.proc.stdin.write(f"JOB {self._job_path} {warmup} {repeat} {int(print_logits)}\n")
        self.proc.stdin.flush()
        logits: list[list[float]] = []
        times: list[list[float]] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("C++ worker stopped: " + self._tail())
            line = line.strip()
            if line == "END":
                break
            if line.startswith("ERROR"):
                err = line
            elif line.startswith("LOGITS"):
                logits.append([float(x) for x in line.split()[1:]])
            elif line.startswith("TIME"):
                times.append([float(x) for x in line.split()[1:]])
        if "err" in locals():
            raise RuntimeError("C++ worker: " + err)
        return (np.array(logits, dtype=np.float32) if print_logits else None), times

    # the base-class hooks are not used: the whole job runs inside the C++ process
    def prepare(self, batch: np.ndarray) -> Any:
        raise NotImplementedError

    def run(self, prepared: Any) -> Any:
        raise NotImplementedError

    def to_logits(self, raw: Any) -> np.ndarray:
        raise NotImplementedError

    def logits(self, batches: Sequence[np.ndarray]) -> np.ndarray:
        out, _ = self._job(batches, 0, 0, True)
        return out

    def timed(self, batches: Sequence[np.ndarray], warmup: int, repeat: int) -> tuple[np.ndarray, list[list[float]]]:
        if warmup < 1:
            raise ValueError("at least one warm-up pass is needed")
        return self._job(batches, warmup, repeat, True)

    def size_mb(self) -> float:
        return self._size

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self.proc.kill()
        with contextlib.suppress(Exception):
            self._err.close()
        with contextlib.suppress(OSError):
            self._job_path.unlink()


# --------------------------------------------------------------------------- model files (conversion happens once and is saved)
class EngineFactory(ABC):
    @abstractmethod
    def labels(self) -> list[str]:
        """Class names in model order, e.g. ['female', 'male']."""

    @abstractmethod
    def preprocessor(self) -> Preprocessor:
        """Input normalisation of the model."""

    @abstractmethod
    def engine(self, variant: Variant) -> Engine:
        """Build (or load from the saved files) the engine of one variant."""


def serialized_size_mb(model: Any) -> float:
    """Size of the stored weights (works for quantised models too, whose weights are packed)."""
    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def make_torch_int8(model: Any, per_channel: bool) -> Any:
    """Dynamic INT8 for every nn.Linear (attention, feed-forward, projector, classifier): per-channel or per-tensor weights,
    activations quantised on the fly. The convolutional feature extractor stays float (PyTorch has no dynamic Conv quantisation)."""
    import torch
    from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig, quantize_dynamic

    qconfig = per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig
    q = quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear: qconfig})
    if not any("quantized" in type(m).__module__ for m in q.modules()):
        raise RuntimeError("no layer was quantised")
    return q


def bake_weight_norm(model: Any) -> None:
    """Replace weight-norm parametrisations by plain weights (same maths) so the ONNX export is clean. Works in place: use on a copy."""
    import torch
    import torch.nn.utils.parametrize as P

    for m in list(model.modules()):
        if hasattr(m, "parametrizations") and "weight" in getattr(m, "parametrizations", {}):
            P.remove_parametrizations(m, "weight", leave_parametrized=True)
        elif hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)
    # weight_norm also leaves a local function in the load-state-dict hooks; it cannot be pickled, so torch.save of
    # the whole model fails. The hook only matters when loading an old checkpoint, which this program never does.
    for m in model.modules():
        hooks = getattr(m, "_load_state_dict_pre_hooks", None)
        if hooks:
            hooks.clear()


class HfEngineFactory(EngineFactory):
    """The real model from Hugging Face. ONNX files and the quantised PyTorch models are written to MODELS_DIR once and reused."""

    def __init__(self, model_id: str, threads: int, models_dir: Path) -> None:
        self.model_id, self.threads, self.models_dir = model_id, threads, Path(models_dir)
        self._model: Any = None
        self._sizes: dict[str, float] = {}

    # ----- base model
    def _base(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModelForAudioClassification

            torch.set_num_threads(self.threads)
            model = AutoModelForAudioClassification.from_pretrained(self.model_id).eval()
            # wav2vec2 stores its positional conv weight as a weight-norm parametrisation. Such a module cannot be
            # deep-copied or saved whole, so it is turned into a plain weight once, here (same numbers, same maths).
            bake_weight_norm(model)
            self._model = model
            self._drop_stale_files()
        return self._model

    def labels(self) -> list[str]:
        cfg = self._base().config
        return [str(cfg.id2label[i]).lower() for i in range(cfg.num_labels)]

    def preprocessor(self) -> Preprocessor:
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(self.model_id)
        if int(fe.sampling_rate) != SAMPLE_RATE:
            raise ValueError(f"the model wants {fe.sampling_rate} Hz audio; this benchmark uses {SAMPLE_RATE} Hz")
        if getattr(fe, "return_attention_mask", False):
            raise ValueError("the model uses an attention mask; same-length batches without a mask are not valid for it")
        return Preprocessor(bool(getattr(fe, "do_normalize", True)))

    # ----- saved files
    def _stamp(self) -> dict:
        from importlib import metadata

        def ver(pkg: str) -> str:
            try:
                return metadata.version(pkg)
            except metadata.PackageNotFoundError:
                return "none"

        return {"model": self.model_id, "revision": str(getattr(self._model.config, "_commit_hash", None)),
                **{p: ver(p) for p in ("torch", "transformers", "onnxruntime", "onnx")}}

    def _drop_stale_files(self) -> None:
        """If the model or a library version changed, the saved files are deleted so a stale file is never used."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        stamp_file, now = self.models_dir / "stamp.json", self._stamp()
        try:
            same = json.loads(stamp_file.read_text()) == now
        except (OSError, ValueError):
            same = False
        if not same:
            for p in list(self.models_dir.glob("model_*.onnx")) + list(self.models_dir.glob("torch_*.pt")):
                p.unlink()
            stamp_file.write_text(json.dumps(now, indent=1))

    # ----- engines
    def engine(self, variant: Variant) -> Engine:
        if variant.runtime == "torch":
            return self._torch_engine(variant)
        path = self._onnx_file(variant.precision)
        return OrtEngine(variant.name, path, self.threads) if variant.runtime == "ort_py" else CppEngine(variant.name, path, self.threads)

    def _torch_engine(self, v: Variant) -> Engine:
        import torch

        base = self._base()
        if v.precision == "fp32":
            self._sizes.setdefault("fp32", serialized_size_mb(base))
            return TorchEngine(v.name, base, torch.float32, self._sizes["fp32"])
        file = self.models_dir / f"torch_{v.precision}.pt"
        model = None
        if file.exists():
            try:
                model = torch.load(file, weights_only=False)  # our own file, written by this program
            except Exception:  # noqa: BLE001 - unreadable file: convert again
                model = None
        if model is None:
            model = copy.deepcopy(base).to(torch.float16) if v.precision == "fp16" else make_torch_int8(base, per_channel=v.precision == "int8_pc")
            tmp = file.with_suffix(".tmp")
            torch.save(model, tmp)
            os.replace(tmp, file)
        self._sizes.setdefault(v.precision, serialized_size_mb(model))
        return TorchEngine(v.name, model.eval(), torch.float16 if v.precision == "fp16" else torch.float32, self._sizes[v.precision])

    def _onnx_file(self, precision: str) -> str:
        self._base()
        path = self.models_dir / f"model_{precision}.onnx"
        if path.exists():
            return str(path)
        fp32 = self.models_dir / "model_fp32.onnx"
        if not fp32.exists():
            self._export_fp32(fp32)
        if precision != "fp32":
            self._derive(precision, fp32, path)
        return str(path)

    def _export_fp32(self, path: Path) -> None:
        import torch
        import onnxruntime as ort

        base = self._base()
        model = copy.deepcopy(base).eval()
        bake_weight_norm(model)

        class Wrap(torch.nn.Module):
            def __init__(self, m: Any) -> None:
                super().__init__()
                self.m = m

            def forward(self, input_values: Any) -> Any:
                return self.m(input_values=input_values).logits

        rng = np.random.default_rng(0)
        pre = self.preprocessor()
        dummy = torch.from_numpy(pre(rng.standard_normal((2, 32000)).astype(np.float32)))
        tmp = path.with_suffix(".tmp")
        with torch.inference_mode():
            torch.onnx.export(Wrap(model).eval(), (dummy,), str(tmp), input_names=["input_values"], output_names=["logits"],
                              dynamic_axes={"input_values": {0: "batch", 1: "samples"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        for shape in ((3, 8000), (2, 32000), (1, 640)):  # different batch sizes and lengths: the dynamic axes must work
            x = pre(rng.standard_normal(shape).astype(np.float32))
            with torch.inference_mode():
                ref = base(input_values=torch.from_numpy(x)).logits.float().numpy()
            diff = float(np.abs(sess.run(None, {"input_values": x})[0] - ref).max())
            if diff > 1e-3:
                raise RuntimeError(f"the exported ONNX model differs from PyTorch (largest logit difference {diff:.2e} for input {shape})")
        os.replace(tmp, path)

    @staticmethod
    def _derive(precision: str, fp32: Path, out: Path) -> None:
        import onnx

        tmp = out.with_suffix(".tmp")
        if precision == "fp16":
            from onnxruntime.transformers.float16 import convert_float_to_float16

            onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(tmp))
        else:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            kw: dict = {"op_types_to_quantize": ["MatMul"]}  # Conv is not quantised: ConvInteger has no int8-weight CPU kernel
            if precision in ("int8_pc", "int8_ffn_pc"):
                kw.update(per_channel=True, reduce_range=True)
            if precision == "int8_ffn_pc":
                nodes = [n.name for n in onnx.load(str(fp32)).graph.node if n.op_type == "MatMul"]
                ffn = [n for n in nodes if "/feed_forward/" in n]
                if not ffn:
                    raise RuntimeError(f"no feed-forward MatMul found; MatMul names look like {nodes[:3]}")
                kw["nodes_to_quantize"] = ffn
            quantize_dynamic(str(fp32), str(tmp), weight_type=QuantType.QInt8, **kw)
        os.replace(tmp, out)


# --------------------------------------------------------------------------- analysis
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def stable_from(acc: dict[int, float]) -> Optional[int]:
    """First audio length (ms) from which the accuracy never falls below its value at the longest length. None if there are no scored steps."""
    steps = sorted(acc)
    if not steps:
        return None
    final, stable = acc[steps[-1]], None
    for ms in reversed(steps):
        if acc[ms] < final - 1e-9:
            break
        stable = ms
    return stable


class EventLog:
    """Thread-safe list of events; the web page asks for everything after the last event it has seen."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[dict] = []
        self.epoch = 0

    def reset(self) -> None:
        with self._lock:
            self._items = []
            self.epoch += 1

    def emit(self, event: dict) -> None:
        with self._lock:
            self._items.append(event)

    def since(self, offset: int) -> tuple[list[dict], int, int]:
        with self._lock:
            return self._items[offset:], len(self._items), self.epoch


class ResultsWriter:
    """Writes the master tables (Markdown) and the 20 ms sweep (CSV) into RESULTS_DIR while the run goes on. Never raises."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.ok = self.folder.is_dir()

    def start_csv(self) -> None:
        if self.ok:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name in ("gender_results.md", "gender_sweep.csv"):  # keep the previous run: a new run never overwrites it
                old = self.folder / name
                if old.exists():
                    with contextlib.suppress(OSError):
                        os.replace(old, self.folder / f"{old.stem}_{stamp}{old.suffix}")
            with contextlib.suppress(OSError), open(self.folder / "gender_sweep.csv", "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["model", "audio_ms", "clip", "true", "predicted", "confidence", "correct"])

    def append_sweep(self, rows: list[list]) -> None:
        if self.ok:
            with contextlib.suppress(OSError), open(self.folder / "gender_sweep.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

    def write_master(self, master: dict[int, dict[str, dict]], header: str, status: str) -> None:
        if not self.ok:
            return
        lines = ["# Gender detection benchmark (wav2vec2, CPU)", "", f"- Status: **{status}**", f"- Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"- {header}", ""]
        for bs in BATCH_SIZES:
            lines += [f"## Batch size {bs}", "", f"| model | size MB | original | predicted | stable from (ms) | avg batch time (ms per batch of {bs}) |", "|---|---|---|---|---|---|"]
            for v, r in master.get(bs, {}).items():
                if "error" in r:
                    lines.append(f"| {v} | | | | | {r['error']} |")
                else:
                    avg = "-" if r.get("avg") is None else f"{r['avg']:.1f}"
                    lines.append(f"| {v} | {r['size']:.1f} | {r['orig']} | {r['pred']} | {r['stable']} | {avg}{'' if r.get('avg') is not None or not r.get('note') else ' ' + r['note']} |")
            lines.append("")
        with contextlib.suppress(OSError):
            tmp = self.folder / "gender_results.md.tmp"
            tmp.write_text("\n".join(lines), encoding="utf-8")
            os.replace(tmp, self.folder / "gender_results.md")


# --------------------------------------------------------------------------- the benchmark
class Benchmark:
    """Runs every variant: sweep, stable-from, then the batch-time measurement for every batch size. Emits events as soon as anything is known."""

    def __init__(self, clips: list[Clip], pre: Preprocessor, factory: EngineFactory, variants: list[Variant], labels: list[str],
                 events: EventLog, writer: ResultsWriter, header: str) -> None:
        self.clips, self.pre, self.factory, self.variants, self.labels = clips, pre, factory, variants, labels
        self.events, self.writer, self.header = events, writer, header
        self.samples = np.stack([c.samples for c in clips])  # [N, 32000]
        self.truth = np.array([c.label for c in clips])
        self.master: dict[int, dict[str, dict]] = {bs: {} for bs in BATCH_SIZES}
        self._ref: Optional[np.ndarray] = None  # python_fp32 probabilities at the longest length (parity check of the other fp32 rows)

    def run(self) -> None:
        for v in self.variants:
            try:
                self._run_variant(v)
            except Exception as e:  # noqa: BLE001 - show the failure in the tables, keep going with the next variant
                text = f"{type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
                for bs in BATCH_SIZES:
                    self._row(bs, v.name, {"error": text})
        self.writer.write_master(self.master, self.header, "FINISHED")
        self.events.emit({"t": "done"})

    # ----- one variant
    def _run_variant(self, v: Variant) -> None:
        self.events.emit({"t": "msg", "text": f"{v.name}: preparing the model (converted models are saved and reused)"})
        engine = self.factory.engine(v)
        try:
            self.events.emit({"t": "msg", "text": f"{v.name}: 20 ms sweep (accuracy and confidence at 20, 40 ... 2000 ms)"})
            probs_by_ms, acc = self._sweep(v, engine)
            stable = stable_from(acc)
            self.events.emit({"t": "stable", "v": v.name, "ms": stable})
            if stable is None:
                raise RuntimeError("no scored step in the sweep")
            self._parity_check(v, probs_by_ms)
            pred_at = probs_by_ms[stable].argmax(axis=1)
            orig_text = " / ".join(f"{int((self.truth == i).sum())} {name[:1].upper()}" for i, name in enumerate(self.labels))
            pred_text = " / ".join(f"{int((pred_at == i).sum())} {name[:1].upper()}" for i, name in enumerate(self.labels)) \
                + f" ({int((pred_at == self.truth).sum())} of {len(self.clips)} correct)"
            base = {"v": v.name, "size": engine.size_mb(), "orig": orig_text, "pred": pred_text, "stable": stable, "avg": None}
            for bs in BATCH_SIZES:  # the rows appear now, the batch time is filled in as each measurement ends
                self._row(bs, v.name, {**base, "note": "" if len(self.clips) >= bs else f"needs at least {bs} clips"})
            for bs in BATCH_SIZES:
                self._time(v, engine, probs_by_ms[stable], stable, bs, base)
        finally:
            engine.close()

    def _row(self, bs: int, name: str, data: dict) -> None:
        self.master[bs][name] = data
        self.events.emit({"t": "row", "bs": bs, "v": name, **data})
        self.writer.write_master(self.master, self.header, "RUNNING")

    # ----- 20 ms sweep
    def _sweep(self, v: Variant, engine: Engine) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        n_clips = len(self.clips)
        probs_by_ms: dict[int, np.ndarray] = {}
        acc: dict[int, float] = {}
        csv_rows: list[list] = []
        for ms in SWEEP_MS:
            n = ms * SAMPLES_PER_MS
            if n < MIN_SAMPLES:
                self.events.emit({"t": "step", "v": v.name, "ms": ms, "na": True})
                continue
            x = self.pre(self.samples[:, :n])
            batches = [x[i:i + SWEEP_BATCH] for i in range(0, n_clips, SWEEP_BATCH)]
            p = softmax(engine.logits(batches))
            pred, conf = p.argmax(axis=1), p.max(axis=1)
            probs_by_ms[ms] = p
            acc[ms] = float((pred == self.truth).mean() * 100.0)
            self.events.emit({"t": "step", "v": v.name, "ms": ms, "pred": pred.tolist(), "conf": np.round(conf, 4).tolist(), "p": np.round(p, 4).tolist()})
            csv_rows += [[v.name, ms, c.name, self.labels[c.label], self.labels[int(pred[i])], f"{conf[i]:.4f}", int(pred[i] == c.label)] for i, c in enumerate(self.clips)]
        self.writer.append_sweep(csv_rows)
        return probs_by_ms, acc

    def _parity_check(self, v: Variant, probs_by_ms: dict[int, np.ndarray]) -> None:
        """Normally silent: the three FP32 rows must give (almost) the same probabilities."""
        if v.precision != "fp32":
            return
        last = probs_by_ms[max(probs_by_ms)]
        if v.name == "python_fp32":
            self._ref = last
        elif self._ref is not None and self._ref.shape == last.shape:
            diff = float(np.abs(last - self._ref).max())
            if diff > 1e-3:
                self.events.emit({"t": "warn", "text": f"WARNING: {v.name} differs from python_fp32 (largest probability difference {diff:.1e} at 2000 ms)."})

    # ----- batch time
    def _time(self, v: Variant, engine: Engine, probs_stable: np.ndarray, stable: int, bs: int, base: dict) -> None:
        n_full = len(self.clips) // bs
        if n_full == 0:
            return  # the row already says how many clips are needed
        used = n_full * bs
        self.events.emit({"t": "msg", "text": f"{v.name}: batch size {bs}, audio cut at {stable} ms ({WARMUP_RUNS} warm-up runs, then {TIMED_RUNS} timed)"})
        x = self.pre(self.samples[:used, : stable * SAMPLES_PER_MS])
        batches = [x[i * bs:(i + 1) * bs] for i in range(n_full)]
        first, times = engine.timed(batches, WARMUP_RUNS, TIMED_RUNS)
        values = [t for one in times for t in one]
        avg = statistics.fmean(values)
        differ = int((softmax(first).argmax(axis=1) != probs_stable[:used].argmax(axis=1)).sum())
        if differ:
            self.events.emit({"t": "warn", "text": f"{v.name}, batch size {bs}: {differ} of {used} predictions at {stable} ms differ from the sweep (rounding differences between batch sizes)."})
        self._row(bs, v.name, {**base, "avg": avg, "calls": len(values), "note": ""})


# --------------------------------------------------------------------------- web page
PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Gender detection benchmark - CPU</title>
<style>
body{font:14px system-ui;margin:20px auto;max-width:1250px;padding:0 14px}h2,h3{margin:14px 0 6px}
button{padding:6px 12px;margin:2px 4px 2px 0;cursor:pointer}button.on{background:#2563eb;color:#fff;border:1px solid #2563eb;border-radius:4px}
table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:4px 8px;text-align:left;white-space:nowrap}th{background:#f1f1f1}
#grid td{min-width:112px;font-size:12px;cursor:pointer}#grid td.st{outline:3px solid #111;outline-offset:-3px;font-weight:700}
#grid th:first-child,#grid td:first-child{position:sticky;left:0;background:#f7f7f7;z-index:1;cursor:default}
#warn{color:#b91c1c;margin:6px 0}pre{background:#f4f4f4;padding:8px;min-height:2.5em;white-space:pre-wrap}
</style>
<h2>wav2vec2 gender detection - CPU benchmark</h2>
<div><button id=run onclick=run()>Run benchmark</button> <span id=status>waiting</span></div>
<p id=info></p><div id=warn></div>
<h3>Master table</h3><div id=tabs></div><div id=master></div>
<h3>Every 20 ms (audio from the start of the clip; bold outline = stable from)</h3>
<label>Clip <select id=clip onchange="drawGrid()"></select></label>
<div style="overflow:auto;max-height:68vh;margin-top:6px"><table id=grid></table></div>
<pre id=detail>Click a cell to see its details.</pre>
<script>
const RO=__READONLY__;
if(RO)document.getElementById('run').style.display='none';
function fresh(){return {clips:[],variants:[],info:{},labels:[],ms:[],bs:[],sweep:{},stable:{},rows:{},warns:[],msg:'',running:false,error:'',done:false,header:''}}
let S=fresh(),off=0,epoch=null,tab=1,sel='all';
const $=id=>document.getElementById(id);
const esc=x=>String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function apply(e){
 if(e.t==='init'){S=Object.assign(fresh(),e);tab=S.bs[0]||1;return}
 if(e.t==='msg')S.msg=e.text;
 else if(e.t==='step'){(S.sweep[e.v]=S.sweep[e.v]||{})[e.ms]=e.na?null:{pred:e.pred,conf:e.conf,p:e.p}}
 else if(e.t==='stable')S.stable[e.v]=e.ms;
 else if(e.t==='row'){(S.rows[e.bs]=S.rows[e.bs]||{})[e.v]=e}
 else if(e.t==='warn')S.warns.push(e.text);
 else if(e.t==='done')S.done=true;
 else if(e.t==='error')S.error=e.text;
}
async function poll(){
 try{
  const j=await (await fetch('/events?offset='+off,{cache:'no-store'})).json();
  if(epoch!==null&&j.epoch!==epoch){S=fresh();off=0;epoch=j.epoch;setTimeout(poll,0);return}
  epoch=j.epoch;for(const e of j.events)apply(e);off=j.next;S.running=j.running;
  draw(j.events.length>0);
 }catch(x){$('status').textContent='cannot reach the server: '+x}
 setTimeout(poll,1000);
}
async function run(){const r=await fetch('/run',{method:'POST'});$('status').textContent=r.ok?'starting':await r.text()}
function draw(changed){
 $('status').textContent=S.error?('error: '+S.error):S.running?('running: '+S.msg):S.done?'finished':(S.clips.length?'stopped':'waiting');
 if(!changed&&S.drawn)return;S.drawn=true;
 $('info').textContent=S.header;
 $('warn').innerHTML=S.warns.map(w=>'<div>'+esc(w)+'</div>').join('');
 $('tabs').innerHTML=S.bs.map(b=>'<button class="'+(b===tab?'on':'')+'" onclick="setTab('+b+')">Batch size '+b+'</button>').join('');
 drawMaster();drawClipSelect();drawGrid();
}
function setTab(b){tab=b;draw(true)}
function f1(x){return x==null?'-':(+x).toFixed(1)}
function drawMaster(){
 const rows=S.rows[tab]||{};const names=Object.keys(rows);
 if(!names.length){$('master').innerHTML='<p>No rows yet. A row appears when the 20 ms sweep of that model is finished.</p>';return}
 let h='<table><tr><th>model<th>size MB<th>original<th>predicted<th>stable from (ms)<th>avg batch time (ms per batch of '+tab+')';
 for(const n of names){const r=rows[n];
  if(r.error){h+='<tr><td>'+esc(n)+'<td colspan=5 style="color:#b91c1c">'+esc(r.error);continue}
  const avg=r.avg!=null?f1(r.avg):(r.note?esc(r.note):(S.running?'measuring...':'-'));
  h+='<tr><td title="'+esc(S.info[n]||'')+'">'+esc(n)+'<td>'+f1(r.size)+'<td>'+r.orig+'<td>'+r.pred+'<td>'+r.stable+'<td>'+avg}
 $('master').innerHTML=h+'</table>';
}
function drawClipSelect(){
 const el=$('clip');const key=S.clips.map(c=>c.name).join('|');if(el.dataset.key===key)return;el.dataset.key=key;
 el.innerHTML='<option value="all">All clips (accuracy)</option>'+S.clips.map((c,i)=>'<option value="'+i+'">'+esc(c.name)+' ('+esc(S.labels[c.y])+')</option>').join('');sel='all';el.value='all';
}
function drawGrid(){
 sel=$('clip').value||'all';const N=S.clips.length;
 let h='<tr><th>Audio'+S.variants.map(v=>'<th title="'+esc(S.info[v]||'')+'">'+esc(v)).join('');
 for(const ms of S.ms){
  h+='<tr><td>'+ms+' ms';
  for(const v of S.variants){
   const col=S.sweep[v];const d=col?col[ms]:undefined;
   if(d===undefined){h+='<td>';continue}
   const st=S.stable[v]===ms?' st':'';
   if(d===null){h+='<td class="'+st+'">n/a';continue}
   if(sel==='all'){let ok=0;for(let i=0;i<N;i++)if(d.pred[i]===S.clips[i].y)ok++;const a=100*ok/N;
    h+='<td class="'+st+'" style="background:hsl('+(1.2*a)+',55%,82%)" onclick="cell(\''+v+'\','+ms+')">'+a.toFixed(1)+'% ('+ok+'/'+N+')'}
   else{const i=+sel;const ok=d.pred[i]===S.clips[i].y;
    h+='<td class="'+st+'" style="color:#fff;background:rgba(37,99,235,'+(0.3+0.65*d.conf[i])+');border-left:5px solid '+(ok?'#16a34a':'#dc2626')+'" onclick="cell(\''+v+'\','+ms+')">'+(ok?'&#10003; ':'&#10007; ')+S.labels[d.pred[i]]+' '+(100*d.conf[i]).toFixed(0)+'%'}
  }}
 $('grid').innerHTML=h;
}
function cell(v,ms){
 const d=S.sweep[v][ms];const N=S.clips.length;
 if(sel==='all'){let bad=[];for(let i=0;i<N;i++)if(d.pred[i]!==S.clips[i].y)bad.push(S.clips[i].name);
  $('detail').textContent=v+' at '+ms+' ms: '+(N-bad.length)+' of '+N+' clips correct'+(bad.length?'\nwrong: '+bad.join(', '):'');return}
 const i=+sel;const rank=d.p[i].map((p,k)=>[p,k]).sort((a,b)=>b[0]-a[0]).slice(0,3).map(x=>S.labels[x[1]]+' '+(100*x[0]).toFixed(1)+'%');
 $('detail').textContent=v+' at '+ms+' ms, clip '+S.clips[i].name+' (true: '+S.labels[S.clips[i].y]+'): '+rank.join('  |  ');
}
poll();
</script>"""


class RunController:
    """Starts the benchmark in a background thread and keeps its events."""

    def __init__(self, factory: EngineFactory, threads: int, cpu_name: str, clips_dir: Path = CLIPS_DIR, results_dir: Path = RESULTS_DIR,
                 variants: Optional[list[Variant]] = None) -> None:
        self.factory, self.threads, self.cpu_name = factory, threads, cpu_name
        self.clips_dir, self.results_dir = Path(clips_dir), Path(results_dir)
        self.variants = variants or build_variants()
        self.events = EventLog()
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> Optional[str]:
        """-> None when started, or the reason why not."""
        with self._lock:
            if self._running:
                return "a run is already in progress"
            self._running = True
            self.events.reset()
        threading.Thread(target=self._job, daemon=True).start()
        return None

    def _job(self) -> None:
        try:
            self.events.emit({"t": "msg", "text": "loading the model and the clips"})
            labels = self.factory.labels()
            pre = self.factory.preprocessor()
            clips, skipped = ClipLoader(self.clips_dir, labels).load()
            if not clips:
                raise ValueError("no usable clips (each clip needs at least 2 s of audio)" + (": " + "; ".join(skipped[:3]) if skipped else ""))
            counts = ", ".join(f"{sum(c.label == i for c in clips)} {name}" for i, name in enumerate(labels))
            header = (f"{len(clips)} clips ({counts}); the first 2 s of each clip is used; {self.threads} threads on {self.cpu_name}; "
                      f"{WARMUP_RUNS} warm-up runs, then {TIMED_RUNS} timed run" + (f"; skipped: {'; '.join(skipped)}" if skipped else ""))
            self.events.emit({"t": "init", "clips": [{"name": c.name, "y": c.label} for c in clips], "variants": [v.name for v in self.variants],
                              "info": {v.name: v.info for v in self.variants}, "labels": labels, "ms": SWEEP_MS, "bs": list(BATCH_SIZES), "header": header})
            writer = ResultsWriter(self.results_dir)
            writer.start_csv()
            Benchmark(clips, pre, self.factory, self.variants, labels, self.events, writer, header).run()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.events.emit({"t": "error", "text": f"{type(e).__name__}: {e}"})
        finally:
            self._running = False


def make_server(controller: RunController, port: int, readonly: bool) -> ThreadingHTTPServer:
    """The full page (readonly=False) can start a run; the read-only server only shows the page and the events."""
    page = PAGE.replace("__READONLY__", "true" if readonly else "false").encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/events":
                try:
                    off = max(0, int(parse_qs(url.query).get("offset", ["0"])[0]))
                except ValueError:
                    off = 0
                events, nxt, epoch = controller.events.since(off)
                body = json.dumps({"events": events, "next": nxt, "epoch": epoch, "running": controller.running}).encode("utf-8")
                return self._send(200, body, "application/json; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if readonly or self.path != "/run":
                return self._send(405, b"not allowed", "text/plain")
            problem = controller.start()
            self._send(409 if problem else 200, (problem or "started").encode(), "text/plain")

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


def serve() -> None:
    cpu = cpu_summary()
    threads = choose_threads(cpu, os.environ.get("HB_THREADS", ""))
    print(f"threads for every row: {threads} (HB_THREADS='{os.environ.get('HB_THREADS', '')}', logical {cpu['logical']}, physical {cpu['physical']}, visible {cpu['affinity']})", flush=True)
    controller = RunController(HfEngineFactory(MODEL_ID, threads, MODELS_DIR), threads, cpu["name"])
    view_port = int(os.environ.get("HB_VIEW_PORT", "8001"))
    threading.Thread(target=make_server(controller, view_port, True).serve_forever, daemon=True).start()
    port = int(os.environ.get("HB_PORT", "8000"))
    print(f"full page: port {port} (private); read-only live view for your supervisor: port {view_port}", flush=True)
    make_server(controller, port, False).serve_forever()


if __name__ == "__main__":
    serve()
