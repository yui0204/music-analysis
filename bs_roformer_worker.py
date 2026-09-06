"""BS RoFormer inference worker for SW 6-stem and Mega53 acoustic guitar."""
import argparse
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent
RATE = 44100

MEGA53_BASE = "https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems/resolve/main/v1"
ACOUSTIC_CHECKPOINT_URL = MEGA53_BASE + "/bs_mega_53stem_acoustic-guitar_mvsep.ckpt?download=true"
ACOUSTIC_CONFIG_URL = MEGA53_BASE + "/bs_mega_53stem_acoustic-guitar_mvsep_config.yaml?download=true"

MODELS = {
    "bs_sw": {
        "slug": "roformer-model-bs-roformer-sw-by-jarredou",
        "label": "BS RoFormer SW",
        "stems": ("vocals", "drums", "bass", "guitar", "piano", "other"),
    },
    "bs_acoustic_guitar": {
        "slug": "bs_mega_53stem_acoustic-guitar_mvsep",
        "label": "BS RoFormer Mega53 Acoustic Guitar",
        "stems": ("acoustic-guitar", "other"),
        "align_chunk": True,
        "max_chunk_seconds": 10,
        "checkpoint_url": ACOUSTIC_CHECKPOINT_URL,
        "config_url": ACOUSTIC_CONFIG_URL,
    },
    "bs_sw_acoustic_guitar": {
        "slug": "bs_sw_to_bs_mega_53stem_acoustic-guitar_mvsep",
        "label": "BS RoFormer SW to Mega53 Acoustic Guitar",
        "stems": ("vocals", "drums", "bass", "guitar", "piano", "other", "acoustic-guitar", "guitar-other"),
        "cascade": True,
        "align_chunk": True,
        "max_chunk_seconds": 10,
        "checkpoint_url": ACOUSTIC_CHECKPOINT_URL,
        "config_url": ACOUSTIC_CONFIG_URL,
    },
}


def output_subtype(output_format):
    if output_format != "int24":
        raise ValueError("保存形式は24-bit WAVのみ対応しています。")
    return "PCM_24"


def download_model(url, path):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {path.name}...", flush=True)
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        received, reported = 0, -1
        with partial.open("wb") as output:
            for block in response.iter_content(1024 * 1024):
                if not block:
                    continue
                output.write(block)
                received += len(block)
                percent = int(100 * received / total) if total else received // (10 * 1024 * 1024)
                if percent // 10 != reported:
                    print(f"Downloading: {received / 1024**2:.0f} MB", flush=True)
                    reported = percent // 10
        if total and received != total:
            raise RuntimeError("モデルのダウンロードが中断されました。再実行してください。")
    partial.replace(path)
    return path


def convert_outputs(store, output, source_stem, stems, output_format, mix_path=None):
    output.mkdir(parents=True, exist_ok=True)
    subtype = output_subtype(output_format)
    arrays = []
    for stem in stems:
        produced = store / f"{source_stem}_{stem}.wav"
        if stem == "other" and not produced.is_file():
            primary = store / f"{source_stem}_acoustic-guitar.wav"
            if mix_path is not None and primary.is_file():
                mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
                acoustic, acoustic_sr = sf.read(primary, dtype="float32", always_2d=True)
                if sr != acoustic_sr:
                    raise RuntimeError(f"other.wav を生成できませんでした。サンプルレートが一致しません: {sr} / {acoustic_sr}")
                length = min(len(mix), len(acoustic))
                sf.write(produced, mix[:length] - acoustic[:length], sr, subtype="FLOAT")
        if not produced.is_file():
            raise RuntimeError(f"{stem}.wav を生成できませんでした。")
        data, sr = sf.read(produced, dtype="float32", always_2d=True)
        if sr != RATE:
            raise RuntimeError(f"{stem}.wav のサンプルレートが想定外です: {sr}")
        if not np.isfinite(data).all():
            raise RuntimeError(f"{stem}.wav に不正な値があります。")
        arrays.append((stem, data))
    peak = max((float(np.max(np.abs(data))) for _, data in arrays), default=0.0)
    gain = min(1.0, 0.99 / max(peak, 1e-10))
    for stem, data in arrays:
        partial = output / f"{stem}.partial.wav"
        sf.write(partial, data * gain, RATE, subtype=subtype)
        partial.replace(output / f"{stem}.wav")


def write_outputs(output, arrays, output_format):
    output.mkdir(parents=True, exist_ok=True)
    subtype = output_subtype(output_format)
    peak = max((float(np.max(np.abs(data))) for _, data in arrays), default=0.0)
    gain = min(1.0, 0.99 / max(peak, 1e-10))
    for stem, data in arrays:
        partial = output / f"{stem}.partial.wav"
        sf.write(partial, data * gain, RATE, subtype=subtype)
        partial.replace(output / f"{stem}.wav")
    return gain


def read_named(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != RATE:
        raise RuntimeError(f"{path.name} のサンプルレートが想定外です: {sr}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"{path.name} に不正な値があります。")
    return data


def collect_sw_and_acoustic(sw_store, acoustic_store, output, source_stem, acoustic_source_stem, output_format):
    arrays = []
    for stem in MODELS["bs_sw"]["stems"]:
        arrays.append((stem, read_named(sw_store / f"{source_stem}_{stem}.wav")))
    acoustic = read_named(acoustic_store / f"{acoustic_source_stem}_acoustic-guitar.wav")
    guitar = dict(arrays)["guitar"]
    length = min(len(guitar), len(acoustic))
    arrays.append(("acoustic-guitar", acoustic[:length]))
    arrays.append(("guitar-other", guitar[:length] - acoustic[:length]))
    gain = write_outputs(output, arrays, output_format)


def acoustic_assets():
    folder = ROOT / ".cache" / "bs_roformer" / "bs_mega_53stem_acoustic-guitar_mvsep"
    checkpoint = download_model(ACOUSTIC_CHECKPOINT_URL, folder / "bs_mega_53stem_acoustic-guitar_mvsep.ckpt")
    config = download_model(ACOUSTIC_CONFIG_URL, folder / "bs_mega_53stem_acoustic-guitar_mvsep_config.yaml")
    return checkpoint, config


def aligned_config(source_config, max_chunk_seconds=None):
    suffix = "_safe" if max_chunk_seconds else "_aligned"
    target = source_config.with_name(source_config.stem + suffix + source_config.suffix)
    if target.is_file() and target.stat().st_mtime >= source_config.stat().st_mtime:
        return target
    config = yaml.load(source_config.read_text(), Loader=yaml.FullLoader)
    hop = int(config["model"].get("stft_hop_length", 512))
    config.setdefault("inference", {})
    original = int(config["inference"].get("chunk_size", config.get("audio", {}).get("chunk_size", 588800)))
    limit = int(RATE * max_chunk_seconds) if max_chunk_seconds else original
    chunk = min(original, limit)
    config["inference"]["batch_size"] = 1
    config["inference"]["chunk_size"] = max(hop, chunk - (chunk % hop))
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target


def torch_device(preferred=None):
    if preferred == "mps":
        try:
            import torch
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return preferred



def configure_mlx_limits(backend):
    if backend != "mlx":
        return
    try:
        import mlx.core as mx
        limit = 4 * 1024 * 1024 * 1024
        cache_limit = 512 * 1024 * 1024
        mx.set_memory_limit(limit)
        mx.set_cache_limit(cache_limit)
        print(f"MLX memory limit: {limit // 1024**3} GB / cache {cache_limit // 1024**2} MB", flush=True)
    except Exception as error:
        print(f"MLX memory limitを設定できませんでした: {error}", flush=True)

def run_with_session(model_slug, input_folder, store_dir, model_path=None, config_path=None, backend=None, device=None):
    from bs_roformer import BSRoformerSession

    backend = backend or ("mlx" if platform.machine() == "arm64" else "auto")
    device = torch_device(device)
    configure_mlx_limits(backend)
    print(f"BS RoFormer backend: {backend}" + (f" / {device}" if device else ""), flush=True)
    with BSRoformerSession(model_name=model_slug, model_path=model_path, config_path=config_path, backend=backend, device=device) as session:
        session.load()
        info = session.cache_info()
        print(f"BS RoFormer active: {info.get('backend')} / {info.get('device')}", flush=True)
        session.infer(input_folder, store_dir=store_dir, output_format="wav_float32")
        return info


def run_with_legacy_cli(model_slug, input_folder, store_dir, model_path=None, config_path=None):
    from bs_roformer.inference import proc_folder

    device = "mps" if platform.machine() == "arm64" else "cpu"
    print(f"BS RoFormer backend: torch / {device}", flush=True)
    args = ["--input_folder", str(input_folder), "--store_dir", str(store_dir), "--device", device]
    if model_path and config_path:
        args.extend(["--model_path", str(model_path), "--config_path", str(config_path)])
    else:
        args.extend(["--model", model_slug])
    proc_folder(args)
    return {}


def infer_model(info, input_folder, store_dir):
    model_path = config_path = None
    if info.get("checkpoint_url"):
        model_path, config_path = acoustic_assets()
        if info.get("align_chunk"):
            config_path = aligned_config(config_path, info.get("max_chunk_seconds"))
    backend = info.get("backend")
    device = info.get("device")
    try:
        return run_with_session(info["slug"], input_folder, store_dir, model_path, config_path, backend, device)
    except (ImportError, AttributeError):
        return run_with_legacy_cli(info["slug"], input_folder, store_dir, model_path, config_path)
    except Exception as error:
        try:
            from bs_roformer.backends.base import BackendUnavailable
        except ImportError:
            BackendUnavailable = ()
        message = str(error)
        mlx_startup_error = any(fragment in message for fragment in (
            "No Metal device available",
            "MLX",
            "mlx",
            "Metal",
        ))
        if backend == "torch" or not (isinstance(error, BackendUnavailable) or mlx_startup_error):
            raise
        print(f"MLXで実行できないためTorch/MPSで再試行します: {error}", flush=True)
        return run_with_session(info["slug"], input_folder, store_dir, model_path, config_path, "torch", torch_device("mps"))


def decode_source(source, target):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vn",
                    "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_f32le", str(target)], check=True)


def run(kind, source, output, output_format):
    if kind not in MODELS:
        raise ValueError(kind)
    info = MODELS[kind]
    print(f"モデルを準備中: {info['label']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = ROOT / ".cache" / "tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bs-roformer-", dir=tmp_parent) as directory:
        root = Path(directory)
        input_folder = root / "input"
        store_dir = root / "store"
        input_folder.mkdir()
        store_dir.mkdir()
        decoded = input_folder / (source.stem + ".wav")
        decode_source(source, decoded)
        print(f"Separating track {source}", flush=True)
        if info.get("cascade"):
            sw_store = root / "sw"
            sw_store.mkdir()
            print("1段目: BS RoFormer SWでguitarを抽出", flush=True)
            sw_cache = infer_model(MODELS["bs_sw"], input_folder, sw_store)
            guitar = sw_store / f"{source.stem}_guitar.wav"
            if not guitar.is_file():
                raise RuntimeError("1段目のguitar.wavを生成できませんでした。")
            acoustic_input = root / "acoustic-input"
            acoustic_input.mkdir()
            acoustic_source = acoustic_input / f"{source.stem}_guitar.wav"
            read_named(guitar)
            sf.write(acoustic_source, read_named(guitar), RATE, subtype="FLOAT")
            print("2段目: guitarからacoustic-guitarを抽出", flush=True)
            cache = infer_model(MODELS["bs_acoustic_guitar"], acoustic_input, store_dir)
            cache = {"stage1": sw_cache, "stage2": cache}
            collect_sw_and_acoustic(sw_store, store_dir, output, source.stem, f"{source.stem}_guitar", output_format)
            print("STEM_PROGRESS 100", flush=True)
            return
        else:
            cache = infer_model(info, input_folder, store_dir)
            convert_stem = source.stem
        print("STEM_PROGRESS 96", flush=True)
        mix_for_other = acoustic_source if info.get("cascade") else decoded
        convert_outputs(store_dir, output, convert_stem, info["stems"], output_format, mix_for_other)
    print("STEM_PROGRESS 100", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=tuple(MODELS))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("format", choices=("int24",))
    args = parser.parse_args()
    run(args.kind, args.source, args.output, args.format)
