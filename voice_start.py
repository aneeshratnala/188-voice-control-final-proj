"""
Wait for a spoken keyword before the sim runs (Vosk + sounddevice).

Setup
-----
1. pip install vosk sounddevice numpy scipy  (same Python as mjpython)
2. Unzip an English Vosk model into this folder, or set VOSK_MODEL_PATH.

Environment (optional)
----------------------
  VOSK_MODEL_PATH      Path to unzipped model directory
  SOUND_DEVICE_INDEX   Integer mic index (see printed device list)
  VOICE_INPUT_CHANNEL  0 or 1 when using stereo capture (default 0)
  VOICE_DEBUG          1 = logs (default), 0 = quiet
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

# --- config -----------------------------------------------------------------

VOSK_SAMPLE_RATE = 16_000
KEYWORDS = frozenset({"start", "begin", "go"})
# Seconds of audio per read (blocking, main thread — avoids callback/thread issues with mjpython)
BLOCK_DURATION_S = 0.25

# Default mic when SOUND_DEVICE_INDEX unset (None = PortAudio default device)
DEFAULT_SOUND_DEVICE_INDEX: int | None = None


def _debug() -> bool:
    return os.environ.get("VOICE_DEBUG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _model_dir() -> Path:
    env = os.environ.get("VOSK_MODEL_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    return here / "vosk-model-small-en-us-0.15"


def _device_index() -> int | None:
    raw = os.environ.get("SOUND_DEVICE_INDEX", "").strip()
    if not raw:
        return DEFAULT_SOUND_DEVICE_INDEX
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SOUND_DEVICE_INDEX


def _input_channel() -> int:
    try:
        return max(0, int(os.environ.get("VOICE_INPUT_CHANNEL", "0")))
    except ValueError:
        return 0


def _rms_peak_int16(pcm: bytes) -> tuple[float, int]:
    n = len(pcm) // 2
    if n == 0:
        return 0.0, 0
    samples = struct.unpack_from("<%dh" % n, pcm)
    rms = math.sqrt(sum(s * s for s in samples) / n)
    peak = max(abs(s) for s in samples)
    return rms, peak


def _float_block_to_pcm16k(block: np.ndarray, stream_sr: int) -> bytes:
    """
    One capture block: float32 (frames, ch) → mono int16 little-endian @ VOSK_SAMPLE_RATE.
    """
    x = np.asarray(block, dtype=np.float32, order="C")
    if x.ndim == 1:
        mono = x
    else:
        ch = min(_input_channel(), x.shape[1] - 1)
        mono = np.ascontiguousarray(x[:, ch], dtype=np.float32)

    np.nan_to_num(mono, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak < 1e-8:
        n_out = max(
            1,
            int(round(len(mono) * VOSK_SAMPLE_RATE / stream_sr)),
        )
        return (np.zeros(n_out, dtype=np.int16)).tobytes()

    if peak > 1.0 + 1e-5:
        mono = mono / peak

    if stream_sr != VOSK_SAMPLE_RATE:
        g = gcd(int(stream_sr), VOSK_SAMPLE_RATE)
        up = VOSK_SAMPLE_RATE // g
        down = int(stream_sr) // g
        mono = signal.resample_poly(mono, up, down).astype(np.float32, copy=False)
        p2 = float(np.max(np.abs(mono))) if mono.size else 0.0
        if p2 > 1.0 + 1e-5:
            mono = mono / p2

    mono = np.clip(mono * (32767.0 * 0.9), -32768.0, 32767.0)
    return mono.astype(np.int16, copy=False).tobytes()


def _text_triggers(text: str) -> bool:
    if not text or not text.strip():
        return False
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        if w in KEYWORDS:
            return True
    return False


def wait_for_start() -> None:
    """Block until the user says start / begin / go (final or partial transcript)."""
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Install sounddevice for this interpreter:\n"
            f"  {sys.executable} -m pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from vosk import KaldiRecognizer, Model
    except ImportError:
        print(
            f"Install vosk:\n  {sys.executable} -m pip install vosk",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = _model_dir()
    if not model_path.is_dir():
        print(
            f"No Vosk model at:\n  {model_path}\n"
            "Download from https://alphacephei.com/vosk/models and unzip, "
            "or set VOSK_MODEL_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    dbg = _debug()
    dev_arg = _device_index()

    try:
        dev_id = dev_arg if dev_arg is not None else sd.default.device[0]
        info = sd.query_devices(dev_id, "input")
        stream_sr = int(round(float(info["default_samplerate"])))
        if not (8_000 <= stream_sr <= 192_000):
            stream_sr = 48_000
        max_ch = int(info.get("max_input_channels", 1))
        num_ch = 2 if max_ch >= 2 else 1
    except Exception:
        stream_sr = 48_000
        num_ch = 1
        info = {"name": "default", "default_samplerate": stream_sr}
        dev_id = None

    native_frames = max(1024, int(round(BLOCK_DURATION_S * stream_sr)))

    if dbg:
        print(f"[voice] Model: {model_path}", flush=True)
        print(
            f"[voice] Mic: {info.get('name', '?')}  device={dev_arg!r}  "
            f"{stream_sr} Hz  {num_ch} ch  block={native_frames} frames",
            flush=True,
        )
        print("[voice] Input-capable devices:", flush=True)
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                mark = (
                    " *"
                    if i == (dev_arg if dev_arg is not None else sd.default.device[0])
                    else ""
                )
                print(f"    [{i}] {d['name']}{mark}", flush=True)

    model = Model(str(model_path))
    rec = KaldiRecognizer(model, VOSK_SAMPLE_RATE)

    try:
        rec.SetGrammar(json.dumps(sorted(KEYWORDS)))
        if dbg:
            print("[voice] Grammar: on (start | begin | go)", flush=True)
    except Exception as e:
        if dbg:
            print(f"[voice] Grammar: off ({e})", flush=True)

    print(
        'Say "start", "begin", or "go" — then pause half a second.',
        flush=True,
    )

    n_block = 0
    last_partial = ""

    try:
        # No callback: blocking read() on the main thread (reliable with mjpython).
        with sd.InputStream(
            device=dev_arg,
            channels=num_ch,
            samplerate=stream_sr,
            dtype="float32",
            latency="high",
        ) as stream:
            while True:
                block, overflowed = stream.read(native_frames)
                if overflowed and dbg:
                    print("[voice] overflow (dropped samples)", flush=True)

                pcm = _float_block_to_pcm16k(block, stream_sr)
                if len(pcm) < 2:
                    continue

                n_block += 1
                if dbg and n_block % 20 == 0:
                    rms, peak = _rms_peak_int16(pcm)
                    print(
                        f"[voice] →Vosk  rms={rms:.0f}  peak={peak}  "
                        f"(typical speech: rms hundreds–few k, peak < 30k)",
                        flush=True,
                    )

                if rec.AcceptWaveform(pcm):
                    text = (json.loads(rec.Result()).get("text") or "").strip()
                    if dbg and text:
                        print(f"[voice] final: {text!r}", flush=True)
                    if _text_triggers(text):
                        print(f'Heard "{text}" — starting.', flush=True)
                        return
                else:
                    partial = (json.loads(rec.PartialResult()).get("partial") or "").strip()
                    if partial and partial != last_partial and dbg:
                        print(f"[voice] partial: {partial!r}", flush=True)
                        last_partial = partial
                    if _text_triggers(partial):
                        print(f'Heard "{partial}" (partial) — starting.', flush=True)
                        return

    except KeyboardInterrupt:
        print("\n[voice] cancelled.", flush=True)
        raise


__all__ = ["wait_for_start"]
