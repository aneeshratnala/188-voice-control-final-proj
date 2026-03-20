"""
Block until the user says "start" (Vosk + microphone).

Set VOSK_MODEL_PATH to the unzipped model directory, or place
vosk-model-small-en-us-0.15 next to this file (see README).
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import struct
import sys

import numpy as np

SAMPLE_RATE = 16000

# When SOUND_DEVICE_INDEX is not set in the environment:
#   None  = use the system default input (PortAudio default)
#   0, 1, … = use that index from the list printed under [voice] Input devices
DEFAULT_SOUND_DEVICE_INDEX: int | None = 0

# Words that begin the run (restricted grammar helps the small model a lot)
_START_TOKENS = frozenset({"start", "begin", "go"})


def _rms_int16(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack_from("<%dh" % n, data)
    return math.sqrt(sum(s * s for s in samples) / n)


def _voice_debug() -> bool:
    return os.environ.get("VOICE_DEBUG", "1").strip().lower() not in ("0", "false", "no", "off")


def _require_sounddevice():
    """Import sounddevice using the same interpreter that runs this script."""
    try:
        import sounddevice as sd
    except ImportError:
        exe = sys.executable
        print(
            "Missing package: sounddevice (microphone input).\n"
            f"Install it for THIS interpreter:\n  {exe} -m pip install sounddevice\n"
            "If you use mjpython, run:\n  mjpython -m pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)
    return sd


def _input_device():
    """Mic index: SOUND_DEVICE_INDEX env overrides DEFAULT_SOUND_DEVICE_INDEX."""
    raw = os.environ.get("SOUND_DEVICE_INDEX")
    if raw is None or raw == "":
        return DEFAULT_SOUND_DEVICE_INDEX
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SOUND_DEVICE_INDEX


def _model_dir() -> str:
    return os.environ.get(
        "VOSK_MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vosk-model-small-en-us-0.15"),
    )


def _input_device_id(sd, device: int | None) -> int | None:
    if device is not None:
        return device
    idx = sd.default.device[0]
    return idx


def _stream_channels(sd, device_id: int | None) -> int:
    """Many macOS inputs expose stereo; mono can read as silence — prefer 2 ch when available."""
    try:
        did = device_id if device_id is not None else sd.default.device[0]
        info = sd.query_devices(did, "input")
        if int(info.get("max_input_channels", 1)) >= 2:
            return 2
    except Exception:
        pass
    return 1


def _indata_to_mono_pcm16_bytes(indata) -> bytes:
    """Convert sounddevice callback ndarray to little-endian int16 PCM bytes."""
    x = np.asarray(indata, dtype=np.int16)
    if x.ndim == 2:
        ch = int(os.environ.get("VOICE_INPUT_CHANNEL", "0"))
        ch = max(0, min(ch, x.shape[1] - 1))
        x = x[:, ch]
    return np.ascontiguousarray(x, dtype=np.int16).tobytes()


def _utterance_means_start(text: str) -> bool:
    """True if the transcript is one of our start commands (whole words)."""
    if not text or not text.strip():
        return False
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        if w in _START_TOKENS:
            return True
    return False


def wait_for_start() -> None:
    """Block until a final utterance contains the word 'start' (whole word)."""
    sd = _require_sounddevice()

    try:
        from vosk import KaldiRecognizer, Model
    except ImportError:
        exe = sys.executable
        print(
            "Missing package: vosk.\n"
            f"  {exe} -m pip install vosk",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = _model_dir()
    if not os.path.isdir(model_path):
        print(f"ERROR: Vosk model not found at:\n  {model_path}", file=sys.stderr)
        print(
            "Download e.g. vosk-model-small-en-us-0.15 from "
            "https://alphacephei.com/vosk/models, unzip it here, or set VOSK_MODEL_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    # Restrict search to a tiny vocabulary — drastically reduces garbage like "by" on small models
    try:
        rec.SetGrammar(json.dumps(sorted(_START_TOKENS)))
    except Exception as e:
        print(f"[voice] warning: SetGrammar failed ({e}); recognition may be noisier.", flush=True)

    device = _input_device()
    dev_id = _input_device_id(sd, device)
    stream_ch = _stream_channels(sd, dev_id)
    try:
        if dev_id is not None:
            info = sd.query_devices(dev_id, "input")
            print(
                f"Microphone: {info['name']} "
                f"(device={dev_id}, channels={stream_ch}, default_sr={info.get('default_samplerate')})"
            )
    except Exception:
        print("Microphone: (default input device)")

    debug = _voice_debug()
    if debug:
        default_in = sd.default.device[0]
        print("[voice] Input devices (index = SOUND_DEVICE_INDEX):", flush=True)
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = " <-- default" if i == (dev_id if dev_id is not None else default_in) else ""
                print(f"  [{i}] {d['name']} (in_ch={d['max_input_channels']}){mark}", flush=True)
    if debug:
        print(
            '[voice] Debug on (set VOICE_DEBUG=0 to silence). '
            f'Vocabulary: {", ".join(sorted(_START_TOKENS))}. '
            'Say one of these, then pause briefly.',
            flush=True,
        )
    else:
        print('Say "start" (or "begin" / "go") to begin the simulation.')

    audio_q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(status, file=sys.stderr)
        audio_q.put(_indata_to_mono_pcm16_bytes(indata))

    last_partial = ""
    chunk_i = 0

    with sd.RawInputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        blocksize=4000,
        dtype="int16",
        channels=stream_ch,
        callback=callback,
    ):
        while True:
            data = audio_q.get()
            chunk_i += 1
            if debug and chunk_i % 20 == 0:
                rms = _rms_int16(data)
                print(f"[voice] audio rms={rms:.0f} (~0=silent, >500=speech likely)", flush=True)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = (result.get("text") or "").strip()
                if debug:
                    print(f'[voice] final transcript: {text!r}', flush=True)
                if _utterance_means_start(text):
                    print(f'Heard "{text}" — starting.')
                    return
                if debug and text:
                    print(
                        '[voice] no match (say: start, begin, or go).',
                        flush=True,
                    )
            else:
                partial = json.loads(rec.PartialResult())
                ptxt = (partial.get("partial") or "").strip()
                if debug and ptxt and ptxt != last_partial:
                    print(f"[voice] partial: {ptxt!r}", flush=True)
                    last_partial = ptxt
                # With grammar, partial often locks to the right word before final
                if ptxt and _utterance_means_start(ptxt):
                    print(f'Heard "{ptxt}" (partial) — starting.')
                    return
