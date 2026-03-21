"""
Vosk voice gates for robosuite (blocking mic read on the main thread).

Typical flow
------------
1. ``wait_for_start()`` — say **start** → returns a loaded ``Model`` (reuse for step 2).
2. Build env, reset, create policy; pass a **between_blocks** callback that renders + zero-steps.
3. ``wait_for_stack_or_grab`` — **stack** (one-shot) or **grab** (then **hover**, **place**).

Setup: pip install vosk sounddevice numpy scipy; Vosk English model path (see VOSK_MODEL_PATH).

Env vars: VOSK_MODEL_PATH, SOUND_DEVICE_INDEX, VOICE_INPUT_CHANNEL, VOICE_DEBUG (see below).
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import threading
import time
from collections.abc import Callable
from math import gcd
from pathlib import Path
import numpy as np
from scipy import signal

VOSK_SAMPLE_RATE = 16_000
BLOCK_DURATION_S = 0.25
DEFAULT_SOUND_DEVICE_INDEX: int | None = None

KEYWORDS_OPEN = frozenset({"start"})
KEYWORDS_STACK_OR_GRAB = frozenset({"stack", "grab"})
KEYWORDS_HOVER = frozenset({"hover"})
KEYWORDS_PLACE = frozenset({"place"})
KEYWORDS_STOP = frozenset({"stop"})


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


def _text_matches(text: str, keywords: frozenset[str]) -> bool:
    if not text or not text.strip():
        return False
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        if w in keywords:
            return True
    return False


def _first_matched_keyword(text: str, keywords: frozenset[str]) -> str | None:
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        if w in keywords:
            return w
    return None


def _exit_on_stop(matched: str) -> bool:
    """If user said stop, print and terminate. Returns True if we exited."""
    if matched == "stop":
        print('Heard "stop" — ending episode and exiting.', flush=True)
        sys.exit(0)
    return False


def wait_for_keywords(
    keywords: frozenset[str],
    *,
    heard_message: str | None,
    prompt: str,
    between_blocks: Callable[[], None] | None = None,
    model: object | None = None,
    return_matched_keyword: bool = False,
    allow_stop: bool = True,
) -> object | tuple[object, str]:
    """
    Block until Vosk hears one of *keywords* (final or partial).

    Parameters
    ----------
    heard_message
        Printed when a match fires, e.g. 'Heard "start" — opening simulation.'
    prompt
        Shown once before listening.
    between_blocks
        If set, called after each audio block (~{BLOCK_DURATION_S}s) so you can
        e.g. ``env.render()`` and ``env.step(zeros)`` while the window is open.
    model
        Reuse an existing vosk ``Model`` from ``wait_for_start()``; if ``None``,
        loads the model from disk (slower).

    Returns
    -------
    The vosk ``Model``, or ``(model, matched_word)`` if *return_matched_keyword* is True.

    If *allow_stop* is True, ``stop`` is always allowed and ends the program immediately.
    """
    kw = keywords | KEYWORDS_STOP if allow_stop else keywords

    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Install sounddevice:\n"
            f"  {sys.executable} -m pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from vosk import KaldiRecognizer, Model
    except ImportError:
        print(f"Install vosk:\n  {sys.executable} -m pip install vosk", file=sys.stderr)
        sys.exit(1)

    model_path = _model_dir()
    if not model_path.is_dir():
        print(
            f"No Vosk model at:\n  {model_path}\n"
            "https://alphacephei.com/vosk/models — unzip or set VOSK_MODEL_PATH.",
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

    native_frames = max(1024, int(round(BLOCK_DURATION_S * stream_sr)))

    if model is None:
        model = Model(str(model_path))
        if dbg:
            print(f"[voice] Model: {model_path}", flush=True)
            print("[voice] Input-capable devices:", flush=True)
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    mark = (
                        " *"
                        if i == (dev_arg if dev_arg is not None else sd.default.device[0])
                        else ""
                    )
                    print(f"    [{i}] {d['name']}{mark}", flush=True)
    if dbg and model is not None:
        print(
            f"[voice] Mic: {info.get('name', '?')}  device={dev_arg!r}  "
            f"{stream_sr} Hz  {num_ch} ch  block={native_frames}",
            flush=True,
        )

    rec = KaldiRecognizer(model, VOSK_SAMPLE_RATE)
    kw_list = sorted(kw)
    try:
        rec.SetGrammar(json.dumps(kw_list))
        if dbg:
            print(f"[voice] Grammar: on {kw_list}", flush=True)
    except Exception as e:
        if dbg:
            print(f"[voice] Grammar: off ({e})", flush=True)

    print(prompt, flush=True)

    n_block = 0
    last_partial = ""

    try:
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
                if len(pcm) >= 2:
                    n_block += 1
                    if dbg and n_block % 20 == 0:
                        rms, peak = _rms_peak_int16(pcm)
                        print(
                            f"[voice] →Vosk  rms={rms:.0f}  peak={peak}",
                            flush=True,
                        )

                    if rec.AcceptWaveform(pcm):
                        text = (json.loads(rec.Result()).get("text") or "").strip()
                        if dbg and text:
                            print(f"[voice] final: {text!r}", flush=True)
                        matched = _first_matched_keyword(text, kw)
                        if matched is not None:
                            _exit_on_stop(matched)
                            msg = heard_message or 'Heard "{word}" — starting motion.'
                            print(msg.replace("{word}", matched), flush=True)
                            if return_matched_keyword:
                                return model, matched
                            return model
                    else:
                        partial = (
                            json.loads(rec.PartialResult()).get("partial") or ""
                        ).strip()
                        if partial and partial != last_partial and dbg:
                            print(f"[voice] partial: {partial!r}", flush=True)
                            last_partial = partial
                        matched = _first_matched_keyword(partial, kw)
                        if matched is not None:
                            _exit_on_stop(matched)
                            msg = heard_message or 'Heard "{word}" — starting motion.'
                            print(msg.replace("{word}", matched), flush=True)
                            if return_matched_keyword:
                                return model, matched
                            return model

                if between_blocks is not None:
                    between_blocks()

    except KeyboardInterrupt:
        print("\n[voice] cancelled.", flush=True)
        raise


_STOP_HINT = ' Say "stop" to quit anytime.'


def wait_for_start() -> object:
    """Say **start** to proceed (loads Vosk model; returns it for the next voice step)."""
    return wait_for_keywords(
        KEYWORDS_OPEN,
        heard_message='Heard "start" — opening simulation.',
        prompt='Say "start" to open the simulation, then wait for the window.' + _STOP_HINT,
        between_blocks=None,
        model=None,
        return_matched_keyword=False,
    )


def wait_for_stack_or_grab(model: object, *, between_blocks: Callable[[], None]) -> str:
    """
    Say **stack** (full stack in one go) or **grab** (then say **hover**, then **place**).

    Returns ``\"stack\"`` or ``\"grab\"``.
    """
    _, cmd = wait_for_keywords(
        KEYWORDS_STACK_OR_GRAB,
        heard_message=None,
        prompt=(
            'Say "stack" for one-shot stack, or "grab" to run grab → hover → place in steps.'
            + _STOP_HINT
        ),
        between_blocks=between_blocks,
        model=model,
        return_matched_keyword=True,
    )
    return cmd


def wait_for_hover(model: object, *, between_blocks: Callable[[], None]) -> None:
    """Say **hover** to move from post-grasp hold to above the green cube."""
    wait_for_keywords(
        KEYWORDS_HOVER,
        heard_message=None,
        prompt='Say "hover" to move to the hover pose above the green block.' + _STOP_HINT,
        between_blocks=between_blocks,
        model=model,
        return_matched_keyword=False,
    )


def wait_for_place(model: object, *, between_blocks: Callable[[], None]) -> None:
    """Say **place** to run the existing place-down + release on the green cube."""
    wait_for_keywords(
        KEYWORDS_PLACE,
        heard_message=None,
        prompt='Say "place" to lower and release on the green block.' + _STOP_HINT,
        between_blocks=between_blocks,
        model=model,
        return_matched_keyword=False,
    )


class StopMicMonitor:
    """
    Listens for **stop** on a background thread while the main thread runs the policy.

    Call ``pause()`` before any ``wait_for_*`` that opens the mic on the main thread,
    then ``resume()`` after. Call ``check()`` each control step to exit if stop was heard.
    """

    def __init__(self, model: object) -> None:
        self._model = model
        self._heard_stop = threading.Event()
        self._listen = threading.Event()
        self._shutdown = threading.Event()
        self._th: threading.Thread | None = None

    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def pause(self) -> None:
        self._listen.clear()

    def resume(self) -> None:
        if not self._shutdown.is_set():
            self._listen.set()

    def check(self) -> None:
        if self._heard_stop.is_set():
            print('Heard "stop" — ending episode and exiting.', flush=True)
            sys.exit(0)

    def close(self) -> None:
        self._shutdown.set()
        self._listen.set()
        if self._th is not None and self._th.is_alive():
            self._th.join(timeout=3.0)

    def _loop(self) -> None:
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer
        except ImportError:
            return

        while not self._shutdown.is_set():
            if not self._listen.wait(timeout=0.15):
                continue
            if self._shutdown.is_set():
                break

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

            native_frames = max(1024, int(round(BLOCK_DURATION_S * stream_sr)))
            rec = KaldiRecognizer(self._model, VOSK_SAMPLE_RATE)
            try:
                rec.SetGrammar(json.dumps(["stop"]))
            except Exception:
                pass

            try:
                with sd.InputStream(
                    device=dev_arg,
                    channels=num_ch,
                    samplerate=stream_sr,
                    dtype="float32",
                    latency="high",
                ) as stream:
                    while self._listen.is_set() and not self._shutdown.is_set():
                        block, _ = stream.read(native_frames)
                        pcm = _float_block_to_pcm16k(block, stream_sr)
                        if len(pcm) < 2:
                            continue
                        if rec.AcceptWaveform(pcm):
                            text = (json.loads(rec.Result()).get("text") or "").strip()
                            if _first_matched_keyword(text, KEYWORDS_STOP):
                                self._heard_stop.set()
                                return
                        else:
                            partial = (
                                json.loads(rec.PartialResult()).get("partial") or ""
                            ).strip()
                            if _first_matched_keyword(partial, KEYWORDS_STOP):
                                self._heard_stop.set()
                                return
            except Exception as e:
                if _debug():
                    print(f"[voice] stop monitor: {e}", flush=True)
                time.sleep(0.3)


__all__ = [
    "KEYWORDS_HOVER",
    "KEYWORDS_OPEN",
    "KEYWORDS_PLACE",
    "KEYWORDS_STACK_OR_GRAB",
    "KEYWORDS_STOP",
    "StopMicMonitor",
    "wait_for_keywords",
    "wait_for_hover",
    "wait_for_place",
    "wait_for_stack_or_grab",
    "wait_for_start",
]
