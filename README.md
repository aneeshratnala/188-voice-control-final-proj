# 188-voice-control-final-proj

## Run

```bash
mjpython test.py
```

`test.py` calls `wait_for_start()` from `voice_start.py` first (spoken **start**, **begin**, or **go**), then runs robosuite.

## Voice setup

Install into the **same** Python as `mjpython`:

```bash
mjpython -m pip install -r requirements.txt
```

Download an English Vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models), unzip **`vosk-model-small-en-us-0.15`** next to `voice_start.py`, or set **`VOSK_MODEL_PATH`** to the model folder.

Optional env vars and behavior are documented in the **module docstring** at the top of `voice_start.py`.

### If recognition is bad

- Prefer a **larger** model (e.g. `vosk-model-en-us-0.22`) via `VOSK_MODEL_PATH`.
- **`SetGrammar` missing** (`vosk_recognizer_set_grm … symbol not found`): reinstall vosk in that env: `mjpython -m pip install --force-reinstall vosk`.
- macOS: allow the microphone for the app that runs `mjpython` (**Privacy & Security → Microphone**).

## Code layout

- `voice_start.py` — Vosk + blocking mic read (no callback queue), resample with **scipy.signal.resample_poly** to 16 kHz for Vosk.
- `policies.py` — task policies.
- `test.py` — env + policy loop.
