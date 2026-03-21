# 188-voice-control-final-proj

## Run

```bash
mjpython test.py
```

`test.py`: **start** → **stack** (one-shot) or **grab** (then say **hover**, then **place**). After the red-on-green stack, say **stack** / **grab** again (or a jog word) to stack the extra blocks on the tower in order (blue → light → dark → gray). Grab uses the same waypoints as the full stack, split across voice. Simulation steps are throttled to real time (`1/control_freq`) so the viewer does not run faster than wall clock. See `voice_start.py`.

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
- `test.py` — `StackExtraCubes` + voice (**start** → **stack** / **grab** / incrementals). `StackPolicy` is unchanged; `test.py` remaps observations so later layers grasp a decor cube and place on the current tower top (`patch_stack_obs`). `stack_extra_env.StackExtraCubes` adds `decor_*_pos` observations and `check_upper_on_lower` for success after the first layer.
- `stack_extra_env.py` — optional richer **Stack** scene: register `StackExtraCubes` (same `cubeA` / `cubeB` as stock `Stack`, plus extra colored blocks). Import before `suite.make`:

```python
import stack_extra_env
import robosuite as suite
env = suite.make("StackExtraCubes", robots="Panda", ...)
```

No `StackPolicy` changes required.
