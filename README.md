# Voice-Controlled Robot Arm (CS 188 Final Project)

**UCLA CS 188**

This project integrates **offline speech recognition (Vosk)** with **Robosuite** manipulation tasks running in **MuJoCo**. A microphone captures spoken commands; audio is resampled to 16 kHz and decoded with a small English Vosk model and a **grammar-restricted vocabulary**. Recognized keywords drive either **full-task policies** (e.g. `stack`, `assemble`), **segmented** grasp–hover–place phases, or **incremental** world-frame jog bursts. A **PID controller** tracks waypoints and end-effector motion at the simulation control rate (20 Hz in our scripts).

**Project website (figures, demo video, write-up):** open [`project.html`](./project.html) in a browser (double-click or `open project.html` on macOS). Assets live under [`website_assets/`](./website_assets/) (e.g. `DEMO_VIDEO.mov`, screenshots, `flowchart.png`).

---

## Prerequisites

| Component | Notes |
|-----------|--------|
| **Python** | 3.9+ recommended (use the **same** interpreter as `mjpython` / MuJoCo). |
| **MuJoCo + Robosuite** | Required for `test.py` and `test_nut_assembly.py`. Install per [Robosuite installation](https://robosuite.ai/docs/installation/installation.html) and your course environment. On macOS, **`mjpython`** is typically used so the MuJoCo viewer (OpenGL) works with the Robosuite stack. |
| **Microphone** | Required for voice control. On macOS: **System Settings → Privacy & Security → Microphone** — allow the terminal/app that runs Python. |
| **Vosk model** | English small model (~40 MB), e.g. **`vosk-model-small-en-us-0.15`**. |

`requirements.txt` lists **voice/audio/math** dependencies only (`vosk`, `sounddevice`, `numpy`, `scipy`). **Robosuite / MuJoCo are not pinned** here because versions are usually fixed by the course conda/env; install those first, then install from `requirements.txt` into that same environment.

---

## Installation (reproducible setup)

1. **Clone or copy** this repository and `cd` into it.

2. **Install Robosuite + MuJoCo** (if not already in your CS 188 environment). Verify:
   ```bash
   mjpython -c "import robosuite; print(robosuite.__version__)"
   ```

3. **Install Python dependencies** into the **same** environment as `mjpython`:
   ```bash
   mjpython -m pip install -r requirements.txt
   ```

4. **Vosk model** — either:
   - Download **`vosk-model-small-en-us-0.15`** from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models), unzip it **next to** `voice_start.py` so the folder is:
     ```
     188-voice-control-final-proj/vosk-model-small-en-us-0.15/
     ```
   - **Or** set **`VOSK_MODEL_PATH`** to the absolute path of that unzipped folder (see below).

5. **Optional: audio device** — if the wrong mic is used, set `SOUND_DEVICE_INDEX` or `VOICE_INPUT_CHANNEL` (see [Environment variables](#environment-variables)).

---

## Running the demos

Use **`mjpython`** (or your course’s MuJoCo-enabled Python) from the **repository root** so imports and the default Vosk path resolve correctly.

### Multi-cube stacking (custom `StackExtraCubes` environment)

```bash
mjpython test.py
```

Flow: say **`start`** to begin → then **`stack`** (full autonomous stack) or **`grab`** (segmented: **`hover`**, **`place`**) or **incremental** jog words (`up`, `down`, `forward`, …). After the first layer, continue stacking extra blocks with **`stack`** / **`grab`** / jogs. Simulation is **throttled to wall clock** (`1/control_freq`) so the viewer stays in sync with real time.

### Nut assembly (stock `NutAssembly` task)

```bash
mjpython test_nut_assembly.py
```

Flow: **`start`** → **`assemble`** (full run) or **`grab`** with **`hover`** / **`place`**, or incremental commands as in stacking.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| **`VOSK_MODEL_PATH`** | Absolute path to the unzipped Vosk model directory. If unset, code uses `./vosk-model-small-en-us-0.15` next to `voice_start.py`. |
| **`SOUND_DEVICE_INDEX`** | Integer index of the input device (see `sounddevice` / PortAudio). If unset, default device is used. |
| **`VOICE_INPUT_CHANNEL`** | Channel index for multi-channel devices (default `0`). |
| **`VOICE_DEBUG`** | Set to `0`, `false`, `no`, or `off` to reduce voice debug prints (default is verbose). |

Additional behavior and keyword wiring are documented in the **module docstring** at the top of [`voice_start.py`](./voice_start.py).

---

## Repository layout

| Path | Role |
|------|------|
| [`project.html`](./project.html) | Project report / website (open in browser). |
| [`website_assets/`](./website_assets/) | Demo video, screenshots, pipeline flowchart for the site. |
| [`voice_start.py`](./voice_start.py) | Vosk loading, mic capture, resampling to 16 kHz, `SetGrammar`, blocking voice “gates” (`wait_for_start`, `wait_for_stack_grab_or_incremental`, etc.). |
| [`policies.py`](./policies.py) | Stack / nut / incremental teleop policies; uses [`pid.py`](./pid.py) for control. |
| [`pid.py`](./pid.py) | PID implementation used by policies. |
| [`stack_extra_env.py`](./stack_extra_env.py) | Registers **`StackExtraCubes`** (extra decor cubes on top of stock Stack). Used only by **`test.py`**. |
| [`test.py`](./test.py) | Voice-driven **stacking** demo (`StackExtraCubes`). |
| [`test_nut_assembly.py`](./test_nut_assembly.py) | Voice-driven **nut assembly** demo (`NutAssembly`). |
| [`vosk-model-small-en-us-0.15/`](./vosk-model-small-en-us-0.15/) | Vosk weights (~40 MB). Download and unzip next to `voice_start.py`, or set `VOSK_MODEL_PATH`. Large; consider `.gitignore` if you do not want it in version control. |
| [`requirements.txt`](./requirements.txt) | `vosk`, `sounddevice`, `numpy`, `scipy`. |

---

## Troubleshooting

- **Recognition is poor** — Try a **larger** Vosk model via `VOSK_MODEL_PATH` (e.g. `vosk-model-en-us-0.22`).
- **`SetGrammar` / `vosk_recognizer_set_grm` missing** — Reinstall Vosk in the same env as `mjpython`: `mjpython -m pip install --force-reinstall vosk`.
- **Microphone denied (macOS)** — Allow the microphone for Terminal, iTerm, or the IDE that launches `mjpython`.
- **`ModuleNotFoundError: robosuite`** — Install Robosuite in this environment; `requirements.txt` does not install it.
- **Viewer / OpenGL issues** — Use `mjpython` as provided with your MuJoCo install; avoid mixing a plain `python` without MuJoCo bindings for these scripts.

---

## Team / attribution

See **Contributions** and **Note on AI Use** in [`project.html`](./project.html).
