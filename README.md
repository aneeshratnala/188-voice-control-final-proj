# 188-voice-control-final-proj

## Voice: say START before the run

1. `pip install vosk sounddevice`
2. Download a Vosk English model (e.g. [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models)), unzip it, and either:
   - Put the folder `vosk-model-small-en-us-0.15` in this project directory, or
   - Set `VOSK_MODEL_PATH` to the unzipped model path.

Then run `mjpython test.py` — the script waits until it hears **start**, **begin**, or **go** (restricted vocabulary so the small Vosk model doesn’t guess random words). For better accuracy, use a larger model (e.g. `vosk-model-en-us-0.22`) via `VOSK_MODEL_PATH`.

Install deps with the **same interpreter** you use to run `test.py` (Conda `mjpython` and system `python3` do not share packages):

```bash
# If you run: mjpython test.py
mjpython -m pip install -r requirements.txt

# If you run: python3 test.py
python3 -m pip install -r requirements.txt
```

To see which Python you are using: `python3 -c "import sys; print(sys.executable)"` (or replace `python3` with `mjpython`).

On macOS, allow the microphone for **Terminal** or **Cursor** under **System Settings → Privacy & Security → Microphone**. Set the default input under **Sound → Input** (e.g. MacBook microphone).

To force a specific input: set `DEFAULT_SOUND_DEVICE_INDEX` at the top of `voice_start.py` (e.g. `0` for the first device in the debug list), or run `SOUND_DEVICE_INDEX=0 mjpython test.py` (env overrides the default).

While waiting for **start**, the script prints partial/final transcripts and occasional **audio rms** (mic level). Say “start” clearly, then **pause briefly** so Vosk can finalize. Set `VOICE_DEBUG=0` to turn that logging off.

If **audio rms stays ~0** while you talk, the wrong input may be selected: use the printed device list and set `SOUND_DEVICE_INDEX` to your MacBook mic. On some Macs, stereo capture is required; the code opens 2 channels when available. If still silent, try `VOICE_INPUT_CHANNEL=1`.