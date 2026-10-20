# Real-Time Human Activity Recognition (Webcam)

A single-file Python script that uses your webcam to recognize broad human activities in real time — things like walking, drinking, clapping, brushing teeth, and 396 other everyday actions — using a deep learning model pretrained on the **Kinetics-400** dataset.

This isn't object detection (like YOLO finding a "person" or a "cup"). It's **action recognition** — the model looks at a short *sequence* of frames to understand motion and context over time, which is necessary to tell the difference between, say, "sitting down" and "standing up."

---

## How It Works (High-Level)

1. Your webcam streams frames continuously.
2. The script collects these frames into a rolling buffer.
3. Every time enough new frames have arrived, a 16-frame "clip" is handed to a pretrained 3D convolutional neural network.
4. The model outputs probabilities across 400 possible activity classes.
5. The highest-confidence label is displayed as an overlay on the live video.

Because running the model is slower than the webcam's frame rate (especially on a CPU), **video display and model inference run on two separate threads** so the camera feed stays smooth while predictions update every second or two in the background.

---

## The Model

- **Architecture:** `r2plus1d_18` — an 18-layer R(2+1)D convolutional neural network from `torchvision.models.video`.
- **What R(2+1)D means:** instead of one expensive 3D convolution (which mixes space and time together), this architecture factorizes it into a 2D spatial convolution (per-frame, like a normal image CNN) followed by a 1D temporal convolution (across frames). This is cheaper to compute while still capturing motion, which is part of why it's a reasonable choice for CPU-only machines.
- **Pretrained on:** Kinetics-400, a dataset of ~400 human action categories sourced from YouTube clips (everyday actions, sports, instrument playing, cooking, etc.)
- **Weights source:** downloaded automatically via `torchvision`'s built-in weights API (`R2Plus1D_18_Weights.KINETICS400_V1`) — no manual model file needed. Downloaded once (~120MB), then cached locally so future runs are instant.

---

## Requirements

```bash
pip install torch torchvision opencv-python numpy
```

- **Python:** 3.9–3.12 recommended. Newer versions (e.g. 3.13+) may not have PyTorch wheels available yet.
- **Hardware:** runs on CPU. A GPU (CUDA) isn't required, though it would make inference noticeably faster. This script does not currently auto-detect/use a GPU even if present — see [Possible Improvements](#possible-improvements) below.
- **Internet connection:** required on first run only, to download the pretrained model weights.

---

## Running It

```bash
python activity_recognition.py
```

Press **`q`** (with the video window focused) to quit.

On first run, expect a short delay while the model weights download. On every run, expect the activity label to take a second or two to first appear, since the buffer needs to fill up before the first prediction can be made.

---

## Code Walkthrough

### Configuration constants

```python
CLIP_LEN = 16
STRIDE = 8
CONF_THRESHOLD = 0.15
CAM_INDEX = 0
DISPLAY_WIDTH = 960
```

- **`CLIP_LEN`** — how many consecutive frames make up one "clip" fed to the model. 16 is the conventional clip length for this model family.
- **`STRIDE`** — how many *new* frames must arrive before the next inference runs. Set to half of `CLIP_LEN`, meaning each new clip overlaps 50% with the previous one. This makes label updates feel more frequent and smooth, rather than the label only changing once every full 16-frame window.
- **`CONF_THRESHOLD`** — if the model's top prediction has a confidence below this value, the overlay shows "Uncertain" instead of a potentially wrong/noisy label.
- **`CAM_INDEX`** — which webcam to use. `0` is typically your default/built-in camera; change to `1`, `2`, etc. if you have multiple cameras and the wrong one opens.
- **`DISPLAY_WIDTH`** — resizes the video window for display only. This does *not* affect what the model sees — full-resolution frames are still used for inference internally.

### `SharedState` class

This is the synchronization point between the two threads (display thread and inference thread). It holds:
- `frame_buffer` — a `deque` (double-ended queue) capped at `CLIP_LEN` frames, automatically dropping the oldest frame as new ones are added.
- `frames_since_last_infer` — a counter that triggers a new inference once it reaches `STRIDE`.
- `current_label` / `current_confidence` / `last_infer_ms` — the most recent prediction results, safely read by the display loop.
- A `threading.Lock()` — ensures the two threads never read/write this shared data at the exact same moment, which would otherwise cause corrupted state or crashes.

Key methods:
- **`push_frame()`** — called by the main thread for every captured webcam frame.
- **`maybe_get_clip()`** — called by the inference thread; returns a snapshot of the current buffer *only* if both a full clip is ready AND enough new frames have accumulated since the last inference. Otherwise returns `None` so the inference thread waits.
- **`set_prediction()` / `get_prediction()`** — thread-safe write/read of the latest result.

### Model loading — `load_model()`

```python
weights = R2Plus1D_18_Weights.KINETICS400_V1
model = r2plus1d_18(weights=weights)
model.eval()
```

- `weights.transforms()` retrieves the **exact preprocessing pipeline** the model was trained with — resizing to 128×171, then center-cropping to 112×112, then normalizing with specific mean/std values. Using the official transform (rather than hand-rolling your own resize/normalize logic) ensures the input matches what the model expects, which matters a lot for prediction accuracy.
- `weights.meta["categories"]` returns the actual list of 400 human-readable activity names (e.g. `"drinking beer"`, `"walking the dog"`) in the exact order the model's output neurons correspond to.
- `model.eval()` switches the model to inference mode (disabling training-only behaviors like dropout).

### Preprocessing — `frames_to_clip_tensor()`

Converts a list of 16 raw webcam frames into the exact tensor shape the model expects:

```
List of (H, W, 3) RGB uint8 frames
  → stack into (T, H, W, C)
  → permute into (T, C, H, W)
  → apply official transform → (C, T, H, W), normalized floats
  → add batch dimension → (1, C, T, H, W)
```

This ordering (`T`=time/frames, `C`=color channels, `H`/`W`=height/width) is specific to how `torchvision`'s video transforms are built — getting this wrong is one of the most common bugs when working with video models, since image models normally don't have a time dimension at all.

### Inference worker — `inference_worker()`

Runs in a background thread (`daemon=True`, so it automatically stops when the main program exits):

1. Continuously checks if a new clip is ready via `maybe_get_clip()`.
2. If not ready, sleeps briefly (`time.sleep(0.03)`) to avoid wasting CPU cycles in a tight loop.
3. If ready, runs the clip through the model with `torch.no_grad()` (disables gradient tracking, since we're not training — this saves memory and compute).
4. Applies `softmax` to convert raw model outputs into probabilities, picks the highest one.
5. Stores the resulting label + confidence + how long inference took (`infer_ms`) back into `SharedState`.

### Drawing overlay — `draw_overlay()`

Pure presentation logic — draws a semi-transparent black banner at the top of the frame, then:
- Shows either "Buffering frames...", "Uncertain", or the predicted label + confidence percentage, color-coded (yellow/orange/green).
- Shows live FPS and the last inference time in milliseconds, useful for understanding your machine's actual performance.
- Shows a quit reminder in the corner.

### Main loop — `main()`

1. Loads the model once.
2. Starts the background inference thread.
3. Opens the webcam via `cv2.VideoCapture`.
4. Loops continuously:
   - Grabs a frame, converts BGR→RGB (OpenCV uses BGR by default; the model expects RGB).
   - Pushes it into the shared buffer for the inference thread to use.
   - Calculates a smoothed FPS value.
   - Reads whatever the latest prediction currently is (doesn't wait for a new one each frame).
   - Resizes the frame for display, draws the overlay, shows it in a window.
   - Checks if `q` was pressed to exit.
5. On exit (including errors), the `finally` block guarantees the webcam is released and windows are closed properly, and signals the background thread to stop.

---

## Why a Background Thread Is Necessary

If inference ran directly inside the main display loop, the video feed would freeze for the entire duration of each model forward pass (potentially 0.5–2+ seconds on a CPU) — every single time a new clip is processed. Running inference on a separate thread means:

- The webcam feed keeps capturing and displaying frames smoothly the whole time.
- The displayed label simply reflects whatever the most *recent* completed prediction was, updated whenever the background thread finishes its work — rather than the entire UI stalling to wait for it.

---

## Known Limitations

- **CPU-only as written.** No GPU/CUDA acceleration is used even if available on your machine.
- **Single-person, webcam-framing bias.** Kinetics-400 was trained on varied internet video clips (different angles, often full-body, sometimes multiple people). A single person seated in front of a laptop webcam may not always map cleanly onto the closest Kinetics category.
- **No temporal smoothing across predictions.** Each clip is classified independently — there's no mechanism (like majority voting across recent predictions) to prevent occasional one-off flickers to an unrelated label.
- **Fixed camera index.** If the wrong camera opens, you must manually edit `CAM_INDEX` in the code rather than selecting it at runtime.

---

## Troubleshooting (Real Issues You May Hit on macOS)

These are issues commonly encountered when setting this up on macOS, especially **Intel Macs** or machines with a **very new Python version** (3.13+) pre-installed.

### 1. `No matching distribution found for torch`

**Cause:** PyTorch doesn't ship pre-built wheels for every Python version immediately after it's released. If your system's default Python is very new (e.g. 3.13 or 3.14), `pip install torch` may fail outright because no compatible build exists yet — this is made worse on **Intel Macs**, since PyTorch's newest builds increasingly prioritize Apple Silicon (`arm64`) and Linux/Windows over Intel Mac (`x86_64`).

**Fix:** Use an older, well-supported Python version (3.11 is the safest choice) inside a dedicated virtual environment, instead of your system's default Python.

```bash
# Check what versions you have available
python3.11 --version
python3.12 --version

# If neither exists, install 3.11 via Homebrew
brew install python@3.11
```

### 2. Creating the virtual environment with the right Python version

```bash
python3.11 -m venv har_env
source har_env/bin/activate
python --version   # should print 3.11.x, NOT 3.13/3.14
```

If `python --version` still shows the wrong version after activating, the venv was created with the wrong interpreter — delete the `har_env` folder and recreate it using the correct `python3.11` (or `python3.12`) command explicitly.

### 3. `ModuleNotFoundError: No module named 'cv2'` (or torch, etc.) despite installing it

**Cause:** This almost always means the package was installed into a *different* Python environment than the one actually running the script. Common triggers:
- The virtual environment wasn't activated in the terminal you ran `pip install` in.
- VS Code's selected interpreter (shown in the bottom-right status bar) doesn't match your venv — it may default back to your system Python (e.g. `/usr/local/bin/python3.14`) especially when using the ▶️ **Run Python File** button.

**Fix — confirm your venv is active and matches:**
```bash
source har_env/bin/activate
python --version
pip show opencv-python    # should show a Location inside .../har_env/...
```

**Fix — VS Code interpreter mismatch:**
1. `Cmd+Shift+P` → **"Python: Select Interpreter"**
2. Choose the entry pointing to `./har_env/bin/python`
3. Confirm the bottom-right status bar now shows your venv's Python version, not the system one
4. For guaranteed correctness while debugging, run the script from the integrated terminal directly (`python activity_recognition.py`) rather than relying on the Run button, since the terminal's activated venv always takes priority.

### 4. NumPy 1.x / 2.x compatibility warning

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6...
```

**Cause:** Some prebuilt `torch`/`torchvision` wheels were compiled against NumPy 1.x, but a newer NumPy 2.x got installed alongside them.

**Fix:**
```bash
pip install "numpy<2"
```

### 5. `SSL: CERTIFICATE_VERIFY_FAILED` when downloading model weights

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate>
```

**Cause:** Python on macOS ships its own bundled SSL certificate handling, separate from the system/browser certificates. If the trusted certificate bundle was never installed/linked for your Python install, any HTTPS download made *from Python* (like downloading the model weights) fails — even though normal web browsing works fine.

**Fix (if you installed Python from python.org):**
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```
(Adjust the version number in the path to match what's actually in `/Applications/`.)

**Fix (if you installed Python via Homebrew — no `Install Certificates.command` exists):**
```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -m certifi)
```

⚠️ **This `export` only applies to your current terminal session.** If you close the terminal or open a new one, you must run the `export SSL_CERT_FILE=$(python -m certifi)` line again before running the script, or the SSL error will return.

### Quick full setup checklist (Intel Mac, Homebrew Python)

```bash
brew install python@3.11
python3.11 -m venv har_env
source har_env/bin/activate
python --version                     # confirm 3.11.x
pip install torch torchvision opencv-python numpy
pip install "numpy<2"
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -m certifi)
python activity_recognition.py
```

---

## Possible Improvements

- Auto-detect and use a CUDA GPU if available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), falling back to CPU otherwise.
- Add temporal smoothing (e.g., only update the displayed label if the same prediction appears in 2 of the last 3 inference results).
- Add a "top-3 predictions" display mode for cases where the top-1 guess feels too rigid.
- Let the user select the camera index and clip length via command-line arguments instead of editing the script.

---

## Requirements Summary

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework running the model |
| `torchvision` | Provides the pretrained `r2plus1d_18` model, weights, and official preprocessing transforms |
| `opencv-python` | Webcam capture, frame display, drawing overlays |
| `numpy` | Array handling, used internally by OpenCV and the preprocessing pipeline |

---

## License / Credit

Model architecture and pretrained weights are provided by `torchvision` (PyTorch project), trained on the Kinetics-400 dataset. This script is original glue code wrapping that model for real-time webcam use.
