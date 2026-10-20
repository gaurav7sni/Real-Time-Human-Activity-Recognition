"""
Real-Time Human Activity Recognition (Webcam)
-----------------------------------------------
Single-file script: captures your webcam feed and recognizes broad human
activities (Kinetics-400 categories: walking, drinking, brushing teeth,
clapping, etc.) in real time.

Model: torchvision r2plus1d_18, pretrained on Kinetics-400 (auto-downloaded
on first run, then cached locally by torchvision).

Architecture notes (why it's built this way):
- CPU inference on a 16-frame clip is too slow to run every single frame,
  so video display and model inference run on SEPARATE threads:
    * Main thread:        grabs webcam frames, draws overlay, shows window
    * Background thread:  takes a 16-frame snapshot every 8 new frames
                           (50% overlap) and runs the model on it
  This keeps the webcam feed smooth while the activity label updates
  every ~1-2 seconds (CPU-dependent), which is fine since we're
  prioritizing accuracy over instant labeling.

Usage:
    pip install torch torchvision opencv-python numpy
    python activity_recognition.py
    Press 'q' to quit.
"""

import time
import threading
from collections import deque

import cv2
import numpy as np
import torch

from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CLIP_LEN = 16          # number of frames per inference clip (model standard)
STRIDE = 8             # run a new inference every STRIDE new frames (50% overlap)
CONF_THRESHOLD = 0.15  # below this confidence, show "Uncertain" instead of the label
CAM_INDEX = 0          # change if you have multiple webcams
DISPLAY_WIDTH = 960    # resize window for display only (inference uses raw frames)


# --------------------------------------------------------------------------- #
# Shared state between the display thread and the inference thread
# --------------------------------------------------------------------------- #
class SharedState:
    """Thread-safe holder for the rolling frame buffer and the latest prediction."""

    def __init__(self, clip_len: int):
        self.lock = threading.Lock()
        self.frame_buffer = deque(maxlen=clip_len)
        self.frames_since_last_infer = 0
        self.current_label = "Buffering..."
        self.current_confidence = 0.0
        self.last_infer_ms = 0.0
        self.running = True

    def push_frame(self, frame_rgb: np.ndarray):
        """Called by the main thread for every captured frame."""
        with self.lock:
            self.frame_buffer.append(frame_rgb)
            self.frames_since_last_infer += 1

    def maybe_get_clip(self, clip_len: int, stride: int):
        """
        Called by the inference thread. Returns a snapshot list of frames
        if a full clip is ready AND enough new frames have arrived since
        the last inference, otherwise returns None.
        """
        with self.lock:
            if len(self.frame_buffer) < clip_len:
                return None
            if self.frames_since_last_infer < stride:
                return None
            # snapshot copy so inference doesn't race with the buffer mutating
            clip = list(self.frame_buffer)
            self.frames_since_last_infer = 0
            return clip

    def set_prediction(self, label: str, confidence: float, infer_ms: float):
        with self.lock:
            self.current_label = label
            self.current_confidence = confidence
            self.last_infer_ms = infer_ms

    def get_prediction(self):
        with self.lock:
            return self.current_label, self.current_confidence, self.last_infer_ms

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model():
    print("Loading r2plus1d_18 (Kinetics-400 pretrained weights)...")
    print("First run will download ~120MB of weights, then they're cached.")
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))  # use available CPU threads
    transform = weights.transforms()
    categories = weights.meta["categories"]
    print(f"Model loaded. {len(categories)} possible activity labels.")
    return model, transform, categories


# --------------------------------------------------------------------------- #
# Preprocessing: list of HxWx3 RGB uint8 frames -> model input tensor
# --------------------------------------------------------------------------- #
def frames_to_clip_tensor(frames_rgb, transform):
    """
    frames_rgb: list of CLIP_LEN numpy arrays, each (H, W, 3) RGB uint8
    Returns: (1, C, T, H, W) float tensor, batch size 1, ready for the model.
    """
    # Stack to (T, H, W, C) uint8
    clip = np.stack(frames_rgb, axis=0)
    clip_t = torch.from_numpy(clip)              # T H W C, uint8
    clip_t = clip_t.permute(0, 3, 1, 2)           # T C H W, uint8  (what transform expects)
    clip_t = transform(clip_t)                    # -> C T H W, normalized float
    clip_t = clip_t.unsqueeze(0)                   # -> N C T H W, batch of 1
    return clip_t


# --------------------------------------------------------------------------- #
# Inference worker thread
# --------------------------------------------------------------------------- #
def inference_worker(state: SharedState, model, transform, categories):
    while state.is_running():
        clip = state.maybe_get_clip(CLIP_LEN, STRIDE)
        if clip is None:
            time.sleep(0.03)  # avoid busy-spinning while waiting for frames
            continue

        start = time.time()
        with torch.no_grad():
            clip_tensor = frames_to_clip_tensor(clip, transform)
            logits = model(clip_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        infer_ms = (time.time() - start) * 1000.0

        label = categories[idx.item()]
        confidence = conf.item()
        state.set_prediction(label, confidence, infer_ms)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_overlay(frame_bgr, label, confidence, fps, infer_ms, buffering):
    h, w = frame_bgr.shape[:2]

    # Semi-transparent banner at the top for readability
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), thickness=-1)
    frame_bgr = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    if buffering:
        display_text = "Buffering frames..."
        color = (0, 200, 255)
    elif confidence < CONF_THRESHOLD:
        display_text = "Uncertain"
        color = (0, 165, 255)
    else:
        display_text = f"{label}  ({confidence * 100:.0f}%)"
        color = (60, 220, 60)

    cv2.putText(frame_bgr, display_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)

    status_text = f"FPS: {fps:.1f}   |   Last inference: {infer_ms:.0f} ms"
    cv2.putText(frame_bgr, status_text, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame_bgr, "Press 'q' to quit", (w - 175, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return frame_bgr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    model, transform, categories = load_model()

    state = SharedState(clip_len=CLIP_LEN)
    worker = threading.Thread(
        target=inference_worker, args=(state, model, transform, categories), daemon=True
    )
    worker.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam at index {CAM_INDEX}.")
        return

    print("Webcam opened. Starting live activity recognition. Press 'q' to quit.")

    fps_smooth = 0.0
    prev_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("WARNING: failed to read frame from webcam.")
                break

            # Feed the RAW frame (RGB) into the rolling buffer for inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            state.push_frame(frame_rgb)

            # FPS calculation (smoothed)
            now = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1
            prev_time = now

            label, confidence, infer_ms = state.get_prediction()
            buffering = label == "Buffering..."

            # Resize only for display, keep aspect ratio
            h, w = frame_bgr.shape[:2]
            scale = DISPLAY_WIDTH / w
            display_frame = cv2.resize(frame_bgr, (DISPLAY_WIDTH, int(h * scale)))

            display_frame = draw_overlay(
                display_frame, label, confidence, fps_smooth, infer_ms, buffering
            )

            cv2.imshow("Real-Time Human Activity Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        state.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()