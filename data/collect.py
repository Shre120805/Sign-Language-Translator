"""
DATA COLLECTION — MediaPipe Edition (Static + Motion)
======================================================

Controls (ZERO letter conflicts):
  A-Z keys            : Select label
  Hold LEFT CLICK     : Continuous auto-capture (release to stop)
  SPACE               : Capture ONE sample
  ENTER               : Record motion sequence (J and Z only)
  BACKSPACE           : Delete last 10 samples for current label
  ESC                 : Quit
"""

import os
import sys
import cv2
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.landmarks import HandLandmarkExtractor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STATIC_CSV      = "data/landmarks.csv"
MOTION_CSV      = "data/motion_landmarks.csv"
SAMPLES_TARGET  = 100
HOLD_INTERVAL   = 0.08
MOTION_FRAMES   = 30
MOTION_LABELS   = ['J', 'Z']
# ──────────────────────────────────────────────────────────────────────────────

mouse_held = False

def mouse_callback(event, x, y, flags, param):
    global mouse_held
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_held = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_held = False


def init_static_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ['x','y','z']]
            writer.writerow(header)


def init_motion_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for frame in range(MOTION_FRAMES):
                for i in range(21):
                    for ax in ['x', 'y', 'z']:
                        header.append(f'f{frame}_{ax}{i}')
            writer.writerow(header)


def get_counts(path):
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def append_static(path, label, landmarks):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([label] + landmarks.tolist())


def append_motion(path, label, sequence):
    with open(path, 'a', newline='') as f:
        row = [label]
        for frame_landmarks in sequence:
            row.extend(frame_landmarks.tolist())
        csv.writer(f).writerow(row)


def delete_last_n(path, label, n=10):
    if not os.path.exists(path):
        return 0
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    indices = []
    for i in range(len(data)-1, -1, -1):
        if data[i] and data[i][0] == label:
            indices.append(i)
            if len(indices) >= n:
                break
    for i in sorted(indices, reverse=True):
        del data[i]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    return len(indices)


def collect():
    global mouse_held

    init_static_csv(STATIC_CSV)
    init_motion_csv(MOTION_CSV)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    cv2.namedWindow("Data Collection")
    cv2.setMouseCallback("Data Collection", mouse_callback)

    current_label  = None
    last_hold_time = 0
    flash_until    = 0
    recording      = False
    motion_buffer  = []

    print("\n[INFO] Press A-Z to select label")
    print("[INFO] Hold LEFT CLICK = continuous capture | SPACE = one sample")
    print("[INFO] ENTER = motion record (J/Z) | BACKSPACE = undo | ESC = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks, results = extractor.extract(frame)
        frame = extractor.draw(frame, results)

        now = time.time()
        is_motion_label = current_label in MOTION_LABELS

        # ── Auto-capture while left mouse held ────────────────────────────────
        if mouse_held and current_label and not is_motion_label and landmarks is not None:
            if now - last_hold_time >= HOLD_INTERVAL:
                append_static(STATIC_CSV, current_label, landmarks)
                last_hold_time = now
                flash_until = now + 0.05

        # ── Motion recording ──────────────────────────────────────────────────
        if recording:
            motion_buffer.append(
                landmarks.copy() if landmarks is not None
                else np.zeros(63, dtype=np.float32)
            )
            if len(motion_buffer) >= MOTION_FRAMES:
                append_motion(MOTION_CSV, current_label, motion_buffer)
                count = get_counts(MOTION_CSV).get(current_label, 0)
                print(f"[MOTION] Saved sequence #{count} for {current_label}")
                motion_buffer = []
                recording = False
                flash_until = now + 0.3

        # ── Counts ────────────────────────────────────────────────────────────
        static_counts = get_counts(STATIC_CSV)
        motion_counts = get_counts(MOTION_CSV)
        current_count = 0
        if current_label:
            current_count = (motion_counts if is_motion_label else static_counts).get(current_label, 0)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 90), (25, 25, 25), -1)

        label_color = (0, 100, 255) if is_motion_label else (0, 255, 100)
        mode_text   = "[MOTION]" if is_motion_label else "[STATIC]"
        cv2.putText(frame,
                    f"Label: {current_label or 'NONE — press any letter key'} {mode_text}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        hint = "press ENTER to record" if is_motion_label else "hold LEFT CLICK to capture"
        cv2.putText(frame,
                    f"{hint}   |   Samples: {current_count} / {SAMPLES_TARGET}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Mouse held indicator
        if mouse_held and not is_motion_label:
            cv2.circle(frame, (w - 30, 30), 14, (0, 0, 255), -1)
            cv2.putText(frame, "CAPTURING", (w - 135, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Motion recording bar
        if recording:
            progress = len(motion_buffer) / MOTION_FRAMES
            bar_w = int(progress * 300)
            cv2.rectangle(frame, (w//2-150, h//2-40), (w//2+150, h//2-10), (50,50,50), -1)
            cv2.rectangle(frame, (w//2-150, h//2-40), (w//2-150+bar_w, h//2-10), (0,0,255), -1)
            cv2.putText(frame, f"RECORDING — {MOTION_FRAMES - len(motion_buffer)} frames left",
                        (w//2-160, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, "PERFORM THE SIGN NOW!",
                        (w//2-120, h//2+55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # Hand status
        status = "Hand DETECTED" if landmarks is not None else "No hand"
        color  = (0, 255, 0) if landmarks is not None else (0, 0, 255)
        cv2.putText(frame, status, (w - 220, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Capture flash
        if now < flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Progress bar
        if current_label:
            bar_y = h - 55
            fill = min(int((current_count / SAMPLES_TARGET) * (w - 40)), w - 40)
            bar_c = (0, 255, 0) if current_count >= SAMPLES_TARGET else (0, 200, 255)
            cv2.rectangle(frame, (20, bar_y), (w-20, bar_y+18), (50,50,50), -1)
            cv2.rectangle(frame, (20, bar_y), (20+fill, bar_y+18), bar_c, -1)

        # Controls reminder
        cv2.putText(frame,
            "A-Z: label | LEFT HOLD: capture | SPACE: 1 sample | ENTER: motion | BKSP: undo | ESC: quit",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        cv2.imshow("Data Collection", frame)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 13:  # ENTER — motion record
            if is_motion_label and not recording:
                recording = True
                motion_buffer = []
                print(f"[MOTION] Recording {current_label} — perform the sign!")
            elif not is_motion_label:
                print(f"[WARN] ENTER only for J and Z. Current label: {current_label}")
            else:
                print("[WARN] Already recording...")

        elif key == ord(' '):  # SPACE — one sample
            if current_label and landmarks is not None and not is_motion_label:
                append_static(STATIC_CSV, current_label, landmarks)
                flash_until = now + 0.2
                print(f"[SAVED] {current_label} #{static_counts.get(current_label,0)+1}")
            elif is_motion_label:
                print(f"[WARN] {current_label} is motion — press ENTER to record")
            elif landmarks is None:
                print("[WARN] No hand detected")
            else:
                print("[WARN] Select a label first")

        elif key == 8:  # BACKSPACE — undo
            if current_label:
                path = MOTION_CSV if is_motion_label else STATIC_CSV
                deleted = delete_last_n(path, current_label, n=10)
                print(f"[UNDO] Deleted {deleted} samples for {current_label}")

        elif key != 255:  # Letter selection
            char = chr(key).upper()
            if char.isalpha():
                current_label = char
                recording = False
                motion_buffer = []
                mouse_held = False
                mode = "MOTION — press ENTER to record" if char in MOTION_LABELS else "STATIC — hold left click"
                print(f"[LABEL] {char} — {mode}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print("Static samples:")
    for label, count in sorted(get_counts(STATIC_CSV).items()):
        print(f"  {label:<10}: {count}")
    print("\nMotion sequences:")
    for label, count in sorted(get_counts(MOTION_CSV).items()):
        print(f"  {label:<10}: {count}")
    print("="*50)


if __name__ == "__main__":
    collect()