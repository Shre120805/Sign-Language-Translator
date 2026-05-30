
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gpu_config import configure_gpu
configure_gpu(memory_growth=True)

import cv2
import json
import time
import numpy as np
import tensorflow as tf
from collections import deque

from utils.landmarks import HandLandmarkExtractor
from nlp.pipeline import NLPPipeline
from nlp.speech import SpeechEngine

STATIC_MODEL       = "models/landmark_model.h5"
MOTION_MODEL       = "models/motion_model.h5"
LABELS_PATH        = "models/label_encoder.json"
MOTION_LABELS_PATH = "models/motion_label_encoder.json"
MOTION_FRAMES      = 30
MOTION_LABELS      = ['J', 'Z']


def run():
    # Load static model
    if not os.path.exists(STATIC_MODEL):
        print("[ERROR] Run train.py first")
        return
    static_model = tf.keras.models.load_model(STATIC_MODEL)
    with open(LABELS_PATH) as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    # Load motion model
    motion_model = None
    motion_idx_to_label = {}
    if os.path.exists(MOTION_MODEL):
        motion_model = tf.keras.models.load_model(MOTION_MODEL)
        with open(MOTION_LABELS_PATH) as f:
            motion_idx_to_label = {int(k): v for k, v in json.load(f).items()}
        print("[INFO] Motion model loaded (J, Z)")
    else:
        print("[WARN] No motion model — J and Z won't be detected")

    extractor = HandLandmarkExtractor()
    nlp = NLPPipeline()
    speech = SpeechEngine()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # ── Motion state ──────────────────────────────────────────────────────────
    motion_buffer        = []
    motion_recording     = False   # True when actively recording a motion sign
    motion_target_label  = None    # 'J' or 'Z' — which sign we're recording

    last_word = ""
    fps_t = time.time()
    fps = 0
    fc = 0

    print("\n[INFO] J/Z keys = record motion sign | SPACE=word | ENTER=speak | B=back | R=reset | Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        landmarks, results = extractor.extract(frame)
        frame = extractor.draw(frame, results)

        letter = "nothing"
        confidence = 0.0

        if landmarks is not None:
            # ── Static prediction (always runs) ───────────────────────────────
            pred = static_model.predict(landmarks.reshape(1, -1), verbose=0)[0]
            idx = int(np.argmax(pred))
            letter = idx_to_label.get(idx, "?")
            confidence = float(pred[idx])

            # ── Motion recording (only when triggered by J/Z key) ─────────────
            if motion_recording:
                motion_buffer.append(landmarks.copy())

                # Once we have enough frames → predict
                if len(motion_buffer) >= MOTION_FRAMES:
                    motion_recording = False

                    if motion_model is not None:
                        seq = np.array(motion_buffer).reshape(1, MOTION_FRAMES, 63)
                        with tf.device('/CPU:0'):
                            motion_pred = motion_model.predict(seq, verbose=0)[0]
                        motion_idx = int(np.argmax(motion_pred))
                        motion_conf = float(motion_pred[motion_idx])
                        detected = motion_idx_to_label.get(motion_idx, "?")

                        print(f"[MOTION] Detected: {detected} ({motion_conf*100:.1f}%)")

                        # Only accept if confident AND matches what user triggered
                        if motion_conf > 0.75:
                            # Directly add to NLP word buffer
                            nlp.word_buffer.append(detected)
                            nlp.last_input_time = time.time()
                            letter = detected
                            confidence = motion_conf
                        else:
                            print("[MOTION] Low confidence — sign not registered")

                    motion_buffer = []
                    motion_target_label = None

        # ── NLP (only feed static predictions when not in motion mode) ────────
        if not motion_recording:
            nlp.receive(letter, confidence)

        tick = nlp.tick()
        if tick:
            if 'word' in tick:
                last_word = tick['word']
                speech.speak(last_word)
            if 'sentence' in tick:
                speech.speak(tick['sentence'])

        # ── FPS ───────────────────────────────────────────────────────────────
        fc += 1
        if fc % 15 == 0:
            fps = 15 / (time.time() - fps_t + 1e-6)
            fps_t = time.time()

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        disp = letter if letter not in ('nothing', 'space') else "—"
        cv2.putText(frame, f"Letter: {disp}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%  FPS: {fps:.1f}",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

        # Motion recording indicator
        if motion_recording:
            frames_done = len(motion_buffer)
            progress = frames_done / MOTION_FRAMES
            bar_w = int(progress * (w - 40))

            # Red progress bar
            cv2.rectangle(frame, (20, 90), (w-20, 115), (50,50,50), -1)
            cv2.rectangle(frame, (20, 90), (20+bar_w, 115), (0,0,255), -1)
            cv2.putText(frame,
                        f"RECORDING {motion_target_label}... {MOTION_FRAMES-frames_done} frames left — PERFORM NOW!",
                        (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        else:
            # Show J/Z hint when not recording
            cv2.putText(frame, "Press J or Z key to record motion sign",
                        (w//2 - 190, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 100, 255), 1)

        # Bottom bar
        cv2.rectangle(frame, (0, h-130), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, f"Typing: {nlp.get_current_word_raw() or '—'}",
                    (15, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Last word: {last_word}",
                    (15, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"Sentence: {nlp.get_sentence() or '—'}",
                    (15, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame,
                    "J/Z: motion sign | SPACE: word | ENTER: speak | B: back | R: reset | Q: quit",
                    (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        cv2.imshow("ASL Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('j') or key == ord('J'):
            if not motion_recording:
                motion_recording = True
                motion_buffer = []
                motion_target_label = 'J'
                print("[MOTION] Recording J — perform the J sign now!")

        elif key == ord('z') or key == ord('Z'):
            if not motion_recording:
                motion_recording = True
                motion_buffer = []
                motion_target_label = 'Z'
                print("[MOTION] Recording Z — perform the Z sign now!")

        elif key == ord(' '):
            word = nlp.manual_word_break()
            if word:
                last_word = word
                speech.speak(word)

        elif key == 13:  # ENTER
            text = " ".join(nlp.sentence)
            if nlp.word_buffer:
                w2 = nlp.manual_word_break()
                if w2:
                    text = (text + " " + w2).strip()
            if text:
                speech.speak(text)
            nlp.sentence = []

        elif key == ord('b'):
            nlp.delete_last_letter()

        elif key == ord('r'):
            nlp.reset()
            last_word = ""
            motion_recording = False
            motion_buffer = []

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    speech.stop()


if __name__ == "__main__":
    run()