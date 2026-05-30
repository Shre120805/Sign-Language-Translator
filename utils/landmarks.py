"""
MEDIAPIPE HAND LANDMARK EXTRACTOR
==================================
Extracts 21 hand keypoints (x, y, z) from each video frame.
Output: 63-dimensional feature vector (21 points × 3 coordinates).

The 21 landmarks (MediaPipe Hands):
  0: Wrist
  1-4: Thumb (CMC, MCP, IP, TIP)
  5-8: Index (MCP, PIP, DIP, TIP)
  9-12: Middle finger
  13-16: Ring finger
  17-20: Pinky

Why landmarks instead of raw pixels?
  - Background-invariant
  - Lighting-invariant
  - Skin-tone invariant
  - Tiny feature vector (63 floats vs 224×224×3 = 150K pixels)
"""

import cv2
import numpy as np
import mediapipe as mp


class HandLandmarkExtractor:
    """
    Wrapper around MediaPipe Hands.
    """

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )

    def extract(self, frame_bgr):
        """
        Extract hand landmarks from a single BGR frame.

        Returns:
            landmarks: np.ndarray of shape (63,) — flattened (x,y,z) for 21 points
                       OR None if no hand detected
            results: raw MediaPipe results (for drawing)
        """
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, results

        # Take first detected hand
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # NORMALIZE: relative to wrist (translation invariance)
        # and scale by hand size (scale invariance)
        landmarks = self._normalize(landmarks)

        return landmarks, results

    def _normalize(self, landmarks):
        """
        Normalize landmarks so they're translation- and scale-invariant.
        - Subtract wrist position (point 0)
        - Divide by max distance from wrist
        """
        coords = landmarks.reshape(21, 3)
        wrist = coords[0].copy()
        coords -= wrist  # Translation invariance

        # Scale invariance: normalize by max distance from wrist
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 0:
            coords /= max_dist

        return coords.flatten()

    def draw(self, frame_bgr, results):
        """Draw landmarks on frame for visualization."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
        return frame_bgr

    def close(self):
        self.hands.close()


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Webcam landmark extraction")
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks, results = extractor.extract(frame)
        frame = extractor.draw(frame, results)

        if landmarks is not None:
            cv2.putText(frame, f"Hand detected | {len(landmarks)} features",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("MediaPipe Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
