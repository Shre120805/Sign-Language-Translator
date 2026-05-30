import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gpu_config import configure_gpu
configure_gpu(memory_growth=True)

import json
import time
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

from utils.landmarks import HandLandmarkExtractor
from nlp.pipeline import NLPPipeline
from nlp.speech import SpeechEngine

MOTION_FRAMES = 30

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #2ECC71, #3498DB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .letter-display {
        font-size: 5rem; font-weight: 800; color: #2ECC71;
        text-align: center; line-height: 1; margin: 0;
    }
    .confidence-text { text-align: center; font-size: 1rem; color: #888; }
    .word-buffer {
        background: #2C2C2C; padding: 0.8rem 1.2rem; border-radius: 8px;
        font-size: 1.5rem; font-family: 'Courier New', monospace;
        color: #F39C12; border-left: 4px solid #F39C12; min-height: 50px;
    }
    .sentence-box {
        background: #2C2C2C; padding: 1rem 1.5rem; border-radius: 8px;
        font-size: 1.2rem; color: #FFF;
        border-left: 4px solid #3498DB; min-height: 60px;
    }
</style>
""", unsafe_allow_html=True)


# ─── CACHED LOADERS ───────────────────────────────────────────────────────────

@st.cache_resource
def load_static_model():
    if not os.path.exists("models/landmark_model.h5"):
        return None, None
    model = tf.keras.models.load_model("models/landmark_model.h5")
    with open("models/label_encoder.json") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}
    return model, idx_to_label


@st.cache_resource
def load_motion_model():
    if not os.path.exists("models/motion_model.h5"):
        return None, {}
    model = tf.keras.models.load_model("models/motion_model.h5")
    with open("models/motion_label_encoder.json") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}
    return model, idx_to_label


@st.cache_resource
def get_extractor():
    return HandLandmarkExtractor()


# ─── SESSION STATE ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        'nlp': NLPPipeline(),
        'speech': SpeechEngine(),
        'running': False,
        'last_word': "",
        'motion_recording': False,
        'motion_target': None,
        'motion_buffer': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    page = st.radio("Navigate",
                    ["🎥 Translator", "📊 Metrics", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🎚️ NLP Tuning")

    confidence_threshold = st.slider("Confidence threshold", 0.4, 0.95, 0.65, 0.05,
        help="Min confidence to register a letter")
    cooldown = st.slider("Letter cooldown (sec)", 0.3, 3.0, 1.2, 0.1,
        help="Wait before same letter accepted again")
    word_pause = st.slider("Word pause (sec)", 1.0, 5.0, 2.0, 0.5,
        help="Silence duration to form a word")

    st.session_state.nlp.min_confidence = confidence_threshold
    st.session_state.nlp.letter_cooldown = cooldown
    st.session_state.nlp.word_pause_threshold = word_pause

    st.markdown("---")
    st.markdown("### 📈 Session Stats")
    stats = st.session_state.nlp.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Letters", stats['accepted'])
    c2.metric("Words", stats['words'])


# ─── PAGE: TRANSLATOR ─────────────────────────────────────────────────────────

def render_translator():
    st.markdown('<p class="main-header">🤟 ASL Sign Translator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time American Sign Language → text → speech</p>',
                unsafe_allow_html=True)

    static_model, idx_to_label = load_static_model()
    motion_model, motion_idx_to_label = load_motion_model()

    if static_model is None:
        st.error("⚠️ Model not found. Run `python train.py` first.")
        return

    # ── Row 1: Main controls ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("▶️ Start", type="primary"):
        st.session_state.running = True
    if c2.button("⏸️ Stop"):
        st.session_state.running = False
    if c3.button("🔊 Speak"):
        text = st.session_state.nlp.get_sentence().replace("[","").replace("]","")
        if text:
            st.session_state.speech.speak(text)
    if c4.button("⏎ Word"):
        word = st.session_state.nlp.manual_word_break()
        if word:
            st.session_state.last_word = word
    if c5.button("🔄 Reset"):
        st.session_state.nlp.reset()
        st.session_state.last_word = ""
        st.session_state.motion_recording = False
        st.session_state.motion_buffer = []

    # ── Row 2: Motion sign buttons ────────────────────────────────────────────
    st.markdown("**Motion Signs — click then immediately perform the sign:**")
    cm1, cm2, cm3 = st.columns([1, 1, 4])

    if cm1.button("✋ Record J"):
        st.session_state.motion_recording = True
        st.session_state.motion_target = 'J'
        st.session_state.motion_buffer = []

    if cm2.button("✋ Record Z"):
        st.session_state.motion_recording = True
        st.session_state.motion_target = 'Z'
        st.session_state.motion_buffer = []

    motion_status = cm3.empty()

    st.markdown("---")

    # ── Layout ────────────────────────────────────────────────────────────────
    col_video, col_info = st.columns([2, 1])
    with col_video:
        st.markdown("##### 📹 Webcam Feed")
        video_placeholder = st.empty()
    with col_info:
        st.markdown("##### 🎯 Prediction")
        letter_placeholder = st.empty()
        conf_placeholder = st.empty()
        st.markdown("##### ✏️ Current word")
        word_placeholder = st.empty()
        st.markdown("##### 📝 Sentence")
        sentence_placeholder = st.empty()

    # ── Main loop ─────────────────────────────────────────────────────────────
    if st.session_state.running:
        extractor = get_extractor()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam")
            return

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                landmarks, results = extractor.extract(frame)
                frame = extractor.draw(frame, results)

                letter = "nothing"
                confidence = 0.0

                if landmarks is not None:
                    # Static prediction
                    pred = static_model.predict(landmarks.reshape(1, -1), verbose=0)[0]
                    idx = int(np.argmax(pred))
                    letter = idx_to_label.get(idx, "?")
                    confidence = float(pred[idx])

                    # Motion recording — collect frames after button pressed
                    if st.session_state.motion_recording:
                        st.session_state.motion_buffer.append(landmarks.copy())
                        frames_done = len(st.session_state.motion_buffer)
                        remaining = MOTION_FRAMES - frames_done
                        motion_status.warning(
                            f"🔴 Recording **{st.session_state.motion_target}**... "
                            f"{frames_done}/{MOTION_FRAMES} frames — **PERFORM NOW!** ({remaining} left)"
                        )

                        # Once we have enough frames — predict
                        if frames_done >= MOTION_FRAMES:
                            st.session_state.motion_recording = False

                            if motion_model is not None:
                                seq = np.array(st.session_state.motion_buffer).reshape(1, MOTION_FRAMES, 63)
                                with tf.device('/CPU:0'):
                                    mp = motion_model.predict(seq, verbose=0)[0]
                                midx = int(np.argmax(mp))
                                mconf = float(mp[midx])
                                detected = motion_idx_to_label.get(midx, "?")

                                if mconf > 0.75:
                                    st.session_state.nlp.word_buffer.append(detected)
                                    st.session_state.nlp.last_input_time = time.time()
                                    motion_status.success(
                                        f"✅ {detected} detected ({mconf*100:.1f}%) — added to word"
                                    )
                                else:
                                    motion_status.error(
                                        f"❌ Low confidence ({mconf*100:.1f}%) — click button and try again"
                                    )
                            st.session_state.motion_buffer = []

                # Feed static letters to NLP only when not recording motion
                if not st.session_state.motion_recording:
                    st.session_state.nlp.receive(letter, confidence)

                tick = st.session_state.nlp.tick()
                if tick:
                    if 'word' in tick:
                        st.session_state.last_word = tick['word']
                    if 'sentence' in tick:
                        st.session_state.speech.speak(tick['sentence'])

                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                disp = letter if letter not in ('nothing', 'space') else "—"
                letter_placeholder.markdown(
                    f'<p class="letter-display">{disp}</p>', unsafe_allow_html=True)

                if confidence >= 0.75:
                    color = "#2ECC71"
                elif confidence >= 0.5:
                    color = "#F39C12"
                else:
                    color = "#E74C3C"
                conf_placeholder.markdown(
                    f'<p class="confidence-text" style="color:{color}">Confidence: {confidence*100:.1f}%</p>',
                    unsafe_allow_html=True)

                word_placeholder.markdown(
                    f'<div class="word-buffer">{st.session_state.nlp.get_current_word_raw() or "—"}</div>',
                    unsafe_allow_html=True)

                sentence_placeholder.markdown(
                    f'<div class="sentence-box">{st.session_state.nlp.get_sentence() or "—"}</div>',
                    unsafe_allow_html=True)

                time.sleep(0.03)
        finally:
            cap.release()
    else:
        video_placeholder.info("👆 Press **Start** to begin translation")


# ─── PAGE: METRICS ────────────────────────────────────────────────────────────

def render_metrics():
    st.markdown('<p class="main-header">📊 Model Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Performance from latest training run</p>', unsafe_allow_html=True)

    for title, path in [
        ("Performance Summary", "metrics_summary_static.png"),
        ("Training Curves",     "training_curves.png"),
        ("Per-Class Metrics",   "per_class_metrics_static.png"),
        ("Confusion Matrix",    "confusion_matrix_static.png"),
    ]:
        if os.path.exists(path):
            st.markdown(f"##### {title}")
            st.image(path, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"📁 {title} not found — run `python train.py` first.")


# ─── PAGE: ABOUT ──────────────────────────────────────────────────────────────

def render_about():
    st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Project Overview
    Real-time ASL translator — hand signs → text → speech.

    ### 🔧 Technology Stack
    - **Hand Tracking**: MediaPipe Hands (21 landmark keypoints)
    - **Static Signs**: MLP neural network (3 hidden layers, 99.7% accuracy)
    - **Motion Signs (J, Z)**: Conv1D classifier on 30-frame sequences
    - **NLP Pipeline**: Stability voting + dictionary + spell correction
    - **Speech**: pyttsx3 (offline TTS)

    ### 🤟 How to use J and Z
    1. Click **Record J** or **Record Z** button
    2. Immediately perform the sign motion
    3. Red bar fills up as frames are collected (30 frames ≈ 1 second)
    4. Letter is automatically added to your word

    ### ✨ Why MediaPipe + MLP?
    | | Image CNN | Landmark MLP |
    |---|---|---|
    | Background | ❌ Sensitive | ✅ Ignored |
    | Lighting | ❌ Sensitive | ✅ Robust |
    | Training time | 1-2 hours | 5-10 min |
    | Accuracy | ~85% | **99.7%** |

    ### 🚀 Future Enhancements
    - Full sentence language model
    - Two-hand signs
    - Mobile deployment via TFLite
    """)


# ─── ROUTING ──────────────────────────────────────────────────────────────────

if page == "🎥 Translator":
    render_translator()
elif page == "📊 Metrics":
    render_metrics()
elif page == "ℹ️ About":
    render_about()

st.markdown("---")
st.markdown('<p style="text-align:center;color:#666;font-size:0.85rem;">ASL Translator · MediaPipe + TensorFlow + Streamlit</p>',
            unsafe_allow_html=True)