# 🤟 ASL Sign Language Translator

Real-time American Sign Language to text + speech, using MediaPipe + Neural Network + NLP.

---

## ⚡ Quick Setup (Run in this order)

### 1. Install Python 3.10 and create virtual environment
```bash
py -3.10 -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get training data — choose ONE of these:

**Option A: Convert existing Kaggle dataset (recommended for speed)**
```bash
python data/convert_dataset.py
```
This converts your existing Kaggle images into landmark CSV. Takes 10-20 min.

**Option B: Collect your own data**
```bash
python data/collect.py
```
Press A-Z to select label, SPACE or hold H to capture. Aim for 100+ samples per letter.

**Option C: Use BOTH (best results)**
Run both above. Training script auto-merges them.

### 4. Train model (5-15 minutes)
```bash
python train.py
```
Outputs: `models/landmark_model.h5`, `models/label_encoder.json`, training graphs

### 5. Run the app

**Streamlit UI (recommended for presentation):**
```bash
streamlit run app.py
```
Opens in browser at http://localhost:8501

**OpenCV fallback:**
```bash
python realtime.py
```

---

## 📁 Project Structure

```
sign_v2/
├── app.py                    ← Streamlit web UI (main interface)
├── realtime.py               ← OpenCV fallback runtime
├── train.py                  ← Model training + evaluation
├── requirements.txt
│
├── utils/
│   └── landmarks.py          ← MediaPipe extractor
│
├── data/
│   ├── collect.py            ← Custom data collection
│   ├── convert_dataset.py    ← Convert Kaggle images → landmarks
│   ├── landmarks.csv         ← (generated) custom data
│   └── landmarks_kaggle.csv  ← (generated) Kaggle data
│
├── models/
│   ├── classifier.py         ← MLP architecture
│   ├── landmark_model.h5     ← (generated) trained model
│   └── label_encoder.json    ← (generated) label mapping
│
├── nlp/
│   ├── pipeline.py           ← Letter→word→sentence
│   └── speech.py             ← Offline TTS
│
└── (generated metric PNGs)
```

---

## 🎓 For Viva — Talking Points

### Architecture explanation:
1. **MediaPipe** detects 21 hand keypoints (x, y, z) per frame
2. **Normalization**: Subtract wrist position + scale by hand size — makes it position/size invariant
3. **MLP classifier** with 3 hidden layers (256 → 128 → 64) classifies the 63-dim landmark vector
4. **NLP pipeline** smooths predictions, forms words, builds sentences
5. **Speech engine** outputs final text via offline TTS

### Why this approach over raw image CNN:
- **Background invariant** — only landmarks, no pixel data
- **Lighting invariant** — landmarks don't change with brightness
- **Skin-tone fair** — works for everyone equally
- **30+ FPS real-time** vs 15 FPS for CNN
- **Trains in 10 minutes** vs hours

### NLP Pipeline (the heart of the project):
- **Confidence gate** — rejects predictions below threshold
- **Stability voting** — majority vote over rolling window prevents flicker
- **Cooldown** — same letter can't repeat within N seconds
- **Pause detection** — silence triggers word boundary
- **Dictionary lookup + spell correction** — using Python's difflib (edit distance)
- **Why no transformers?** — Latency, RAM, overkill. Difflib is <1ms per call, no model needed.

---

## 🚀 Deployment Path

This is structured for easy production deployment:
- **Web**: Streamlit Cloud, Render, Hugging Face Spaces
- **Mobile**: Convert MLP to TFLite, ship MediaPipe via mediapipe-android
- **Desktop**: Bundle with PyInstaller
- **API**: Wrap predict() in FastAPI endpoint

The MLP model is <1MB and runs at 30+ FPS — production-ready.
