import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gpu_config import configure_gpu, verify_gpu
configure_gpu(memory_growth=True)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

from models.classifier import build_classifier

# CONFIG
STATIC_DATA_FILES  = ["data/landmarks.csv", "data/landmarks_kaggle.csv"]
MOTION_DATA_FILE   = "data/motion_landmarks.csv"
STATIC_MODEL_SAVE  = "models/landmark_model.h5"
MOTION_MODEL_SAVE  = "models/motion_model.h5"
LABEL_ENCODER_SAVE = "models/label_encoder.json"
MOTION_LABELS      = ['J', 'Z']
MOTION_FRAMES      = 30
EPOCHS             = 50
BATCH_SIZE         = 64
TEST_SIZE          = 0.2


# ── STATIC TRAINING ───────────────────────────────────────────────────────────

def load_static_data():
    dfs = []
    for path in STATIC_DATA_FILES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[DATA] Loaded {len(df)} samples from {path}")
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No CSVs found. Run data/collect.py or data/convert_dataset.py")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[DATA] Total: {len(combined)} samples")
    print("\n[DATA] Class distribution:")
    for label, count in combined['label'].value_counts().sort_index().items():
        print(f"  {label:<10}: {count}")
    return combined


def train_static():
    print("\n" + "="*55)
    print("  TRAINING STATIC CLASSIFIER (GPU)")
    print("="*55)

    df = load_static_data()
    X = df.drop('label', axis=1).to_numpy().astype(np.float32)
    y_str = df['label'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    num_classes = len(encoder.classes_)
    y_cat = to_categorical(y, num_classes)

    X_train, x_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {len(X_train)} | Val: {len(x_val)}")
    print(f"[SPLIT] Classes ({num_classes}): {list(encoder.classes_)}")

    model = build_classifier(num_classes=num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint("models/best_static_model.h5",
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )

    model.save(STATIC_MODEL_SAVE)
    label_map = dict(enumerate(encoder.classes_))
    with open(LABEL_ENCODER_SAVE, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"\n[SAVED] {STATIC_MODEL_SAVE}")
    print(f"[SAVED] {LABEL_ENCODER_SAVE}")

    plot_history(history, title="Static Classifier", save="training_curves.png")
    evaluate_model(model, x_val, y_val, encoder, save_prefix="static")
    return model, encoder


# ── MOTION TRAINING ───────────────────────────────────────────────────────────

def build_motion_model(num_classes, frames=MOTION_FRAMES, features=63):
    """
    LSTM model for J and Z motion sequences.
    Uses Conv1D instead of LSTM (works with DirectML, no cuDNN needed).
    """
    inputs = tf.keras.Input(shape=(frames, features))
    
    # Conv1D layers to extract temporal features (works on DirectML)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling to reduce sequence dimension
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="MotionConv1D")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_motion():
    if not os.path.exists(MOTION_DATA_FILE):
        print("\n[MOTION] No motion data found — skipping")
        return None, None

    df = pd.read_csv(MOTION_DATA_FILE)
    if len(df) < 10:
        print(f"\n[MOTION] Only {len(df)} sequences — skipping (need >= 10)")
        return None, None

    print("\n" + "="*55)
    print("  TRAINING MOTION CLASSIFIER (CPU — LSTM)")
    print("="*55)
    print(f"[DATA] Motion sequences: {len(df)}")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count} sequences")

    X = df.drop('label', axis=1).to_numpy().astype(np.float32)
    X = X.reshape(-1, MOTION_FRAMES, 63)
    y_str = df['label'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    num_classes = len(encoder.classes_)
    y_cat = to_categorical(y, num_classes)

    X_train, x_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    # ── FORCE CPU — DirectML does not support CudnnRNN used by LSTM ──────────
    print("[INFO] Forcing CPU for LSTM (DirectML does not support CudnnRNN)")
    with tf.device('/CPU:0'):
        model = build_motion_model(num_classes=num_classes)
        model.summary()

        callbacks = [
            ModelCheckpoint("models/best_motion_model.h5",
                            monitor='val_accuracy', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10,
                          restore_best_weights=True, verbose=1),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=min(BATCH_SIZE, len(X_train)),
            callbacks=callbacks,
            verbose=1
        )

    model.save(MOTION_MODEL_SAVE)
    motion_label_map = dict(enumerate(encoder.classes_))
    with open("models/motion_label_encoder.json", 'w') as f:
        json.dump(motion_label_map, f, indent=2)
    print(f"\n[SAVED] {MOTION_MODEL_SAVE}")

    plot_history(history, title="Motion Classifier (LSTM)", save="training_curves_motion.png")
    return model, encoder


# ── PLOTTING & EVALUATION ─────────────────────────────────────────────────────

def plot_history(history, title="Training", save="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax1.plot(history.history['accuracy'],     label='Train', color='steelblue', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val',   color='darkorange', linewidth=2)
    ax1.axhline(0.85, color='red', linestyle=':', label='85% target')
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim([0, 1])
    ax2.plot(history.history['loss'],     label='Train', color='steelblue', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val',   color='darkorange', linewidth=2)
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    plt.show()
    print(f"[SAVED] {save}")


def evaluate_model(model, x_val, y_val, encoder, save_prefix="static"):
    y_pred_probs = model.predict(x_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)
    target_names = list(encoder.classes_)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*55}")
    print(f"  EVALUATION — {save_prefix.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc*100:.2f}%  {'✓' if acc>=0.85 else '✗ Below 85%'}")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"{'='*55}\n")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Donut summary
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Performance — {save_prefix.title()}", fontsize=14, fontweight='bold')
    for ax, (name, value, color) in zip(axes, [
        ("Accuracy",  acc,       "#2ECC71"),
        ("Precision", precision, "#3498DB"),
        ("Recall",    recall,    "#9B59B6"),
        ("F1 Score",  f1,        "#E67E22"),
    ]):
        ax.pie([value, 1-value], radius=1, colors=[color, "#EEEEEE"],
               wedgeprops={"width": 0.35, "edgecolor": 'white'})
        ax.text(0, 0, f"{value*100:.1f}%", ha='center', va='center',
                fontsize=20, fontweight='bold', color=color)
        ax.set_title(name, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"metrics_summary_{save_prefix}.png", dpi=120)
    plt.show()

    # Per-class bar
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(target_names)), zero_division=0
    )
    x = np.arange(len(target_names))
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x-0.25, prec_pc, 0.25, label='Precision', color='#3498DB', alpha=0.85)
    ax.bar(x,      rec_pc,  0.25, label='Recall',    color='#9B59B6', alpha=0.85)
    ax.bar(x+0.25, f1_pc,   0.25, label='F1',        color='#E67E22', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(target_names, fontsize=9)
    ax.set_ylim([0, 1.1]); ax.axhline(0.85, color='red', linestyle='--', alpha=0.5)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title(f"Per-Class Metrics — {save_prefix.title()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"per_class_metrics_{save_prefix}.png", dpi=120)
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                linewidths=0.3)
    plt.title(f"Confusion Matrix — {save_prefix.title()}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{save_prefix}.png", dpi=100)
    plt.show()
    return acc


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"[GPU] {'Training on: ' + gpus[0].name if gpus else 'WARNING: CPU only'}")

    train_static()
    train_motion()

    print("\n[DONE] Training complete.")
    print("  Run: streamlit run app.py")