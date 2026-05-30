import os
import sys
import csv
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.landmarks import HandLandmarkExtractor

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SOURCE_DIR = r"C:\Users\ADMIN\Desktop\files\asl_alphabet\asl_alphabet_train\asl_alphabet_train"
OUTPUT_CSV = "data/landmarks_kaggle.csv"
MAX_PER_CLASS = 500   # Limit to avoid taking forever; 500 × 28 = 14000 samples
SKIP_CLASSES = ['del']   # Skip these folders if they exist
# ──────────────────────────────────────────────────────────────────────────────


def create_csv_header():
    """Create CSV header with label and 21 landmark coordinates."""
    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    return header


def get_classes_to_process(source_dir):
    """Get list of valid class directories."""
    classes = sorted(os.listdir(source_dir))
    return [c for c in classes if c not in SKIP_CLASSES
            and os.path.isdir(os.path.join(source_dir, c))]


def process_class(extractor, class_dir, label, max_per_class):
    """Process all images in a single class directory."""
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = images[:max_per_class]

    detected = 0
    for img_name in tqdm(images, desc=f"Class {label}", unit="img"):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        landmarks, _ = extractor.extract(img)
        yield landmarks, label
        if landmarks is not None:
            detected += 1

    return detected, len(images)


def print_summary(per_class_stats, total_processed, total_detected, output_path):
    """Print conversion summary."""
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    for label, (detected, total) in per_class_stats.items():
        rate = (detected / total * 100) if total > 0 else 0
        print(f"  {label:<10}: {detected:>4}/{total:<4} ({rate:.1f}%)")
    print("-" * 50)
    overall_rate = (total_detected / total_processed * 100) if total_processed > 0 else 0
    print(f"  OVERALL  : {total_detected}/{total_processed} ({overall_rate:.1f}%)")
    print(f"  Saved to : {output_path}")
    print("=" * 50)


def convert():
    if not os.path.exists(SOURCE_DIR):
        print(f"[ERROR] Source not found: {SOURCE_DIR}")
        print("Edit SOURCE_DIR in this file to your Kaggle dataset path.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    extractor = HandLandmarkExtractor(detection_confidence=0.5)

    classes = get_classes_to_process(SOURCE_DIR)
    print(f"[INFO] Found {len(classes)} classes: {classes}")

    # Write CSV with header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(create_csv_header())

        total_processed = 0
        total_detected = 0
        per_class_stats = {}

        for label in classes:
            class_dir = os.path.join(SOURCE_DIR, label)
            detected, total = process_class(extractor, class_dir, label, MAX_PER_CLASS)
            per_class_stats[label] = (detected, total)
            print(f"  {label}: {detected}/{total} hands detected")

    extractor.close()
    print_summary(per_class_stats, total_processed, total_detected, OUTPUT_CSV)


if __name__ == "__main__":
    convert()
