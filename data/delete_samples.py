import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATIC_CSV = "data/landmarks.csv"
MOTION_CSV = "data/motion_landmarks.csv"


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


def delete_last_n(path, label, n):
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return 0
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]

    # Find last N rows matching label
    indices = []
    for i in range(len(data)-1, -1, -1):
        if data[i] and data[i][0] == label:
            indices.append(i)
            if len(indices) >= n:
                break

    if not indices:
        print(f"[WARN] No samples found for label '{label}'")
        return 0

    for i in sorted(indices, reverse=True):
        del data[i]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    return len(indices)


def delete_all(path, label):
    """Delete ALL samples for a label."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return 0
    with open(path, 'r') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]

    before = len(data)
    data = [row for row in data if not (row and row[0] == label)]
    deleted = before - len(data)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    return deleted


if __name__ == "__main__":
    # ── Show current counts ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  CURRENT SAMPLE COUNTS")
    print("="*50)
    static = get_counts(STATIC_CSV)
    motion = get_counts(MOTION_CSV)

    print("Static:")
    for label, count in sorted(static.items()):
        print(f"  {label:<10}: {count}")
    print("Motion:")
    for label, count in sorted(motion.items()):
        print(f"  {label:<10}: {count}")
    print("="*50)

    # ── Get input ─────────────────────────────────────────────────────────────
    print("\nWhich label do you want to delete from?")
    label = input("Label (e.g. S): ").strip().upper()

    print("\nHow many to delete?")
    print("  Enter a number (e.g. 150) to delete last N samples")
    print("  Enter 'all' to delete ALL samples for this label")
    amount = input("Amount: ").strip().lower()

    is_motion = label in ['J', 'Z']
    path = MOTION_CSV if is_motion else STATIC_CSV

    print(f"\n[INFO] Targeting: {path}")
    print(f"[INFO] Label: {label}")

    # ── Confirm ───────────────────────────────────────────────────────────────
    current = (motion if is_motion else static).get(label, 0)
    print(f"[INFO] Current count for {label}: {current}")

    if amount == 'all':
        confirm = input(f"\nDelete ALL {current} samples for {label}? (yes/no): ").strip().lower()
        if confirm == 'yes':
            deleted = delete_all(path, label)
            print(f"[DONE] Deleted all {deleted} samples for {label}")
        else:
            print("[CANCELLED]")
    else:
        try:
            n = int(amount)
        except ValueError:
            print("[ERROR] Invalid input. Enter a number or 'all'")
            sys.exit(1)

        if n > current:
            print(f"[WARN] You only have {current} samples — will delete all {current}")
            n = current

        confirm = input(f"\nDelete last {n} samples for {label}? (yes/no): ").strip().lower()
        if confirm == 'yes':
            deleted = delete_last_n(path, label, n)
            remaining = get_counts(path).get(label, 0)
            print(f"[DONE] Deleted {deleted} samples for {label}")
            print(f"[INFO] Remaining: {remaining} samples")
        else:
            print("[CANCELLED]")