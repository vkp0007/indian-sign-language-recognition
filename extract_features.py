import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp
import csv
import pandas as pd
import matplotlib.pyplot as plt

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------------- CONFIG ----------------
VIDEO_PATH = "dataset_isl"
SAVE_PATH = "features"

MAX_FRAMES = 45
MIN_RATIO = 0.4
MIN_GESTURE_FRAMES = 15
MAX_MISS = 20

EXCLUDED_CLASSES = ["who"]   # 🔥 removed class

os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------- CSV LOG ----------------
csv_file = open("results_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Video", "Label", "Total_Frames", "Gesture_Frames", "Detected_Frames", "Ratio"])

total_videos = 0
used_videos = 0

# ---------------- FEATURE FUNCTIONS ----------------
def normalize_landmarks(coords):
    wrist = coords[0]
    coords = coords - wrist

    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale

    return coords


def extract_hand_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    coords = normalize_landmarks(coords)

    tips = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(coords[t]) for t in tips]

    return np.concatenate([coords.flatten(), distances])  # 68


def extract_keypoints(result):
    left = np.zeros(68)
    right = np.zeros(68)

    if result.hand_landmarks and result.handedness:
        for i in range(len(result.hand_landmarks)):
            label = result.handedness[i][0].category_name
            feats = extract_hand_features(result.hand_landmarks[i])

            if label == "Left":
                left = feats
            elif label == "Right":
                right = feats

    return np.concatenate([left, right])  # 136


# ---------------- PROCESS ----------------
for label in os.listdir(VIDEO_PATH):

    # 🔥 SKIP REMOVED CLASS
    if label in EXCLUDED_CLASSES:
        print(f"🚫 Skipping class: {label}")
        continue

    class_path = os.path.join(VIDEO_PATH, label)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📁 Processing class: {label}")

    for video_file in os.listdir(class_path):

        total_videos += 1

        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                num_hands=2,
                running_mode=vision.RunningMode.VIDEO,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15,
                min_tracking_confidence=0.15
            )
        )

        sequence = []
        detected_frames = 0
        gesture_frames = 0

        started = False
        miss_count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = frame_id * 33
            result = hand_landmarker.detect_for_video(mp_image, timestamp)

            detected = bool(result.hand_landmarks)

            if not started:
                if detected:
                    started = True
                else:
                    continue

            gesture_frames += 1

            if detected:
                detected_frames += 1
                miss_count = 0

                keypoints = extract_keypoints(result)
                sequence.append(keypoints)

            else:
                miss_count += 1

            if miss_count > MAX_MISS:
                break

        cap.release()
        hand_landmarker.close()

        # ---------------- VALIDATION ----------------
        if gesture_frames < MIN_GESTURE_FRAMES:
            continue

        ratio = detected_frames / gesture_frames if gesture_frames > 0 else 0

        if ratio < MIN_RATIO or len(sequence) == 0:
            continue

        sequence = np.array(sequence)

        # ---------------- FIX LENGTH ----------------
        if len(sequence) > MAX_FRAMES:
            indices = np.linspace(0, len(sequence)-1, MAX_FRAMES).astype(int)
            sequence = sequence[indices]

        if len(sequence) < MAX_FRAMES:
            pad = np.zeros((MAX_FRAMES - len(sequence), 136))
            sequence = np.vstack([sequence, pad])

        # ---------------- SAVE ----------------
        save_name = f"{label}_{video_file.split('.')[0]}.npy"
        np.save(os.path.join(SAVE_PATH, save_name), sequence)

        used_videos += 1

        csv_writer.writerow([
            video_file,
            label,
            frame_id,
            gesture_frames,
            detected_frames,
            round(ratio, 3)
        ])

        print(f"✅ {save_name} | Ratio: {ratio:.2f}")

csv_file.close()

print("\n🚀 Feature extraction complete.")

# =====================================================
# 📊 RESULTS (FOR PAPER)
# =====================================================

df = pd.read_csv("results_log.csv")

df["Detection_%"] = df["Ratio"] * 100

print("\n📊 OVERALL RESULTS")
print(f"Total Videos Processed: {total_videos}")
print(f"Valid Videos Used: {used_videos}")
print(f"Average Detection %: {df['Detection_%'].mean():.2f}")

# -------- CLASS-WISE --------
class_summary = df.groupby("Label")["Detection_%"].mean().reset_index()
class_summary.to_csv("feature_class_results.csv", index=False)

print("📊 Class-wise results saved")

# -------- HIGH-QUALITY GRAPH --------
plt.figure(figsize=(8,5), dpi=300)
plt.hist(df["Detection_%"], bins=12, edgecolor='black')

plt.xlabel("Detection Accuracy (%)")
plt.ylabel("Number of Videos")
plt.title("Detection Accuracy Distribution")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig("detection_distribution.pdf")
plt.close()

print("📈 Graph saved → detection_distribution.pdf")