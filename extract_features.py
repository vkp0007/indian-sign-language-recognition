import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------------- CONFIG ----------------
VIDEO_PATH = "dataset_isl"
SAVE_PATH = "features"
MAX_FRAMES = 45
MIN_RATIO = 0.2

os.makedirs(SAVE_PATH, exist_ok=True)

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


# ---------------- MEDIAPIPE ----------------
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
)

# ---------------- PROCESS ----------------
for label in os.listdir(VIDEO_PATH):

    class_path = os.path.join(VIDEO_PATH, label)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {label}")

    for video_file in os.listdir(class_path):

        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open: {video_file}")
            continue

        sequence = []
        detected_frames = 0
        gesture_frames = 0

        # 🔥 NEW: gesture control
        started = False
        miss_count = 0
        MAX_MISS = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------- PREPROCESS ----------------
            h, w, _ = frame.shape

            # ✅ balanced crop
            frame = frame[int(h*0.1):int(h*0.9), int(w*0.15):int(w*0.85)]

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = hand_landmarker.detect(mp_image)

            # 🔥 AUTO START
            if not started:
                if result.hand_landmarks:
                    started = True
                else:
                    continue

            gesture_frames += 1

            if result.hand_landmarks:
                detected_frames += 1
                miss_count = 0

                keypoints = extract_keypoints(result)
                sequence.append(keypoints)

            else:
                miss_count += 1

            # 🔥 AUTO STOP
            if started and miss_count > MAX_MISS:
                break

        cap.release()

        # ---------------- QUALITY CHECK ----------------
        if gesture_frames == 0:
            print(f"❌ No gesture found: {video_file}")
            continue

        ratio = detected_frames / gesture_frames

        if ratio < MIN_RATIO:
            print(f"❌ Skipped {video_file} (low quality: {ratio:.2f})")
            continue

        if len(sequence) == 0:
            print(f"❌ No valid frames: {video_file}")
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

        print(f"✅ Saved: {save_name} | ratio: {ratio:.2f} | frames: {len(sequence)}")

hand_landmarker.close()

print("\n🚀 Feature extraction complete.")