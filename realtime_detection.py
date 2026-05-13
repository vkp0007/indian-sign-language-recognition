import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ---------------- LOAD ----------------
model = tf.keras.models.load_model("isl_model_regularized.keras")
labels = np.load("labels.npy", allow_pickle=True)
mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

# ---------------- CONFIG ----------------
SEQ_LENGTH = 45
PREDICTION_WINDOW = 8

CONF_THRESHOLD = 0.7
STABILITY_THRESHOLD = 0.7

MIN_PRED_FRAMES = 4     # consistency requirement
COOLDOWN_FRAMES = 8     # delay between words
PAUSE_FRAMES = 10

# ---------------- STATE ----------------
sequence = deque(maxlen=SEQ_LENGTH)
pred_buffer = deque(maxlen=PREDICTION_WINDOW)
sentence = []

pause_counter = 0
cooldown = 0

current_word = "..."
confidence = 0.0

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

    return np.concatenate([coords.flatten(), distances])


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


# ---------------- SMOOTHING ----------------
def smooth_seq(seq):
    smoothed = seq.copy()
    for i in range(1, len(seq) - 1):
        smoothed[i] = (seq[i - 1] + seq[i] + seq[i + 1]) / 3.0
    return smoothed


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

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

print("🚀 Realtime ISL started | Show sign → remove hand → next sign")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- PREPROCESS ----------------
    h, w, _ = frame.shape
    frame = frame[int(h * 0.1):int(h * 0.9), int(w * 0.15):int(w * 0.85)]
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = hand_landmarker.detect(mp_image)

    # ---------------- COOLDOWN UPDATE ----------------
    if cooldown > 0:
        cooldown -= 1

    # ---------------- NO HAND = PAUSE ----------------
    if not result.hand_landmarks:

        pause_counter += 1

        if pause_counter > PAUSE_FRAMES and len(pred_buffer) > 0:

            preds = [p for p, c in pred_buffer]
            most_common, count = Counter(preds).most_common(1)[0]
            stability = count / len(preds)

            confs = [c for p, c in pred_buffer if p == most_common]
            avg_conf = np.mean(confs) if confs else 0

            if avg_conf > CONF_THRESHOLD and stability > STABILITY_THRESHOLD:

                if most_common < len(labels):
                    word = labels[most_common]

                    if cooldown == 0:
                        if len(sentence) == 0 or sentence[-1] != word:
                            sentence.append(word)
                            cooldown = COOLDOWN_FRAMES

            pred_buffer.clear()
            sequence = deque(maxlen=SEQ_LENGTH)

        current_word = "..."
        confidence = 0.0

        cv2.putText(frame, "Pause", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame,
                    "Sentence: " + " ".join(sentence[-4:]),
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("ISL Realtime", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ---------------- HAND PRESENT ----------------
    pause_counter = 0

    keypoints = extract_keypoints(result)

    # 🔥 Match training (remove first 3 features)
    keypoints = keypoints[3:]

    sequence.append(keypoints)

    # ---------------- PREDICTION ----------------
    if len(sequence) == SEQ_LENGTH:

        seq = np.array(sequence)

        # smoothing
        seq = smooth_seq(seq)

        # normalization
        seq = (seq - mean[3:]) / std[3:]

        seq = np.expand_dims(seq, axis=0)

        probs = model.predict(seq, verbose=0)[0]
        pred = np.argmax(probs)
        confidence = probs[pred]

        # stronger filtering
        if confidence > 0.65:
            pred_buffer.append((pred, confidence))

        preds = [p for p, _ in pred_buffer]

        if preds:
            most_common, count = Counter(preds).most_common(1)[0]

            if count >= MIN_PRED_FRAMES and most_common < len(labels):
                current_word = labels[most_common]

    # ---------------- DISPLAY ----------------
    cv2.putText(frame,
                f"Word: {current_word}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(frame,
                f"Conf: {confidence:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    cv2.putText(frame,
                "Sentence: " + " ".join(sentence[-4:]),
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.imshow("ISL Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()