import os
import numpy as np
from collections import Counter

FEATURE_PATH = "features"
EXPECTED_FEATURES = 136
SEQ_LENGTH = 45  # 🔥 match extraction

X, y = [], []
labels = {}
label_index = 0

class_counts = Counter()

# ---------- MAIN ----------
files = sorted(os.listdir(FEATURE_PATH))

for file in files:

    if not file.endswith(".npy"):
        continue

    # label extraction
    word = file.split("_")[0]

    if word not in labels:
        labels[word] = label_index
        label_index += 1

    label_id = labels[word]

    data = np.load(os.path.join(FEATURE_PATH, file))

    # ---------- VALIDATION ----------
    if len(data.shape) != 2:
        print("❌ Skipping (bad shape):", file)
        continue

    if data.shape[1] != EXPECTED_FEATURES:
        print("❌ Wrong feature size:", file, data.shape)
        continue

    # ---------- ENSURE LENGTH ----------
    if len(data) != SEQ_LENGTH:
        print("⚠️ Fixing length:", file, data.shape)

        if len(data) > SEQ_LENGTH:
            data = data[:SEQ_LENGTH]
        else:
            pad = np.zeros((SEQ_LENGTH - len(data), EXPECTED_FEATURES))
            data = np.vstack([data, pad])

    # ---------- ADD TO DATASET ----------
    X.append(data)
    y.append(label_id)
    class_counts[word] += 1


# ---------- FINAL ----------
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# ---------- NORMALIZATION ----------
mean = X.mean()
std = X.std() + 1e-8

X = (X - mean) / std

np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

# save labels
label_list = [k for k, v in sorted(labels.items(), key=lambda x: x[1])]
np.save("labels.npy", label_list)

# save dataset
np.save("X.npy", X)
np.save("y.npy", y)

# ---------- SUMMARY ----------
print("\n✅ Dataset Summary:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(label_list))

print("\nClass Distribution:")
for k, v in class_counts.items():
    print(f"{k}: {v}")