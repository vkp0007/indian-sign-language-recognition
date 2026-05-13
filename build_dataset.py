import os
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FEATURE_PATH = "features"

EXPECTED_FEATURES = 136
SEQ_LENGTH = 45

MIN_SAMPLES = 15
MAX_SAMPLES = 20

X, y = [], []
labels = {}
label_index = 0

class_files = defaultdict(list)

# ----------- TRACKING -----------
total_files = 0
valid_files = 0
skipped_files = 0

skip_reasons = {
    "bad_shape": 0,
    "wrong_features": 0,
    "zero_sequence": 0
}

# ---------- STEP 1 ----------
files = sorted(os.listdir(FEATURE_PATH))

for file in files:
    if not file.endswith(".npy"):
        continue

    total_files += 1
    word = file.split("_")[0]
    class_files[word].append(file)

# ---------- STEP 2 ----------
filtered_classes = {}

for word, file_list in class_files.items():

    if len(file_list) < MIN_SAMPLES:
        print(f"❌ Skipping class '{word}' (too few: {len(file_list)})")
        continue

    filtered_classes[word] = file_list[:MAX_SAMPLES]

# ---------- STEP 3 ----------
for word, file_list in filtered_classes.items():

    if word not in labels:
        labels[word] = label_index
        label_index += 1

    label_id = labels[word]

    for file in file_list:

        path = os.path.join(FEATURE_PATH, file)
        data = np.load(path)

        # VALIDATION
        if len(data.shape) != 2:
            skip_reasons["bad_shape"] += 1
            skipped_files += 1
            continue

        if data.shape[1] != EXPECTED_FEATURES:
            skip_reasons["wrong_features"] += 1
            skipped_files += 1
            continue

        if np.all(data == 0):
            skip_reasons["zero_sequence"] += 1
            skipped_files += 1
            continue

        valid_files += 1

        # FIX LENGTH
        if len(data) != SEQ_LENGTH:
            if len(data) > SEQ_LENGTH:
                data = data[:SEQ_LENGTH]
            else:
                pad = np.zeros((SEQ_LENGTH - len(data), EXPECTED_FEATURES))
                data = np.vstack([data, pad])

        X.append(data)
        y.append(label_id)

# ---------- CONVERT ----------
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# ---------- NORMALIZATION ----------
mean = X.mean(axis=(0, 1))
std = X.std(axis=(0, 1)) + 1e-8
X = (X - mean) / std

np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

# =====================================================
# 🔥 STRATIFIED SPLIT (IMPORTANT)
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

# ---------- SAVE LABELS ----------
label_list = [k for k, v in sorted(labels.items(), key=lambda x: x[1])]
np.save("labels.npy", label_list)

# =====================================================
# 📊 RESULTS
# =====================================================

print("\n📊 DATASET STATISTICS")
print(f"Total files found: {total_files}")
print(f"Valid samples used: {valid_files}")
print(f"Skipped samples: {skipped_files}")

print("\n📊 SKIP REASONS")
for k, v in skip_reasons.items():
    print(f"{k}: {v}")

print("\n📊 DATA SPLIT")
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ---------- CLASS DISTRIBUTION ----------
class_dist = []
for word, file_list in filtered_classes.items():
    class_dist.append([word, len(file_list)])

df_dist = pd.DataFrame(class_dist, columns=["Class", "Samples"])

# 🔥 SORT for better visualization
df_dist = df_dist.sort_values(by="Samples", ascending=False)

df_dist.to_csv("class_distribution.csv", index=False)

# ---------- GRAPH ----------
plt.figure(figsize=(10, 5), dpi=300)
plt.bar(df_dist["Class"], df_dist["Samples"], edgecolor='black')

avg = df_dist["Samples"].mean()
plt.axhline(avg, linestyle='--', linewidth=1)
plt.text(len(df_dist)-1, avg+0.5, f"Avg: {avg:.1f}", ha='right')

plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution of Dataset")

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("class_distribution.pdf")
plt.close()

# ---------- FINAL ----------
print("\n✅ FINAL DATASET")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Classes:", len(label_list))