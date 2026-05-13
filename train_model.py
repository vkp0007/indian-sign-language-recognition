import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------- LOAD DATA ----------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

labels = np.load("labels.npy", allow_pickle=True)

# 🔥 SAFETY CHECK
if "who" in labels:
    raise ValueError("❌ 'who' still present")

num_classes = len(labels)

print("Train:", X_train.shape, "Val:", X_val.shape)

# =====================================================
# 🔧 FEATURE CLEANING
# =====================================================
X_train = X_train[:, :, 3:]
X_val   = X_val[:, :, 3:]

assert X_train.shape[2] == 133

# =====================================================
# 🔧 TEMPORAL SMOOTHING
# =====================================================
def smooth_sequence(X):
    X_smooth = X.copy()
    for i in range(X.shape[0]):
        for t in range(1, X.shape[1] - 1):
            X_smooth[i, t] = (X[i, t-1] + X[i, t] + X[i, t+1]) / 3.0
    return X_smooth

X_train = smooth_sequence(X_train)
X_val   = smooth_sequence(X_val)

# =====================================================
# 🔧 STRONGER AUGMENTATION
# =====================================================
def augment(X):
    noise = np.random.normal(0, 0.02, X.shape)  # 🔥 increased noise
    return X + noise

X_train_aug = augment(X_train)

X_train = np.concatenate([X_train, X_train_aug])
y_train = np.concatenate([y_train, y_train])

print("After augmentation:", X_train.shape)

# =====================================================
# 🔧 CLASS WEIGHTS
# =====================================================
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = {int(i): float(w) for i, w in zip(classes, weights)}

# =====================================================
# 🧠 MODEL (OVERFITTING CONTROL)
# =====================================================
model = models.Sequential([
    layers.Input(shape=(45, 133)),

    # 🔥 Noise for generalization
    layers.GaussianNoise(0.02),

    # 🔥 Reduced + regularized LSTM
    layers.Bidirectional(
        layers.LSTM(48, return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.001))
    ),
    layers.Dropout(0.5),

    layers.Bidirectional(
        layers.LSTM(24,
                    kernel_regularizer=regularizers.l2(0.001))
    ),
    layers.Dropout(0.5),

    # 🔥 Dense with regularization
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# =====================================================
# 🔁 CALLBACKS (IMPROVED)
# =====================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,   # 🔥 reduced
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# =====================================================
# 🚀 TRAIN
# =====================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# 📊 TRAINING RESULTS
# =====================================================
history_dict = history.history

best_val_acc = max(history_dict['val_accuracy'])
best_train_acc = max(history_dict['accuracy'])

print("\n📊 BEST RESULTS")
print("Train Accuracy:", best_train_acc)
print("Val Accuracy:", best_val_acc)
print("Gap:", best_train_acc - best_val_acc)

# =====================================================
# 📈 TRAINING CURVES
# =====================================================
plt.figure(figsize=(10,4), dpi=300)

plt.subplot(1,2,1)
plt.plot(history_dict['accuracy'], label='Train')
plt.plot(history_dict['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_dict['loss'], label='Train')
plt.plot(history_dict['val_loss'], label='Val')
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_regularized.pdf")
plt.close()

# =====================================================
# 📊 EVALUATION
# =====================================================
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(
    y_val,
    y_pred_classes,
    labels=np.arange(num_classes),
    target_names=labels,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv("classification_report_regularized.csv")

cm = confusion_matrix(
    y_val,
    y_pred_classes,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(8,6), dpi=300)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_regularized.pdf")
plt.close()

# =====================================================
# 📊 FINAL METRICS
# =====================================================
accuracy = accuracy_score(y_val, y_pred_classes)
precision = precision_score(y_val, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred_classes, average='weighted', zero_division=0)

print("\n📊 FINAL PERFORMANCE")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# =====================================================
# 💾 SAVE MODEL
# =====================================================
model.save("isl_model_regularized.keras")

print("\n✅ Done. Regularized model saved.")