import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- LOAD DATA ----------------
X = np.load("X.npy")
y = np.load("y.npy")

num_classes = len(np.unique(y))

# ---------------- TRAIN / VAL SPLIT ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- CLASS WEIGHTS ----------------
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(i): float(w) for i, w in zip(classes, weights)}

print("Class Weights:", class_weights)

# ---------------- MODEL ----------------
model = models.Sequential([

    layers.Input(shape=(45, 136)),

    layers.Masking(mask_value=0.),

    layers.LSTM(96, return_sequences=True),
    layers.Dropout(0.3),

    layers.LSTM(64, return_sequences=True),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
]

# ---------------- TRAIN ----------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred_classes))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_classes))

# ---------------- SAVE MODEL ----------------
model.save("isl_model.keras")

print("\n✅ Training complete. Model saved as isl_model.keras")