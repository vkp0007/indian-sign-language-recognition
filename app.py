import streamlit as st
import numpy as np
import tensorflow as tf

st.title("🖐️ ISL Recognition System")

st.write("Model loading...")

# Load model
model = tf.keras.models.load_model("isl_model.keras")
labels = np.load("labels.npy", allow_pickle=True)

st.success("✅ Model loaded successfully!")

st.write("Classes:", labels)

st.write("🚀 Deployment successful. Next step: add webcam support.")