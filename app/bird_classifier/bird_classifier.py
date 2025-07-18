import onnxruntime as ort
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.csv")

# Load labels
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_data = np.array(image).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # CHW
    img_data = np.expand_dims(img_data, axis=0)  # NCHW
    return img_data

def classify_species_chriamue(image_path):
    input_tensor = preprocess_image(image_path)
    outputs = session.run(None, {input_name: input_tensor})

    probs = outputs[0][0]
    if not probs.any():
        return "Unknown", 0.0

    idx = np.argmax(probs)
    
    if idx >= len(labels):
        return "Unknown", round(float(probs[idx]), 3)

    confidence = float(probs[idx])
    return labels[idx], round(confidence, 3)