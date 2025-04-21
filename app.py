from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import hashlib

app = Flask(__name__)

model = None
model_path = "new_model.keras"
temp_model_path = "temp_model.keras"
drive_url = "https://drive.google.com/uc?id=1Q3zFAHsxijJY2ywoTIK6yTbQM9GpJrb5"

# === حساب الهاش لأي ملف ===
def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# === تحميل أو تحديث الموديل ===
def load_or_update_model():
    if os.path.exists(model_path):
        gdown.download(drive_url, temp_model_path, quiet=False)
        old_hash = get_file_hash(model_path)
        new_hash = get_file_hash(temp_model_path)

        if old_hash != new_hash:
            print("🔁 Model updated.")
            os.remove(model_path)
            os.rename(temp_model_path, model_path)
        else:
            print("✅ Model is up-to-date.")
            os.remove(temp_model_path)
    else:
        print("⬇️ First time model download...")
        gdown.download(drive_url, model_path, quiet=False)

# === تحميل الموديل أول مرة في أول request فقط ===
def ensure_model_loaded():
    global model
    if model is None:
        load_or_update_model()
        model = tf.keras.models.load_model(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    ensure_model_loaded()

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    img = Image.open(image).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    categories = ["dry", "normal", "oily"]
    predicted_index = np.argmax(prediction)
    predicted_class = categories[predicted_index]
    confidence = float(prediction[0][predicted_index])

    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": {
            "dry": float(prediction[0][0]),
            "normal": float(prediction[0][1]),
            "oily": float(prediction[0][2])
        }
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
