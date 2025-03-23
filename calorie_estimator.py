import os
import numpy as np
import requests
from PIL import Image
from functools import lru_cache
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === CONFIG ===
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "food_recognition_model.h5")  # Or .keras
IMAGE_SIZE = (128, 128)

# === CLASS LABELS (sorted by directory name) ===
IMAGE_DIR = os.path.join("data", "food-101", "food-101", "images")
CLASS_LABELS = sorted([
    folder for folder in os.listdir(IMAGE_DIR)
    if os.path.isdir(os.path.join(IMAGE_DIR, folder))
])

# === NUTRITIONIX CREDENTIALS ===
API_ID = "13b073e4"
API_KEY = "9793e8fcbbf96cb737bb15e58ae7869f"
NUTRITIONIX_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

# ✅ LOAD MODEL ONCE & CACHE
@lru_cache(maxsize=1)
def get_model():
    print("📦 Loading model (optimized for CPU)...")
    return load_model(MODEL_PATH)

# ✅ IMAGE PREPROCESSING FUNCTION
def preprocess_image(image_path):
    img = Image.open(image_path).resize(IMAGE_SIZE).convert("RGB")
    img_array = img_to_array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ CALORIE ESTIMATION FUNCTION
def estimate_calories(image_path):
    if not os.path.exists(image_path):
        return f"❌ Image not found: {image_path}"

    # --- Step 1: Preprocess Image ---
    image_data = preprocess_image(image_path)

    # --- Step 2: Predict Food Category ---
    model = get_model()
    prediction = model.predict(image_data, verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]
    food_name = CLASS_LABELS[predicted_index].replace("_", " ")
    print(f"🔍 Predicted Food: {food_name}")

    # --- Step 3: Query Nutritionix API ---
    headers = {
        "x-app-id": API_ID,
        "x-app-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": food_name}

    response = requests.post(NUTRITIONIX_URL, headers=headers, json=data)

    # --- Step 4: Parse Response ---
    if response.status_code == 200:
        result = response.json()
        if result.get("foods"):
            food_data = result["foods"][0]
            return (
                f"🍽️ Food: {food_name.title()}\n"
                f"🔢 Serving: {food_data['serving_qty']} {food_data['serving_unit']}\n"
                f"🔥 Calories: {food_data['nf_calories']} kcal"
            )
        else:
            return f"⚠️ No nutrition data found for {food_name}"
    else:
        return f"❌ API Error {response.status_code}: {response.text}"

# ✅ SAMPLE USAGE
if __name__ == "__main__":
    sample_img = os.path.join(BASE_DIR, "data", "food-101", "food-101", "images", "apple_pie", "78081.jpg")
    print(estimate_calories(sample_img))
