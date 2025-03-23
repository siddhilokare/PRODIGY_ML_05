import os
from flask import Flask, request, jsonify
from calorie_estimator import estimate_calories

# Initialize Flask app
app = Flask(__name__)

# Upload folder for temporary images
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the Food Calorie Estimator API 🍽️"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        # Call calorie estimation
        result = estimate_calories(image_path)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
