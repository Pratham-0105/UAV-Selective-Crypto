import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

# Import the exact pipeline logic from demo_app
import sys
sys.path.append(os.path.dirname(__file__))
from demo_app import process_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'latest_drone_capture.jpg')

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        file.save(UPLOAD_FOLDER)
        print(f"✅ Received new image from DroneCam: {UPLOAD_FOLDER}")
        return jsonify({"success": "Image saved successfully"}), 200

@app.route('/process_on_drone', methods=['POST'])
def process_on_drone():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
        
    file = request.files['image']
    if file:
        img = Image.open(file).convert("RGB")
        print("🧠 Edge Cloud: Running local ML + Encryption simulation...")
        
        # Run the exact same ML logic as the ground station
        map_img, enc_img, dec_img, stats = process_image(img)
        
        # We only return the map and encrypted image to the phone screen
        return jsonify({
            "map_image": pil_to_base64(map_img),
            "encrypted_image": pil_to_base64(enc_img)
        }), 200

if __name__ == '__main__':
    print("🚁 Starting Drone Receiver Server on port 5001...")
    print("Ensure your laptop and phone are on the same Wi-Fi network.")
    # Threaded=False is important when sharing the Numba/Scikit models in memory
    app.run(host='0.0.0.0', port=5001, threaded=False)
