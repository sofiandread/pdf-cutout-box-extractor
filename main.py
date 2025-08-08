from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from PIL import Image
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Art Cutout Extractor API is live!"

@app.route('/detect-art-format', methods=['POST'])
def detect_art_format():
    try:
        # Save uploaded file
        file = request.files['data']
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Open PDF, only process the first page
        doc = fitz.open(filepath)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)  # High-res
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Convert to grayscale and find rectangles
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for rectangles (based on area and shape)
        cutouts = []
        min_area = 10000  # You can adjust this based on your proofs
        for i, cnt in enumerate(contours):
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            # Rectangle: 4 corners, reasonable area, reasonable aspect
            if len(approx) == 4 and area > min_area and 0.5 < aspect_ratio < 2.5:
                crop = img_array[y:y+h, x:x+w]
                out_filename = f"cutout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                out_path = os.path.join(STATIC_FOLDER, out_filename)
                # Save with PIL for best compatibility
                im = Image.fromarray(crop)
                im.save(out_path)
                url = request.host_url.rstrip('/') + '/static/' + out_filename
                cutouts.append({
                    "url": url,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                })

        # Optional: sort by x or y if you want a consistent order (e.g., left-to-right)
        cutouts = sorted(cutouts, key=lambda k: (k["x"], k["y"]))

        # Clean up upload after processing
        os.remove(filepath)

        return jsonify({"cutouts": cutouts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve files from static directory
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
