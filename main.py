from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------- Tunables (can be overridden via env vars) ----------
DPI = int(os.environ.get("EXTRACT_DPI", 300))
# Keep boxes whose area is between 5% and 60% of the full page area
MIN_AREA_FRAC = float(os.environ.get("MIN_AREA_FRAC", 0.05))
MAX_AREA_FRAC = float(os.environ.get("MAX_AREA_FRAC", 0.60))
# Minimum side length in pixels at DPI above
MIN_SIDE_PX = int(os.environ.get("MIN_SIDE_PX", 350))
# Allow fairly square to wide rectangles
MIN_ASPECT = float(os.environ.get("MIN_ASPECT", 0.55))  # w/h
MAX_ASPECT = float(os.environ.get("MAX_ASPECT", 2.2))
# Focus detection on lower portion of page (0..1). 1=bottom, 0=top
FOCUS_LOWER_FRACTION = float(os.environ.get("FOCUS_LOWER_FRACTION", 0.65))
# Return a debug overlay image?
RETURN_DEBUG = os.environ.get("RETURN_DEBUG", "1") == "1"
# -------------------------------------------------------------

@app.route("/")
def home():
    return "Art Cutout Extractor API is live!"

@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    """
    POST multipart/form-data with a file field named 'data' (PDF).
    Returns JSON:
    {
      "cutouts": [
        {"url": "...", "x":..., "y":..., "width":..., "height":..., "page": 1}
      ],
      "debug_overlay_url": "..."   # when RETURN_DEBUG=1
    }
    """
    try:
        if "data" not in request.files:
            return jsonify({"error": "No 'data' file field in request"}), 400

        # Save upload
        f = request.files["data"]
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{ts}_{f.filename}"
        in_path = os.path.join(UPLOAD_FOLDER, safe_name)
        f.save(in_path)

        # Render FIRST page at DPI
        doc = fitz.open(in_path)
        if doc.page_count == 0:
            return jsonify({"error": "PDF has no pages"}), 400
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=DPI)
        # Convert PyMuPDF pixmap to RGB numpy
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        H, W, _ = img.shape
        full_area = H * W

        # Optionally focus on lower portion of page to avoid headers/mockups
        y0 = int(H * (1 - FOCUS_LOWER_FRACTION))  # start row
        roi = img[y0:, :].copy()

        # --- Preprocess for robust box borders ---
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        # light blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edges + dilation to connect rounded rectangle borders
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find outermost contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        overlay = roi.copy()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Map back to full image coordinates
            abs_x, abs_y = x, y + y0
            area = w * h
            area_frac = area / full_area if full_area > 0 else 0

            # Filter 1: big enough and not absurdly big
            if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
                continue
            if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
                continue

            # Filter 2: aspect ratio
            aspect = w / h if h else 0
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            # Filter 3: rectangularity (extent ~ how filled the bounding box is)
            rect_area = w * h
            cnt_area = cv2.contourArea(cnt)
            extent = cnt_area / rect_area if rect_area > 0 else 0
            if extent < 0.60:  # very fragmented shapes are out
                continue

            # Passed all filters
            candidates.append((abs_x, abs_y, w, h))

            # draw debug rect
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # If nothing found, try a fallback: relax area lower bound a bit
        if not candidates:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                abs_x, abs_y = x, y + y0
                area = w * h
                area_frac = area / full_area if full_area > 0 else 0
                if area_frac < (MIN_AREA_FRAC * 0.6) or area_frac > MAX_AREA_FRAC:
                    continue
                if w < (MIN_SIDE_PX * 0.9) or h < (MIN_SIDE_PX * 0.9):
                    continue
                aspect = w / h if h else 0
                if aspect < (MIN_ASPECT * 0.9) or aspect > (MAX_ASPECT * 1.1):
                    continue
                cnt_area = cv2.contourArea(cnt)
                extent = cnt_area / (w * h) if (w * h) > 0 else 0
                if extent < 0.55:
                    continue
                candidates.append((abs_x, abs_y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Sort left-to-right then top-to-bottom for stable ordering
        candidates.sort(key=lambda r: (r[0], r[1]))

        cutouts = []
        for i, (cx, cy, cw, ch) in enumerate(candidates):
            crop = img[cy:cy + ch, cx:cx + cw]
            out_name = f"cutout_{ts}_{i}.png"
            out_path = os.path.join(STATIC_FOLDER, out_name)
            Image.fromarray(crop).save(out_path)
            url = request.host_url.rstrip("/") + "/static/" + out_name
            cutouts.append({
                "url": url,
                "x": int(cx),
                "y": int(cy),
                "width": int(cw),
                "height": int(ch),
                "page": 1
            })

        # Save debug overlay if requested
        debug_url = None
        if RETURN_DEBUG:
            dbg_name = f"debug_{ts}.png"
            dbg_path = os.path.join(STATIC_FOLDER, dbg_name)
            # Put overlay back into full image canvas for clarity
            full_overlay = img.copy()
            full_overlay[y0:, :] = overlay
            Image.fromarray(full_overlay).save(dbg_path)
            debug_url = request.host_url.rstrip("/") + "/static/" + dbg_name

        # Clean up uploaded file
        try:
            os.remove(in_path)
        except Exception:
            pass

        return jsonify({
            "cutouts": cutouts,
            "debug_overlay_url": debug_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
