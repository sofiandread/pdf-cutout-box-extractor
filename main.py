import os
import io
import datetime
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify, send_from_directory
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

# --- Config ---
PORT = int(os.getenv("PORT", 8080))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
DEBUG_SAVE = True
RENDER_DPI = int(os.getenv("RENDER_DPI", "300"))  # Higher DPI helps detection

# Railway typically runs the app as `python main.py` or via a Procfile. Keep module name `main`.
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def pil_to_cv2(im: Image.Image) -> np.ndarray:
    arr = np.array(im)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def render_first_page_to_image(pdf_bytes: bytes, dpi: int = 300) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("Empty PDF")
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n >= 3 else "L"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


# ---- Geometry helpers ----

def contour_rectangularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h == 0:
        return 0.0
    return float(area) / float(w * h)


def approx_is_quadrilateral(cnt: np.ndarray, eps_ratio: float = 0.02) -> Tuple[bool, np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
    if len(approx) != 4:
        return False, approx
    return True, approx


def angles_are_near_right(quad: np.ndarray, cos_thresh: float = 0.3) -> bool:
    # cos(90deg) ~ 0. For robustness allow |cos| <= cos_thresh (0..1)
    pts = quad.reshape(-1, 2)

    def angle_cos(p0, p1, p2):
        d1 = p0 - p1
        d2 = p2 - p1
        denom = (np.linalg.norm(d1) * np.linalg.norm(d2)) + 1e-9
        cosang = np.dot(d1, d2) / denom
        return abs(cosang)

    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        if angle_cos(p0, p1, p2) > cos_thresh:
            return False
    return True


def suppress_nested_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    kept = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a
        a_area = aw * ah
        contained = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            bx, by, bw, bh = b
            if bw * bh < a_area:
                continue
            if ax >= bx and ay >= by and (ax + aw) <= (bx + bw) and (ay + ah) <= (by + bh):
                contained = True
                break
        if not contained:
            kept.append(a)
    return kept


# ---- Detection ----

def detect_design_boxes_bgr(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = bgr.shape[:2]

    # 1) Preprocess
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # normalize contrast
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Edge detection (auto thresholds from median)
    v = np.median(gray_blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_blur, lower, upper)

    # 3) Morphology to close gaps in rectangle borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) Find contours
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) Filter contours by size/shape
    page_area = W * H
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02 * page_area:  # ignore very small shapes (<2% of page)
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.1 * W or h < 0.1 * H:  # ignore narrow stripes and small rects
            continue
        rect_prop = contour_rectangularity(c)  # area / (w*h)
        if rect_prop < 0.70:  # prefer well-filled rectangles
            continue
        ok_quad, quad = approx_is_quadrilateral(c, eps_ratio=0.02)
        if not ok_quad:
            continue
        if not angles_are_near_right(quad, cos_thresh=0.35):
            continue
        aspect = w / float(h)
        if aspect < 0.2 or aspect > 5.0:
            continue
        boxes.append((x, y, w, h))

    # Fallback path if nothing found (adaptive threshold)
    if not boxes:
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 41, 8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.02 * page_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 0.1 * W or h < 0.1 * H:
                continue
            rect_prop = contour_rectangularity(c)
            if rect_prop < 0.70:
                continue
            ok_quad, quad = approx_is_quadrilateral(c, eps_ratio=0.03)
            if not ok_quad:
                continue
            if not angles_are_near_right(quad, cos_thresh=0.40):
                continue
            aspect = w / float(h)
            if aspect < 0.2 or aspect > 5.0:
                continue
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    boxes = suppress_nested_boxes(boxes)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    return boxes  # return ALL big boxes, sorted by area desc


def draw_debug_overlay(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    overlay = bgr.copy()
    cv2.rectangle(overlay, (5, 5), (bgr.shape[1] - 6, bgr.shape[0] - 6), (0, 0, 0), 2)
    for idx, (x, y, w, h) in enumerate(boxes):
        color = (0, 180, 255) if idx < 2 else (180, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 6)
        cv2.putText(overlay, f"#{idx+1} {w}x{h}", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return overlay


@app.route("/")
def root():
    return jsonify({"status": "ok", "message": "PDF cutout box extractor is running"})


@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    # Keep POST field name 'data' for n8n compatibility
    if 'data' not in request.files:
        return jsonify({"error": "No file part 'data' in request"}), 400

    f = request.files['data']
    pdf_bytes = f.read()

    try:
        pil_img = render_first_page_to_image(pdf_bytes, dpi=RENDER_DPI)
    except Exception as e:
        return jsonify({"error": f"Failed to render PDF: {str(e)}"}), 400

    bgr = pil_to_cv2(pil_img)
    boxes = detect_design_boxes_bgr(bgr)  # ALL big boxes

    overlay_bgr = draw_debug_overlay(bgr, boxes)
    debug_name = f"debug_{_ts()}.png"
    debug_path = os.path.join(STATIC_DIR, debug_name)
    if DEBUG_SAVE:
        cv2.imwrite(debug_path, overlay_bgr)

    results: List[Dict] = []
    for i, (x, y, w, h) in enumerate(boxes):
        crop = bgr[y:y + h, x:x + w]
        crop_img = cv2_to_pil(crop)
        crop_name = f"cutout_{_ts()}_{i+1}.png"
        crop_path = os.path.join(STATIC_DIR, crop_name)
        crop_img.save(crop_path, format="PNG")
        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{crop_name}",
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "page": 1
        })

    response = {
        "cutouts": results,  # array of objects
        "debug_overlay_url": f"{request.host_url.rstrip('/')}/static/{debug_name}",
    }

    return jsonify(response), 200


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
