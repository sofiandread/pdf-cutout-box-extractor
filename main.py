import os
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
    """PIL RGB/RGBA -> OpenCV BGR/BGRA"""
    arr = np.array(im)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR/BGRA -> PIL RGB/RGBA"""
    if arr.ndim == 2:
        return Image.fromarray(arr)
    if arr.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    elif arr.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(arr)


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


# ---- Detection (demo box) ----

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
        if area < 0.02 * page_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.1 * W or h < 0.1 * H:
            continue
        rect_prop = contour_rectangularity(c)
        if rect_prop < 0.70:
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

    # Fallback path
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
    return boxes


def draw_debug_overlay(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    overlay = bgr.copy()
    cv2.rectangle(overlay, (5, 5), (bgr.shape[1] - 6, bgr.shape[0] - 6), (0, 0, 0), 2)
    for idx, (x, y, w, h) in enumerate(boxes):
        color = (0, 180, 255) if idx < 2 else (180, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 6)
        cv2.putText(overlay, f"#{idx+1} {w}x{h}", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return overlay


# ---- Tight-crop helpers (no color changes) ----

def _dominant_bg_lab_from_interior(rgb: np.ndarray, interior_frac: float = 0.06) -> np.ndarray:
    """
    Estimate the background color by sampling the INTERIOR of the crop (ignoring borders).
    Returns 1x1x3 Lab median.
    """
    H, W, _ = rgb.shape
    m = max(2, int(round(min(H, W) * interior_frac)))
    inner = rgb[m:H-m, m:W-m, :]
    if inner.size == 0:
        inner = rgb
    inner_lab = cv2.cvtColor(inner.astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    med = np.median(inner_lab, axis=0).astype(np.float32)[None, None, :]
    return med  # (1,1,3)


def _tight_bbox_by_deltaE(bgr_crop: np.ndarray,
                          deltaE_thresh: float = 9.0,
                          min_area_frac: float = 0.0005,
                          border_strip_frac: float = 0.02,
                          close_px: int = 1) -> Tuple[int, int, int, int]:
    """
    Build foreground mask = pixels whose Lab distance from interior background >= deltaE_thresh.
    Compute minimal bounding rectangle of that mask.
    Returns x0,y0,x1,y1 (LOCAL crop coords). Falls back to crop bounds if empty.
    """
    H, W = bgr_crop.shape[:2]
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    bg_lab = _dominant_bg_lab_from_interior(rgb, interior_frac=0.06)  # (1,1,3)
    d = np.linalg.norm(lab - bg_lab, axis=2)  # ΔE approx

    # Foreground where ΔE >= threshold
    mask = (d >= float(deltaE_thresh)).astype(np.uint8) * 255

    # Strip a thin border so we ignore the demo-box stroke/rounded corners
    b = max(2, int(round(min(H, W) * border_strip_frac)))
    mask[:b, :] = 0; mask[-b:, :] = 0; mask[:, :b] = 0; mask[:, -b:] = 0

    # Close small gaps, remove tiny specks
    if close_px > 0:
        k = np.ones((close_px, close_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    keep = np.zeros_like(mask)
    min_area = int(round(min_area_frac * H * W))
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            keep[labels == i] = 255

    if np.count_nonzero(keep) == 0:
        # Nothing confidently different from bg -> fallback to original crop bounds
        return 0, 0, W, H

    ys, xs = np.where(keep > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


# ---- Routes ----

@app.route("/")
def root():
    return jsonify({"status": "ok", "message": "PDF cutout box extractor is running"})


@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    # Keep POST field name 'pdf' for n8n compatibility
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400

    f = request.files['pdf']
    pdf_bytes = f.read()

    try:
        pil_img = render_first_page_to_image(pdf_bytes, dpi=RENDER_DPI)
    except Exception as e:
        return jsonify({"error": f"Failed to render PDF: {str(e)}"}), 400

    bgr = pil_to_cv2(pil_img)
    boxes = detect_design_boxes_bgr(bgr)  # ALL big boxes

    overlay_bgr = draw_debug_overlay(bgr, boxes)
    debug_name = f"debug_{_ts()}.png"
    if DEBUG_SAVE:
        cv2.imwrite(os.path.join(STATIC_DIR, debug_name), overlay_bgr)

    results: List[Dict] = []
    for i, (x, y, w, h) in enumerate(boxes):
        crop = bgr[y:y + h, x:x + w]
        crop_img = cv2_to_pil(crop)
        crop_name = f"cutout_{_ts()}_{i+1}.png"
        crop_img.save(os.path.join(STATIC_DIR, crop_name), format="PNG")
        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{crop_name}",
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "page": 1
        })

    response = {
        "cutouts": results,
        "debug_overlay_url": f"{request.host_url.rstrip('/')}/static/{debug_name}",
    }
    return jsonify(response), 200


# Kept for compatibility; you can ignore if not needed
@app.route("/extract-art-no-box", methods=["POST"])
def extract_art_no_box():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400
    f = request.files['pdf']
    pdf_bytes = f.read()
    try:
        pil_img = render_first_page_to_image(pdf_bytes, dpi=RENDER_DPI).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"Failed to render PDF: {str(e)}"}), 400

    bgr = pil_to_cv2(pil_img)
    boxes = detect_design_boxes_bgr(bgr)
    if not boxes:
        return jsonify({"error": "No design box found"}), 404

    results: List[Dict] = []
    for i, (x, y, w, h) in enumerate(boxes):
        out = pil_img.crop((x, y, x + w, y + h))
        out_name = f"art_box_{_ts()}_{i+1}.png"
        out.save(os.path.join(STATIC_DIR, out_name), format="PNG")
        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{out_name}",
            "x": int(x), "y": int(y), "width": int(w), "height": int(h), "page": 1
        })
    return jsonify({"art_no_box": results}), 200


# NEW: tight crop endpoint (primary one to use)
@app.route("/crop-art-tight", methods=["POST"])
def crop_art_tight():
    """
    Tight-crop the PDF so that the returned PNG bounds match the artwork bounds.
    Does NOT modify colors or remove background — only crops.

    Optional params (form or query):
      - deltaE_thresh (float, default 9.0) : Lab distance from bg to treat as foreground
      - min_area_frac (float, default 0.0005): min connected-component area to keep
      - border_strip_frac (float, default 0.02): ignore this border band inside the box
      - close_px (int, default 1): morphology to close small gaps (0..3)
      - pad_px (int, default 2): padding added after tight bbox crop
      - save_debug (0/1): if 1, saves the foreground mask used for the bbox
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400

    f = request.files['pdf']
    pdf_bytes = f.read()

    # Params
    deltaE_thresh = float(request.values.get("deltaE_thresh", 9.0))
    min_area_frac = float(request.values.get("min_area_frac", 0.0005))
    border_strip_frac = float(request.values.get("border_strip_frac", 0.02))
    close_px = int(request.values.get("close_px", 1))
    pad_px = int(request.values.get("pad_px", 2))
    save_debug = str(request.values.get("save_debug", "0")) == "1"

    try:
        pil_img = render_first_page_to_image(pdf_bytes, dpi=RENDER_DPI)
    except Exception as e:
        return jsonify({"error": f"Failed to render PDF: {str(e)}"}), 400

    bgr = pil_to_cv2(pil_img)
    boxes = detect_design_boxes_bgr(bgr)
    if not boxes:
        return jsonify({"error": "No design box found"}), 404

    results: List[Dict] = []

    for i, (x, y, w, h) in enumerate(boxes):
        crop_bgr = bgr[y:y+h, x:x+w].copy()

        # Foreground via ΔE from interior background
        x0r, y0r, x1r, y1r = _tight_bbox_by_deltaE(
            crop_bgr,
            deltaE_thresh=deltaE_thresh,
            min_area_frac=min_area_frac,
            border_strip_frac=border_strip_frac,
            close_px=close_px
        )

        # Add padding (clamped to crop bounds)
        x0 = max(0, x0r - pad_px); y0 = max(0, y0r - pad_px)
        x1 = min(w, x1r + pad_px); y1 = min(h, y1r + pad_px)

        # Final page-relative bbox
        art_x, art_y = int(x + x0), int(y + y0)
        art_w, art_h = int(x1 - x0), int(y1 - y0)

        # Save tightly cropped PNG from the ORIGINAL pixels (no color changes)
        out = cv2_to_pil(crop_bgr[y0:y1, x0:x1])
        out_name = f"art_tight_{_ts()}_{i+1}.png"
        out.save(os.path.join(STATIC_DIR, out_name), format="PNG")

        debug_urls = {}
        if DEBUG_SAVE && save_debug:
            # Rebuild the mask image for inspection
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            bg_lab = _dominant_bg_lab_from_interior(rgb, interior_frac=0.06)
            d = np.linalg.norm(lab - bg_lab, axis=2)
            mask = (d >= float(deltaE_thresh)).astype(np.uint8) * 255
            b = max(2, int(round(min(h, w) * border_strip_frac)))
            mask[:b, :] = 0; mask[-b:, :] = 0; mask[:, :b] = 0; mask[:, -b:] = 0
            if close_px > 0:
                k = np.ones((close_px, close_px), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            mname = f"dbg_tightmask_{_ts()}_{i+1}.png"
            cv2.imwrite(os.path.join(STATIC_DIR, mname), mask)
            debug_urls = {"mask_url": f"{request.host_url.rstrip('/')}/static/{mname}"}

        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{out_name}",
            "page": 1,
            "x": art_x, "y": art_y, "width": art_w, "height": art_h,
            "detected_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "params": {
                "deltaE_thresh": deltaE_thresh,
                "min_area_frac": min_area_frac,
                "border_strip_frac": border_strip_frac,
                "close_px": close_px,
                "pad_px": pad_px,
                "dpi": RENDER_DPI
            },
            **({"debug": debug_urls} if debug_urls else {})
        })

    return jsonify({"art_tight": results}), 200


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
