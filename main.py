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

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def pil_to_cv2(im: Image.Image) -> np.ndarray:
    arr = np.array(im)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
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


# ---- Detect the big demo box ----

def detect_design_boxes_bgr(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


# ---- Tight-crop logic (NO color changes) ----

def _build_mask_bgdiff_and_edges(gray: np.ndarray,
                                 ignore_border_px: int,
                                 min_area: int,
                                 dilate_px: int = 1) -> np.ndarray:
    """
    Foreground = pixels whose brightness differs from the interior background,
    OR strong edges. Returns uint8 mask (255 = foreground).
    """
    H, W = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Estimate background from interior (avoid the frame)
    b = max(2, ignore_border_px)
    inner = blur[b:H - b, b:W - b]
    if inner.size == 0:
        inner = blur

    bg_med = np.median(inner)
    # Robust noise estimate via MAD
    mad = np.median(np.abs(inner - np.median(inner)))
    sigma = 1.4826 * mad
    # Threshold: at least 8 levels away from bg, or 0.6*sigma if larger
    t = max(8.0, 0.6 * float(sigma))

    diff = np.abs(blur.astype(np.float32) - float(bg_med))
    fg1 = (diff >= t).astype(np.uint8) * 255

    # Edges (dynamic thresholds), dilated a bit to thicken
    v = float(np.median(inner))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    mask = cv2.bitwise_or(fg1, edges)

    # Strip inner band so the frame can't leak in
    mask[:b, :] = 0
    mask[-b:, :] = 0
    mask[:, :b] = 0
    mask[:, -b:] = 0

    # Connect gaps and drop tiny blobs
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    keep = np.zeros_like(mask)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            keep[labels == i] = 255

    return keep


def _tight_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    H, W = mask.shape[:2]
    if np.count_nonzero(mask) == 0:
        return 0, 0, W, H
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


# ---- Routes ----

@app.route("/")
def root():
    return jsonify({"status": "ok", "message": "PDF cutout box extractor is running"})


@app.route("/detect-art-format", methods=["POST"])
def detect_art_format():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400

    f = request.files['pdf']
    pdf_bytes = f.read()

    try:
        pil_img = render_first_page_to_image(pdf_bytes, dpi=RENDER_DPI)
    except Exception as e:
        return jsonify({"error": f"Failed to render PDF: {str(e)}"}), 400

    bgr = pil_to_cv2(pil_img)
    boxes = detect_design_boxes_bgr(bgr)

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
            "x": int(x), "y": int(y), "width": int(w), "height": int(h), "page": 1
        })

    return jsonify({
        "cutouts": results,
        "debug_overlay_url": f"{request.host_url.rstrip('/')}/static/{debug_name}",
    }), 200


# Simple pass-through (kept for compatibility)
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


# NEW: tight crop endpoint (primary)
@app.route("/crop-art-tight", methods=["POST"])
def crop_art_tight():
    """
    Tight-crop to the minimal rectangle enclosing all visible artwork inside the big demo box.
    NO color edits; only cropping.

    Optional params (form or query):
      - border_strip_frac (float, default 0.02): inner band to ignore (avoid counting the frame)
      - min_area_frac (float, default 0.0003): drop tiny components below this fraction of crop area
      - dilate_px (int, default 1): connect small gaps in the mask
      - pad_px (int, default 2): padding added to final crop
      - save_debug (0/1): if 1, saves the mask used
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400

    f = request.files['pdf']
    pdf_bytes = f.read()

    border_strip_frac = float(request.values.get("border_strip_frac", 0.02))
    min_area_frac = float(request.values.get("min_area_frac", 0.0003))
    dilate_px = int(request.values.get("dilate_px", 1))
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
        crop = bgr[y:y + h, x:x + w].copy()
        Hc, Wc = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        ignore_px = max(2, int(round(min(Hc, Wc) * border_strip_frac)))
        min_area = int(round(min_area_frac * Hc * Wc))

        mask = _build_mask_bgdiff_and_edges(
            gray,
            ignore_border_px=ignore_px,
            min_area=min_area,
            dilate_px=dilate_px
        )

        x0r, y0r, x1r, y1r = _tight_bbox_from_mask(mask)

        # Add padding within bounds
        x0 = max(0, x0r - pad_px)
        y0 = max(0, y0r - pad_px)
        x1 = min(Wc, x1r + pad_px)
        y1 = min(Hc, y1r + pad_px)

        # Save tightly cropped ORIGINAL pixels
        tight = cv2_to_pil(crop[y0:y1, x0:x1])
        out_name = f"art_tight_{_ts()}_{i+1}.png"
        tight.save(os.path.join(STATIC_DIR, out_name), format="PNG")

        debug_urls = {}
        if DEBUG_SAVE and save_debug:
            mname = f"dbg_mask_{_ts()}_{i+1}.png"
            cv2.imwrite(os.path.join(STATIC_DIR, mname), mask)
            debug_urls = {"mask_url": f"{request.host_url.rstrip('/')}/static/{mname}"}

        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{out_name}",
            "page": 1,
            "x": int(x + x0), "y": int(y + y0),
            "width": int(x1 - x0), "height": int(y1 - y0),
            "detected_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "params": {
                "border_strip_frac": border_strip_frac,
                "min_area_frac": min_area_frac,
                "dilate_px": dilate_px,
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
