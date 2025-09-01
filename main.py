import os
import io
import datetime
from typing import List, Tuple, Dict, Optional

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
    """PIL RGB -> OpenCV BGR (keeps 3 channels)"""
    arr = np.array(im)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB"""
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


# ---- Detection (for cutout box) ----

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


# ---- Helpers for background removal and trimming ----

def inches_to_px(inches: float, dpi: int) -> int:
    return int(round(inches * dpi))


def _kmeans_centers_lab(rgb: np.ndarray, k: int = 3, sample_step: int = 4) -> np.ndarray:
    """
    Fast color clustering using OpenCV kmeans (no sklearn).
    Returns cluster centers in LAB (float32, shape (k,3)).
    """
    h, w, _ = rgb.shape
    # Subsample to speed up
    sampled = rgb[::sample_step, ::sample_step, :].reshape(-1, 3).astype(np.float32)
    if sampled.shape[0] < k:
        k = max(1, sampled.shape[0])
    # Convert to LAB for perceptual distance
    sampled_lab = cv2.cvtColor(sampled.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    compactness, labels, centers = cv2.kmeans(sampled_lab, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)


def _label_by_nearest_center_lab(lab_img: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    """
    Assign each pixel the index of the nearest LAB center. Returns label image (H,W).
    """
    # lab_img: HxWx3 float32
    H, W, _ = lab_img.shape
    C = centers_lab.shape[0]
    # Compute squared distance to each center efficiently
    lab_ = lab_img.reshape(-1, 3)[:, None, :]  # (N,1,3)
    centers_ = centers_lab[None, :, :]         # (1,C,3)
    d2 = np.sum((lab_ - centers_) ** 2, axis=2)  # (N,C)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    return labels.reshape(H, W)


def _find_background_centers_from_border(labels: np.ndarray, centers_lab: np.ndarray, border_frac_thresh: float = 0.15) -> List[int]:
    """
    Identify which clusters are background by checking which labels dominate on the border.
    """
    H, W = labels.shape
    bw = max(2, min(H, W) // 100)  # thin border strip ~1%
    border_mask = np.zeros_like(labels, dtype=bool)
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    border_labels = labels[border_mask]
    bg_labels, counts = np.unique(border_labels, return_counts=True)

    # Choose any cluster that covers >= border_frac_thresh of border pixels
    total = border_labels.size
    bg = [int(l) for l, c in zip(bg_labels, counts) if c / float(total) >= border_frac_thresh]
    # Fallback: at least take the most common border label
    if not bg and bg_labels.size > 0:
        bg = [int(bg_labels[np.argmax(counts)])]
    return bg


def _keep_only_border_connected(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the components of 'mask' that touch the image border.
    """
    H, W = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if num <= 1:
        return mask.astype(bool)

    border_touching = set()
    # Any component ID that appears on the border?
    border_ids = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) | \
                 set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_touching |= border_ids

    keep = np.isin(labels, list(border_touching))
    return keep


def remove_background_box_from_crop_rgba(
    crop_rgba: Image.Image,
    lab_tolerance: float = 12.0,
    k_colors: int = 3,
    dilate_px: int = 1
) -> Tuple[Image.Image, np.ndarray]:
    """
    Remove the large background (demo box + any page white corners) from a crop.
    - Uses color clustering in LAB, labels the whole image by nearest centers.
    - Marks clusters that dominate the border as background.
    - Optionally dilates the bg mask to clean anti-aliased edges.
    Returns (clean_rgba_image, bg_mask_bool_array)
    """
    if crop_rgba.mode != "RGBA":
        crop_rgba = crop_rgba.convert("RGBA")
    arr = np.array(crop_rgba)
    rgb = arr[:, :, :3]
    H, W = rgb.shape[:2]

    # Full LAB image (float32)
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    # Cluster colors (centers in LAB)
    centers_lab = _kmeans_centers_lab(rgb, k=k_colors, sample_step=max(1, min(H, W) // 400 or 1))

    # Label each pixel by nearest center
    labels = _label_by_nearest_center_lab(lab, centers_lab)

    # Background centers determined from border dominance
    bg_centers = _find_background_centers_from_border(labels, centers_lab, border_frac_thresh=0.10)

    # Build background candidate mask by LAB distance to each bg center
    bg_mask = np.zeros((H, W), dtype=bool)
    for idx in bg_centers:
        center = centers_lab[idx][None, None, :]  # (1,1,3)
        d2 = np.sum((lab - center) ** 2, axis=2)
        bg_mask |= (np.sqrt(d2) <= lab_tolerance)

    # Keep only border-connected bg (avoid nuking internal shapes similar to bg color)
    bg_mask = _keep_only_border_connected(bg_mask)

    # Slightly grow mask to swallow anti-aliased edges
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        bg_mask = cv2.dilate(bg_mask.astype(np.uint8), kernel, 1).astype(bool)

    # Apply transparency
    out = arr.copy()
    out[bg_mask, 3] = 0

    return Image.fromarray(out, mode="RGBA"), bg_mask


def trim_transparent_margin(
    rgba_img: Image.Image,
    min_alpha: int = 8,
    pad_px: int = 4
) -> Tuple[Image.Image, int, int, int, int]:
    """
    Crops away near-transparent margins.
    Returns (trimmed_image, x0, y0, x1, y1) where x0,y0.. are coordinates within the input image.
    """
    if rgba_img.mode != "RGBA":
        rgba_img = rgba_img.convert("RGBA")
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]
    content = alpha > min_alpha
    if not np.any(content):
        H, W = alpha.shape
        return rgba_img, 0, 0, W, H

    ys, xs = np.where(content)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # padding
    y0 = max(0, y0 - pad_px)
    x0 = max(0, x0 - pad_px)
    y1 = min(alpha.shape[0], y1 + pad_px)
    x1 = min(alpha.shape[1], x1 + pad_px)

    arr = arr[y0:y1, x0:x1, :]
    return Image.fromarray(arr, mode="RGBA"), x0, y0, x1, y1


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


@app.route("/extract-art-no-box", methods=["POST"])
def extract_art_no_box():
    """
    Produces transparent PNG(s) of the art with the large background removed,
    and trims away empty margins.

    Query/body params you can pass (optional):
      - k_colors (int, default 3): number of color clusters
      - lab_tolerance (float, default 12.0): LAB distance threshold for bg
      - dilate_px (int, default 1): grow bg mask to avoid halos
      - pad_px (int, default 4): pixels of padding after trim
      - pad_in (float, default None): inches of padding after trim (overrides pad_px if set)
      - min_alpha (int, default 8): alpha threshold for "empty"
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part 'pdf' in request"}), 400

    f = request.files['pdf']
    pdf_bytes = f.read()

    # Parse options
    k_colors = int(request.values.get("k_colors", 3))
    lab_tolerance = float(request.values.get("lab_tolerance", 12.0))
    dilate_px = int(request.values.get("dilate_px", 1))
    pad_px = int(request.values.get("pad_px", 4))
    pad_in = request.values.get("pad_in", None)
    min_alpha = int(request.values.get("min_alpha", 8))
    if pad_in is not None:
        try:
            pad_px = max(pad_px, inches_to_px(float(pad_in), RENDER_DPI))
        except Exception:
            pass

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
        # 1) crop the detected box
        crop_rgba = pil_img.crop((x, y, x + w, y + h))

        # 2) remove background (color-agnostic) and clean edges
        cleaned_rgba, bg_mask = remove_background_box_from_crop_rgba(
            crop_rgba,
            lab_tolerance=lab_tolerance,
            k_colors=k_colors,
            dilate_px=dilate_px
        )

        # 3) trim transparent margins with padding
        trimmed_rgba, x0_rel, y0_rel, x1_rel, y1_rel = trim_transparent_margin(
            cleaned_rgba, min_alpha=min_alpha, pad_px=pad_px
        )

        # Page-relative bbox of trimmed art
        art_x = int(x + x0_rel)
        art_y = int(y + y0_rel)
        art_w = int(x1_rel - x0_rel)
        art_h = int(y1_rel - y0_rel)

        # 4) Save
        out_name = f"art_no_box_{_ts()}_{i+1}.png"
        out_path = os.path.join(STATIC_DIR, out_name)
        trimmed_rgba.save(out_path, format="PNG")

        results.append({
            "url": f"{request.host_url.rstrip('/')}/static/{out_name}",
            "page": 1,
            # Page-relative bounding box of the returned image
            "x": art_x,
            "y": art_y,
            "width": art_w,
            "height": art_h,
            # Extra info (optional; useful for debugging)
            "detected_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "params": {
                "k_colors": k_colors,
                "lab_tolerance": lab_tolerance,
                "dilate_px": dilate_px,
                "pad_px": pad_px,
                "min_alpha": min_alpha,
                "dpi": RENDER_DPI
            }
        })

    response = {"art_no_box": results}
    return jsonify(response), 200


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
