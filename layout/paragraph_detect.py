import cv2
import numpy as np

# ==========================================================
# 行高估计
# ==========================================================
def estimate_line_height(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 15 < h < 120 and w > 30:
            heights.append(h)
    return int(np.median(heights)) if heights else 25


# ==========================================================
# 自适应膨胀（仅 proposal）
# ==========================================================
def proposal_dilation(binary, line_h):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(line_h * 2.5), int(line_h * 1.2))
    )
    return cv2.dilate(binary, kernel, iterations=1)


# ==========================================================
# 段落合并（几何）
# ==========================================================
def is_title_block(block, line_h):
    _, _, _, h = block
    return h > 1.6 * line_h


def merge_paragraph_blocks(blocks, line_h):
    merged = []
    blocks = sorted(blocks, key=lambda b: b[1])

    for b in blocks:
        if not merged:
            merged.append(b)
            continue

        x, y, w, h = b
        px, py, pw, ph = merged[-1]

        v_gap = y - (py + ph)
        x_overlap = min(x + w, px + pw) - max(x, px)

        if (
            x_overlap > 0.7 * min(w, pw)
            and v_gap < 1.8 * line_h
            and not is_title_block(b, line_h)
            and not is_title_block(merged[-1], line_h)
        ):
            nx = min(x, px)
            ny = py
            nw = max(x + w, px + pw) - nx
            nh = (y + h) - py
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(b)

    return merged


# ==========================================================
# 主接口：detect_paragraphs
# ==========================================================
def detect_paragraphs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    line_h = estimate_line_height(binary)
    rough = proposal_dilation(binary, line_h)

    cnts, _ = cv2.findContours(
        rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 4000:
            continue
        blocks.append((x, y, w, h))

    return merge_paragraph_blocks(blocks, line_h)
