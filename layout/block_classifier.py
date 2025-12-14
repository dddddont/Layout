import cv2
import numpy as np


# ==========================================================
# 页眉 / 页脚
# ==========================================================
def is_header_footer(block, page_h, gray_img):
    x, y, w, h = block
    if h > 0.08 * page_h:
        return False

    if y < 0.08 * page_h or y + h > 0.92 * page_h:
        crop = gray_img[y:y+h, x:x+w]
        if crop.size == 0:
            return False
        return np.mean(crop) > 180

    return False


# ==========================================================
# ✅ 表格判断（结构线版本）
# ==========================================================
def is_table(block, binary_img):
    x, y, w, h = block
    crop = binary_img[y:y+h, x:x+w]
    if crop.size == 0:
        return False

    # 边缘
    edges = cv2.Canny(crop, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(w, h) * 0.35),
        maxLineGap=8
    )

    if lines is None:
        return False

    h_count = 0
    v_count = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # 横线：长 & 平
        if dy < 0.02 * h and dx > 0.4 * w:
            h_count += 1

        # 竖线：长 & 直
        elif dx < 0.02 * w and dy > 0.4 * h:
            v_count += 1

    # ===== 核心判定 =====
    if h_count < 3 or v_count < 3:
        return False

    # ===== 防止“竖向白缝”误判 =====
    # 表格线应是黑线（binary_img 中为 255）
    black_ratio = np.sum(crop > 0) / crop.size
    if black_ratio < 0.05:
        return False

    return True

# ==========================================================
# 文本
# ==========================================================
def is_text_like(block, binary_img):
    x, y, w, h = block
    crop = binary_img[y:y+h, x:x+w]   # ✅ 一定要最先定义
    if crop.size == 0:
        return False

    # ======================================================
    # Rule 0：反文本规则（自然图剔除）
    # ======================================================
    edges = cv2.Canny(crop, 60, 150)

    gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1)

    sum_x = np.sum(np.abs(gx))
    sum_y = np.sum(np.abs(gy))

    # 各向同性 → 更像自然图
    if sum_y > 0 and (sum_x / sum_y) < 1.3:
        return False

    # ======================================================
    # Rule 1：行结构（水平投影）
    # ======================================================
    proj = np.sum(crop > 0, axis=1) / (w + 1e-6)

    text_lines = np.logical_and(proj > 0.05, proj < 0.5)
    line_count = np.sum(text_lines)
    line_ratio = line_count / (h + 1e-6)

    strong_text = line_ratio > 0.15

    # ======================================================
    # Rule 2：行距稳定性
    # ======================================================
    stable_spacing = False
    idx = np.where(text_lines)[0]
    if len(idx) >= 3:
        gaps = np.diff(idx)
        if np.std(gaps) < 3.0:
            stable_spacing = True

    # ======================================================
    # Rule 3：黑白比例
    # ======================================================
    black_ratio = np.sum(crop > 0) / crop.size
    if black_ratio > 0.45:
        return False

    reasonable_density = 0.03 < black_ratio < 0.35

    # ======================================================
    # 最终判定
    # ======================================================
    if strong_text and reasonable_density:
        return True

    if stable_spacing and reasonable_density:
        return True

    return False






# ==========================================================
# 图像 / Figure
# ==========================================================
def is_figure(block, gray, binary):
    x, y, w, h = block
    crop_g = gray[y:y+h, x:x+w]
    crop_b = binary[y:y+h, x:x+w]

    if crop_g.size == 0:
        return False

    # ======================================================
    # 0️⃣ 先排除明显文本（行结构太稳定）
    # ======================================================
    proj = np.sum(crop_b > 0, axis=1)
    text_like_rows = np.sum(proj > 0.15 * w)

    # if text_like_rows > 0.2 * h:
    #     return False   # 太像文本，直接否掉

    # ======================================================
    # 🆕 0.5️⃣ 行高差距异常 → figure
    # ======================================================
    # 找文本行
    text_rows = proj > 0.05 * w

    line_heights = []
    in_line = False
    start = 0

    for i, v in enumerate(text_rows):
        if v and not in_line:
            in_line = True
            start = i
        elif not v and in_line:
            end = i
            in_line = False
            line_heights.append(end - start)

    if in_line:
        line_heights.append(h - start)

    # 至少要有多行才判断“行高差距”
    if len(line_heights) >= 3:
        median_h = np.median(line_heights)
        if median_h > 0:
            if (max(line_heights) - min(line_heights)) > 1 * median_h:
                return True   # 行高变化远大于文本 → 图片



    if text_like_rows > 0.2 * h:
        return False  # 太像文本，直接否掉
    # ======================================================
    # 1️⃣ 子图并排白缝规则
    # ======================================================
    col_white_ratio = np.mean(crop_b == 0, axis=0)
    white_columns = col_white_ratio > 0.95

    gaps = []
    cnt = 0
    for v in white_columns:
        if v:
            cnt += 1
        else:
            if cnt > 0:
                gaps.append(cnt)
                cnt = 0
    if cnt > 0:
        gaps.append(cnt)

    valid_gaps = [g for g in gaps if 3 <= g <= 0.05 * w]

    if len(valid_gaps) >= 2:
        return True   # 多子图 figure

    # ======================================================
    # 2️⃣ 单图兜底规则（OR）
    # ======================================================
    white_ratio = np.mean(crop_b == 0)

    edges = cv2.Canny(crop_g, 60, 150)
    gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1)

    grad_balance = np.sum(np.abs(gx)) / (np.sum(np.abs(gy)) + 1e-6)
    edge_density = np.sum(edges > 0) / edges.size

    if (
        white_ratio > 0.45 and edge_density > 0.02
    ) or (
        0.6 < grad_balance < 1.6 and edge_density > 0.04
    ):
        return True

    return False


def is_column_left_aligned(block, blocks, tol=15):
    x, _, _, _ = block
    col_left = min(b[0] for b in blocks)
    return abs(x - col_left) < tol
#
# def is_title_like(block, text_start_x, page_h):
#     x, y, w, h = block
#
#     aligned = abs(x - text_start_x) < 10
#     tall = h > 1.2 * median_text_height
#     not_footer = y > 0.15 * page_h
#
#     return aligned and tall and not_footer


# ==========================================================
# 主接口
# ==========================================================
def classify_block(block, img, binary_img, page_h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_header_footer(block, page_h, gray):
        return "header_footer"

    if is_table(block, binary_img):
        return "table"

    if is_text_like(block, binary_img):
        return "text"

    # if is_title_like(block, gray, binary_img):
    #     return "text"  # 标题也归为 text
    if is_figure(block, gray, binary_img):
        return "figure"




    return "text"
