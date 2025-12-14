import os
import cv2
import json
import numpy as np
from collections import defaultdict
from pdf2image import convert_from_path

from config import POPPLER_PATH, PDF_PATH, OUTPUT_DIR
from layout.paragraph_detect import detect_paragraphs
from layout.block_classifier import classify_block
from utils.visualize import draw_blocks_with_id
from utils.visualize import draw_text_line_heights


# ==========================================================
# 页眉 / 页码统计（重复位置 + 浅色）
# ==========================================================
def block_mean_intensity(gray, bbox):
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 255.0
    return float(np.mean(roi))


def collect_position_stats(all_page_blocks, page_h):
    counter = defaultdict(int)
    for blocks in all_page_blocks:
        for (_, y, _, h) in blocks:
            yc = round((y + h / 2) / page_h, 2)
            counter[yc] += 1
    return counter


def is_header_footer_block(
    bbox, gray, page_h,
    pos_counter, page_count,
    intensity_thresh=210,
    repeat_ratio=0.8
):
    if block_mean_intensity(gray, bbox) < intensity_thresh:
        return False

    x, y, w, h = bbox
    yc = round((y + h / 2) / page_h, 2)

    if pos_counter[yc] < page_count * repeat_ratio:
        return False

    return y < 0.15 * page_h or y + h > 0.85 * page_h


# ==========================================================
# 阅读顺序排序（左→右，上→下）
# ==========================================================
def sort_blocks_reading_order(blocks):
    return sorted(blocks, key=lambda b: (b[1], b[0]))


# ==========================================================
# 画横线 / 竖线（所有 block 都画，用于 debug）
#   横线：浅蓝
#   竖线：深绿
# ==========================================================
def draw_lines_for_block(vis_img, binary_img, bbox):
    x, y, w, h = bbox
    crop = binary_img[y:y+h, x:x+w]
    if crop.size == 0:
        return vis_img

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(30, w // 8), 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(30, h // 8))
    )

    horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN, h_kernel)
    vertical   = cv2.morphologyEx(crop, cv2.MORPH_OPEN, v_kernel)

    ys, xs = np.where(horizontal > 0)
    vis_img[y+ys, x+xs] = (255,200,100)  # 🩵 横线

    ys, xs = np.where(vertical > 0)
    vis_img[y+ys, x+xs] = (0,150,0)      # 🟩 竖线

    return vis_img




# ==========================================================
# 主流程
# ==========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pages = convert_from_path(
        PDF_PATH, dpi=300, poppler_path=POPPLER_PATH
    )

    all_page_blocks = []
    all_page_imgs = []

    # ---------- 第一遍：分块 ----------
    for pil_img in pages:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blocks = sort_blocks_reading_order(
            detect_paragraphs(img)
        )

        all_page_blocks.append(blocks)
        all_page_imgs.append((img, gray))

    page_h = all_page_imgs[0][0].shape[0]
    pos_counter = collect_position_stats(all_page_blocks, page_h)
    page_count = len(all_page_blocks)

    all_results = []

    # ---------- 第二遍：分类 + 可视化 ----------
    for page_idx, blocks in enumerate(all_page_blocks):
        img, gray = all_page_imgs[page_idx]

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        vis_img = img.copy()
        page_blocks_out = []

        for i, bbox in enumerate(blocks, start=1):
            block_id = f"{page_idx + 1}.{i}"

            if is_header_footer_block(
                bbox, gray, page_h,
                pos_counter, page_count
            ):
                block_type = "header_footer"
            else:
                block_type = classify_block(
                    bbox, img, binary, page_h
                )

            # 防御式兜底（理论上 classify_block 已经返回）
            if block_type is None:
                block_type = "none"

            page_blocks_out.append({
                "block_id": block_id,
                "bbox": bbox,
                "type": block_type
            })

            # ---------- 画 block 框 ----------
            color_map = {
                "text": (0, 0, 255),          # 🔴
                "table": (255, 0, 0),         # 🔵
                "header_footer": (180,180,180), # ⚪
                "figure": (255,255,0),        # 🟡
                # "none": (160, 0, 160)         # 🟣
            }

            vis_img = draw_blocks_with_id(
                vis_img,
                [{
                    "block_id": block_id,
                    "bbox": bbox
                }],
                color=color_map[block_type]
            )

            # if block_type in ("text", "none"):
            #     # none 也画，方便你 debug 为什么不是 text
            #     vis_img = draw_text_line_heights(
            #         vis_img,
            #         binary,
            #         bbox
            #     )
        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"page_{page_idx + 1:02d}.png"
            ),
            vis_img
        )

        all_results.append({
            "page": page_idx + 1,
            "blocks": page_blocks_out
        })

    with open(
        os.path.join(OUTPUT_DIR, "blocks.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            all_results, f,
            indent=2, ensure_ascii=False
        )

    print("完成 ✅")



if __name__ == "__main__":
    main()