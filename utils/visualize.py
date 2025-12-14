import cv2
import numpy as np

def draw_blocks_with_id(img, blocks, color=(0, 0, 255)):
    """
    blocks: list of dict
        {
            "block_id": "1.3",
            "bbox": [x, y, w, h]
        }
    """
    out = img.copy()

    for b in blocks:
        x, y, w, h = b["bbox"]
        bid = b["block_id"]

        cv2.rectangle(
            out,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            color,
            2
        )

        cv2.putText(
            out,
            bid,
            (int(x), max(15, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return out

def draw_text_line_heights(vis_img, binary_img, bbox):
    """
    在 block 内部，用黑线画出每一行文本的上下边界
    """
    x, y, w, h = bbox
    crop = binary_img[y:y+h, x:x+w]
    if crop.size == 0:
        return vis_img

    # 水平投影：每一行的黑像素数
    proj = np.sum(crop > 0, axis=1)

    # 判定“像文本的行”
    text_rows = proj > 0.05 * w

    # 找连续区间
    in_line = False
    start = 0
    for i, v in enumerate(text_rows):
        if v and not in_line:
            in_line = True
            start = i
        elif not v and in_line:
            end = i
            in_line = False

            # 画上下边界
            cv2.line(vis_img, (x, y + start), (x + w, y + start), (0, 0, 0), 1)
            cv2.line(vis_img, (x, y + end),   (x + w, y + end),   (0, 0, 0), 1)

    # 处理最后一行
    if in_line:
        end = h
        cv2.line(vis_img, (x, y + start), (x + w, y + start), (0, 0, 0), 1)
        cv2.line(vis_img, (x, y + end),   (x + w, y + end),   (0, 0, 0), 1)

    return vis_img
