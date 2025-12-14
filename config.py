import os

# ================== 路径配置 ==================
POPPLER_PATH = r"D:\poppler\Library\bin"
PDF_PATH = r"D:\109\datapirture\Yin_Side_Window_Filtering_CVPR_2019_paper.pdf"

OUTPUT_DIR = r"D:\109\datapirture\red_he_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR / 模型路径（以后用）
TESS_PATH = r"D:\Tesseract-OCR\tesseract.exe"
LAYOUTLM_MODEL_PATH = r"D:\109\datapirture\layoutlmv3-base"

# OUTPUT_DIR = r"D:\109\datapirture\doc_layout_output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# ========= PDF 渲染 =========
DPI = 300
