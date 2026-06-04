"""
Tải embedding model về local 1 lần trước khi thi.

Chạy script này TRƯỚC khi vào LAN thi (lúc còn có Internet):
    python download_model.py

Model sẽ được lưu vào ./models/ — đi kèm project, không phụ thuộc ~/.cache.
"""
import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "keepitreal/vietnamese-sbert")
MODEL_CACHE_DIR  = os.getenv("MODEL_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
print(f"[download] model = {EMBED_MODEL_NAME}")
print(f"[download] cache = {MODEL_CACHE_DIR}")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

# Sanity check: encode thử 1 câu để chắc rằng tất cả file cần thiết đã có
vec = model.encode(["Đây là câu kiểm tra tiếng Việt."])
print(f"[download] OK — embedding dim = {len(vec[0])}")
print("[download] Done. Có thể chạy offline với OFFLINE=1.")
