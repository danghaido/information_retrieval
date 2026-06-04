"""
RAG Demo - FastAPI với FAISS Vector Store
Endpoint 1: POST /upload  - nhận tài liệu, chunk, đưa vào FAISS
Endpoint 2: POST /ask     - nhận câu hỏi, tìm kiếm, gọi LLM, trả lời
"""

import os
import re
import uuid
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "bkai-foundation-models/vietnamese-bi-encoder")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.50.218:8000/api/v1/proxy")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STUDENT_ID = os.getenv("STUDENT_ID", "B22DCAT082")

CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
TOP_K = 10

# ── Load local embedding model (lazy, loaded once) ───────────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
_embed_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=MODEL_CACHE_DIR)
logger.success(f"Embedding model ready (dim={_embed_model.get_embedding_dimension()})")


# ── In-memory Vector Store ───────────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []
        self.ids: list[str] = []

    def _init_index(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, chunk_id: str, text: str, embedding: list[float]):
        vec = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vec)  # cosine similarity
        if self.index is None:
            self._init_index(vec.shape[1])
        self.index.add(vec)
        self.chunks.append(text)
        self.ids.append(chunk_id)

    def search(self, embedding: list[float], k: int = TOP_K) -> list[tuple[str, str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vec)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)
        return [(self.ids[i], self.chunks[i], float(scores[0][rank])) for rank, i in enumerate(indices[0]) if i != -1]


store = VectorStore()


# ── Pydantic Schemas ─────────────────────────────────────────────────────────
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []


# ── Helpers ──────────────────────────────────────────────────────────────────
def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current and current_len + len(sent) > size:
            chunks.append(" ".join(current))
            # giữ lại các câu cuối làm overlap cho chunk tiếp theo
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail, tail_len
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    logger.debug(f"Embedding {len(texts)} đoạn văn bản")
    vecs = _embed_model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


_VALID_LETTERS = {"A", "B", "C", "D"}


def extract_letter(raw: str) -> str:
    if not raw:
        return "A"
    s = raw.strip().upper()
    if s[0] in _VALID_LETTERS:
        return s[0]
    m = re.search(r"(?:ĐÁP\s*ÁN|ANSWER|CHỌN)\s*[:\-]?\s*([ABCD])", s)
    if m:
        return m.group(1)
    m = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", s)
    if m:
        return m.group(1)
    return "A"


def call_llm(question: str, context: str) -> str:
    logger.debug(f"Gọi LLM [{LLM_MODEL}] | câu hỏi: {question!r}")
    system_prompt = (
        "Bạn là trợ lý trả lời trắc nghiệm. "
        "Dựa CHỈ vào tài liệu được cung cấp để chọn đáp án đúng. "
        "BẮT BUỘC chỉ trả lời bằng MỘT ký tự duy nhất: A, B, C hoặc D. "
        "Không giải thích, không viết gì thêm."
    )
    user_prompt = f"Tài liệu tham khảo:\n{context}\n\nCâu hỏi trắc nghiệm:\n{question}\n\nĐáp án (chỉ 1 ký tự A/B/C/D):"
    client = OpenAI(base_url=LLM_BASE_URL, api_key=STUDENT_ID)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    answer = extract_letter(raw)
    logger.debug(f"LLM raw={raw!r} → answer={answer}")
    return answer


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Demo", version="1.0.0")


@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    """Nhận JSON {doc_id?, text}, chunk, embed, lưu vào FAISS."""
    logger.info(f"Upload request | doc_id='{req.doc_id}'")

    if not req.text.strip():
        logger.warning("text rỗng, bỏ qua")
        raise HTTPException(status_code=400, detail="text không được để trống")

    doc_id = req.doc_id or f"doc_{uuid.uuid4().hex[:8]}"
    chunks = split_chunks(req.text)
    logger.info(f"Tách thành {len(chunks)} chunk | doc_id='{doc_id}'")

    try:
        vectors = embed(chunks)
    except Exception as e:
        logger.error(f"Embedding thất bại: {e}")
        raise HTTPException(status_code=502, detail=f"Embedding API lỗi: {e}")

    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        store.add(f"{doc_id}_chunk_{idx}", chunk, vec)

    logger.success(f"Indexed xong '{doc_id}' | {len(chunks)} chunk | tổng FAISS: {store.index.ntotal}")
    return UploadResponse(status="indexed", doc_id=doc_id, chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Nhận câu hỏi, tìm chunk liên quan, gọi LLM, trả lời."""
    logger.info(f"Ask request | question: {req.question!r}")

    if not req.question.strip():
        logger.warning("Câu hỏi rỗng, bỏ qua")
        raise HTTPException(status_code=400, detail="question không được để trống")

    if store.index is None or store.index.ntotal == 0:
        logger.warning("Chưa có tài liệu trong FAISS")
        raise HTTPException(status_code=404, detail="Chưa có tài liệu nào được upload")

    try:
        q_vec = embed([req.question])[0]
    except Exception as e:
        logger.error(f"Embedding câu hỏi thất bại: {e}")
        raise HTTPException(status_code=502, detail=f"Embedding API lỗi: {e}")

    results = store.search(q_vec, k=TOP_K)
    if not results:
        logger.warning("Không tìm thấy chunk liên quan")
        return AskResponse(answer="A", sources=[])

    sources = [chunk_id for chunk_id, _, _ in results]
    logger.info(f"Tìm thấy {len(results)} chunk | sources: {sources} | scores: {[round(s, 3) for _, _, s in results]}")
    context = "\n\n---\n\n".join(text for _, text, _ in results)

    try:
        answer = call_llm(req.question, context)
    except Exception as e:
        logger.error(f"LLM thất bại: {e}")
        raise HTTPException(status_code=502, detail=f"LLM API lỗi: {e}")

    logger.success(f"Trả lời xong | {len(answer)} ký tự")
    return AskResponse(answer=answer, sources=sources)


@app.get("/")
def root():
    total_chunks = store.index.ntotal if store.index else 0
    return {
        "message": "RAG Demo đang chạy",
        "total_chunks_indexed": total_chunks,
        "endpoints": {"upload": "POST /upload", "ask": "POST /ask", "docs": "/docs"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_demo:app", host="0.0.0.0", port=5000)
