"""
CLI gọi 4 endpoint của Teacher Server: register, evaluate, reset, result.

Cách dùng:
    python trigger.py register              # tự detect IP LAN, port 5000
    python trigger.py register --ip 192.168.50.87 --port 5000
    python trigger.py evaluate              # bắt đầu thi
    python trigger.py result                # xem điểm hiện tại
    python trigger.py reset                 # reset trạng thái
    python trigger.py all                   # register -> evaluate -> result
"""
import argparse
import json
import os
import socket
import sys
from typing import Optional
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv

load_dotenv()

STUDENT_ID       = os.getenv("STUDENT_ID",       "B22DCAT082").upper()
TEACHER_BASE_URL = os.getenv("TEACHER_BASE_URL", "http://10.170.45.200:8000/api/v1")

HEADERS = {"X-Student-ID": STUDENT_ID}

# Vị trí file vector DB mà rag_demo.py persist xuống (để tự suy ra document_received)
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.getenv("VECTOR_DIR", os.path.join(_BASE_DIR, "vector_store"))
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH  = os.path.join(VECTOR_DIR, "meta.json")


def has_vector_db() -> bool:
    """True nếu vector DB đã được build & persist trên đĩa."""
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def get_lan_ip() -> str:
    """Lấy IP LAN của máy bằng cách 'giả vờ' connect tới Teacher Server.
    Không gửi packet thật — chỉ để OS chọn network interface đúng."""
    parsed = urlparse(TEACHER_BASE_URL)
    teacher_host = parsed.hostname or "192.168.50.218"
    teacher_port = parsed.port or 8000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((teacher_host, teacher_port))
        return s.getsockname()[0]
    finally:
        s.close()


def pretty(resp: httpx.Response) -> None:
    print(f"[{resp.status_code}] {resp.request.method} {resp.request.url}")
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)


def register(ip: str | None, port: int) -> None:
    if not ip:
        ip = get_lan_ip()
        print(f"[auto] detected LAN IP = {ip}")

    server_url = f"http://{ip}:{port}"
    print(f"[register] student={STUDENT_ID}  server_url={server_url}")

    resp = httpx.post(
        f"{TEACHER_BASE_URL}/competition/register",
        headers=HEADERS,
        json={"server_url": server_url},
        timeout=10,
    )
    pretty(resp)


def evaluate(document_received: Optional[bool] = None) -> None:
    # None = tự suy ra từ việc đã có vector DB persisted hay chưa
    if document_received is None:
        document_received = has_vector_db()

    print(f"[evaluate] student={STUDENT_ID} — bắt đầu thi...")
    print(f"[evaluate] document_received={document_received} "
          f"({'đã có vector DB, Teacher chỉ gửi câu hỏi' if document_received else 'chưa có, Teacher sẽ gửi document để upload trước'})")
    if document_received:
        print("(Teacher sẽ gọi thẳng 10 lần /ask (60s mỗi câu) — bỏ qua /upload)")
    else:
        print("(Teacher sẽ gọi /upload (120s) rồi 10 lần /ask (60s mỗi câu) tới server của bạn)")
    resp = httpx.post(
        f"{TEACHER_BASE_URL}/competition/evaluate",
        headers=HEADERS,
        json={"document_received": document_received},
        timeout=60 * 15,
    )
    pretty(resp)


def reset() -> None:
    print(f"[reset] student={STUDENT_ID}")
    resp = httpx.post(
        f"{TEACHER_BASE_URL}/competition/reset",
        headers=HEADERS,
        timeout=10,
    )
    pretty(resp)


def result() -> None:
    print(f"[result] student={STUDENT_ID}")
    resp = httpx.get(
        f"{TEACHER_BASE_URL}/competition/result",
        headers=HEADERS,
        timeout=10,
    )
    pretty(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger Teacher Server endpoints.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Đăng ký Student Server URL")
    p_reg.add_argument("--ip", default=None, help="IP LAN máy bạn (mặc định: auto-detect)")
    p_reg.add_argument("--port", type=int, default=5000, help="Port server (mặc định: 5000)")

    DOC_CHOICES = ("auto", "true", "false")
    doc_help = "document_received gửi cho Teacher: auto=tự suy từ vector DB trên đĩa (mặc định), true/false=ép giá trị"

    p_eval = sub.add_parser("evaluate", help="Bắt đầu thi")
    p_eval.add_argument("--document-received", choices=DOC_CHOICES, default="auto", help=doc_help)
    sub.add_parser("reset",    help="Reset trạng thái thi")
    sub.add_parser("result",   help="Xem điểm hiện tại")
    p_all = sub.add_parser("all", help="register -> evaluate -> result (chạy tuần tự)")
    p_all.add_argument("--ip",   default=None)
    p_all.add_argument("--port", type=int, default=5000)
    p_all.add_argument("--document-received", choices=DOC_CHOICES, default="auto", help=doc_help)

    args = parser.parse_args()

    def parse_doc(val: str) -> Optional[bool]:
        return None if val == "auto" else (val == "true")

    print(f"[config] STUDENT_ID    = {STUDENT_ID}")
    print(f"[config] TEACHER_BASE  = {TEACHER_BASE_URL}")
    print()

    try:
        if args.cmd == "register":
            register(args.ip, args.port)
        elif args.cmd == "evaluate":
            evaluate(parse_doc(args.document_received))
        elif args.cmd == "reset":
            reset()
        elif args.cmd == "result":
            result()
        elif args.cmd == "all":
            register(args.ip, args.port)
            print()
            evaluate(parse_doc(args.document_received))
            print()
            result()
    except httpx.HTTPError as e:
        print(f"[error] HTTP: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
