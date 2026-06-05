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
from urllib.parse import urlparse
import httpx
from dotenv import load_dotenv

load_dotenv()

STUDENT_ID       = os.getenv("STUDENT_ID",       "B22DCAT082").upper()
TEACHER_BASE_URL = os.getenv("TEACHER_BASE_URL", "http://10.170.45.200:8000/api/v1")

HEADERS = {"X-Student-ID": STUDENT_ID}


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


def evaluate() -> None:
    print(f"[evaluate] student={STUDENT_ID} — bắt đầu thi...")
    print("(Teacher sẽ gọi /upload (120s) rồi 10 lần /ask (60s mỗi câu) tới server của bạn)")
    resp = httpx.post(
        f"{TEACHER_BASE_URL}/competition/evaluate",
        headers=HEADERS,
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

    sub.add_parser("evaluate", help="Bắt đầu thi")
    sub.add_parser("reset",    help="Reset trạng thái thi")
    sub.add_parser("result",   help="Xem điểm hiện tại")
    p_all = sub.add_parser("all", help="register -> evaluate -> result (chạy tuần tự)")
    p_all.add_argument("--ip",   default=None)
    p_all.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    print(f"[config] STUDENT_ID    = {STUDENT_ID}")
    print(f"[config] TEACHER_BASE  = {TEACHER_BASE_URL}")
    print()

    try:
        if args.cmd == "register":
            register(args.ip, args.port)
        elif args.cmd == "evaluate":
            evaluate()
        elif args.cmd == "reset":
            reset()
        elif args.cmd == "result":
            result()
        elif args.cmd == "all":
            register(args.ip, args.port)
            print()
            evaluate()
            print()
            result()
    except httpx.HTTPError as e:
        print(f"[error] HTTP: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
