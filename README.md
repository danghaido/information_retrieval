# RAG Demo — Hướng dẫn chạy

## 1. Cấp quyền & kích hoạt uv (Windows PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Kích hoạt môi trường uv:

```powershell
.venv\Scripts\activate
```

(Nếu chưa có môi trường: `uv sync`)

## 2. Download model (làm 1 lần, lúc còn Internet)

```bash
python download_model.py
```

## 3. Host server

```bash
python rag_demo.py
```

Server chạy ở `http://0.0.0.0:5000`.

## 4. Chạy trigger (mở terminal khác, để server vẫn chạy)

Chạy lần lượt:

```bash
python trigger.py register
python trigger.py evaluate
python trigger.py result
python trigger.py reset
```
