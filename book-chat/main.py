import os
import uuid
import math
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict

import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Book Chat")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = "gemini-1.5-flash"

sessions: dict[str, list[dict]] = {}
books: dict[str, dict] = {}
# book_id -> list of {"text": str, "page": int}
book_chunks: dict[str, list[dict]] = {}


def extract_text_chunks(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            continue
        for i in range(0, len(text), 500):
            chunk = text[i:i + 500].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page_num + 1})
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def retrieve_context(book_id: str, query: str, n: int = 5) -> str:
    chunks = book_chunks.get(book_id, [])
    if not chunks:
        return ""
    query_tokens = set(tokenize(query))
    scores = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk["text"])
        token_freq: dict[str, int] = defaultdict(int)
        for t in chunk_tokens:
            token_freq[t] += 1
        score = sum(token_freq.get(t, 0) for t in query_tokens)
        scores.append(score)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    parts = [f"[p.{chunks[i]['page']}] {chunks[i]['text']}" for i in top_indices if scores[i] > 0]
    return "\n\n".join(parts) if parts else "\n\n".join(
        chunks[i]["text"] for i in top_indices[:3]
    )


class ChatRequest(BaseModel):
    session_id: str
    book_id: str
    message: str
    mode: str = "discuss"


class BookMeta(BaseModel):
    book_id: str
    title: str
    author_hint: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def root():
    return (BASE_DIR / "static/index.html").read_text()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDFファイルのみ対応しています")
    book_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{book_id}.pdf"
    save_path.write_bytes(await file.read())
    try:
        chunks = extract_text_chunks(save_path)
        if not chunks:
            raise ValueError("PDFからテキストを抽出できませんでした")
        book_chunks[book_id] = chunks
        doc = fitz.open(str(save_path))
        pages = len(doc)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, str(e))
    title = Path(file.filename).stem
    books[book_id] = {"title": title, "author_hint": "", "pages": pages}
    return {"book_id": book_id, "title": title, "pages": pages, "chunks": len(chunks)}


@app.post("/meta")
async def set_meta(meta: BookMeta):
    if meta.book_id not in books:
        raise HTTPException(404, "書籍が見つかりません")
    books[meta.book_id]["title"] = meta.title
    books[meta.book_id]["author_hint"] = meta.author_hint or ""
    return {"ok": True}


@app.get("/books")
async def list_books():
    return [{"book_id": k, **v} for k, v in books.items()]


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.book_id not in books:
        raise HTTPException(404, "書籍が見つかりません")
    meta = books[req.book_id]
    context = retrieve_context(req.book_id, req.message)

    if req.mode == "summary":
        chunks = book_chunks.get(req.book_id, [])
        full_text = " ".join(c["text"] for c in chunks)[:8000]
        prompt = (
            f"あなたは『{meta['title']}』の読書アシスタントです。"
            "以下の本文を踏まえて日本語で詳しく要約してください。"
            "章立て・主要な主張・結論を含めてください。\n\n"
            f"本文:\n{full_text}\n\n要約してください。"
        )
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return {"reply": response.text, "mode": "summary"}

    if req.mode == "author":
        author_part = f"（著者名のヒント: {meta['author_hint']}）" if meta["author_hint"] else ""
        system = (
            f"あなたは『{meta['title']}』{author_part}の著者本人です。"
            "本書に書かれた思想・主張・文体を忠実に再現して一人称で答えてください。"
            "本書に書かれていないことは「本書では触れていませんが…」と断ったうえで答えてください。"
            f"\n\n【本書の関連箇所】\n{context}"
        )
    else:
        system = (
            f"あなたは『{meta['title']}』の内容に精通した読書アシスタントです。"
            "日本語で丁寧に、本書の内容を根拠にしながら議論・質問に答えてください。"
            f"\n\n【本書の関連箇所】\n{context}"
        )

    history = sessions.setdefault(req.session_id, [])
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(req.message)
    reply = response.text
    history.clear()
    history.extend(chat_session.history)
    if len(history) > 40:
        sessions[req.session_id] = history[-40:]
    return {"reply": reply, "mode": req.mode}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}
