import os
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import chromadb
from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

UPLOAD_DIR = Path("uploads")
DB_DIR = Path("db")
UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Book Chat")
app.mount("/static", StaticFiles(directory="static"), name="static")

chroma = chromadb.PersistentClient(path=str(DB_DIR))
gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = "gemini-2.0-flash"

sessions: dict[str, list[dict]] = {}
books: dict[str, dict] = {}


def extract_text_chunks(pdf_path: Path) -> list[dict]:
    """Extract text from PDF, split into chunks with page info."""
    doc = fitz.open(str(pdf_path))
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            continue
        # split into ~500 char chunks
        for i in range(0, len(text), 500):
            chunk = text[i:i + 500].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page_num + 1})
    return chunks


def ingest_book(book_id: str, pdf_path: Path) -> dict:
    """Extract and store book content in ChromaDB."""
    chunks = extract_text_chunks(pdf_path)
    if not chunks:
        raise ValueError("PDFからテキストを抽出できませんでした")

    collection = chroma.get_or_create_collection(f"book_{book_id}")
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[f"{book_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    # infer page count
    doc = fitz.open(str(pdf_path))
    return {"chunks": len(chunks), "pages": len(doc)}


def retrieve_context(book_id: str, query: str, n: int = 5) -> str:
    """RAG: retrieve relevant passages for a query."""
    try:
        collection = chroma.get_collection(f"book_{book_id}")
        results = collection.query(query_texts=[query], n_results=min(n, collection.count()))
        passages = results["documents"][0]
        pages = [m["page"] for m in results["metadatas"][0]]
        parts = [f"[p.{p}] {t}" for t, p in zip(passages, pages)]
        return "\n\n".join(parts)
    except Exception:
        return ""


# ── API models ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    book_id: str
    message: str
    mode: str = "discuss"  # "discuss" | "author" | "summary"


class BookMeta(BaseModel):
    book_id: str
    title: str
    author_hint: Optional[str] = ""


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path("static/index.html")).read_text()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDFファイルのみ対応しています")

    book_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{book_id}.pdf"
    save_path.write_bytes(await file.read())

    try:
        info = ingest_book(book_id, save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, str(e))

    title = Path(file.filename).stem
    books[book_id] = {"title": title, "author_hint": "", "pages": info["pages"]}

    return {"book_id": book_id, "title": title, "pages": info["pages"], "chunks": info["chunks"]}


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
        try:
            col = chroma.get_collection(f"book_{req.book_id}")
            all_docs = col.get()["documents"]
            full_text = " ".join(all_docs)[:8000]
        except Exception:
            full_text = context

        prompt = (
            f"あなたは『{meta['title']}』の読書アシスタントです。"
            "以下の本文を踏まえて日本語で詳しく要約してください。"
            "章立て・主要な主張・結論を含めてください。\n\n"
            f"本文:\n{full_text}\n\n要約してください。"
        )
        response = gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
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
    history.append({"role": "user", "parts": [{"text": req.message}]})

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1024),
    )
    reply = response.text
    history.append({"role": "model", "parts": [{"text": reply}]})

    if len(history) > 40:
        sessions[req.session_id] = history[-40:]

    return {"reply": reply, "mode": req.mode}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}
