# Book Chat — PDFと対話するWebアプリ

## セットアップ

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## 起動

```bash
cd book-chat
uvicorn main:app --reload --port 8000
```

ブラウザで http://localhost:8000 を開く。

## 使い方

1. 左サイドバーにPDFをドラッグ&ドロップ（または選択）
2. 著者名を入力（任意・著者モード用）
3. モードを選択:
   - 💬 **議論モード** — 本の内容を根拠に質問・議論
   - ✍️ **著者モード** — 著者本人として一人称で回答
4. **📝 要約** ボタンで全体要約を生成

## アーキテクチャ

```
PDF → PyMuPDF (テキスト抽出) → ChromaDB (RAGベクトルDB)
                                      ↓
ユーザー質問 → 関連箇所検索 → Claude API → 回答
```
