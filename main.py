# === main.py (Pinecone Integrated + Gemini Reasoning, Render-ready) ===
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os
import json
import re
import inspect
import google.generativeai as genai

# Pinecone (serverless SDK)
from pinecone import Pinecone

# Local doc parsers
from pdf import (
    extract_text_from_eml_file,
    extract_text_from_docx_file,
    extract_text_from_pdf_url,
    download_file,
)
import pdf

# ------------------ NLTK setup for Render ------------------
import pathlib
import nltk

NLTK_DATA_DIR = "/opt/render/nltk_data"
pathlib.Path(NLTK_DATA_DIR).mkdir(parents=True, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Ensure both resources exist (newer NLTK may require punkt_tab)
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, download_dir=NLTK_DATA_DIR, quiet=True)

# Safe tokenizer wrapper with regex fallback
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
def sent_tokenize_safe(text: str) -> List[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
# ------------------------------------------------------------

# ------------------ Environment (Render) --------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")              # e.g., "hackrx-rhwc8t2"
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "default")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment.")
if not PINECONE_INDEX:
    raise RuntimeError("PINECONE_INDEX is not set in environment.")
# ------------------------------------------------------------

# ------------------ FastAPI app -----------------------------
app = FastAPI()

# ------------------ Data models -----------------------------
@dataclass
class Clause:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class ClauseReference(BaseModel):
    id: str
    text: str

class AnswerResponse(BaseModel):
    query: str
    answer: str
    explanation: str
    clause_references: List[ClauseReference]

class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class RunRequest(BaseModel):
    documents: str  # single URL
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# ------------------ Auth -----------------------------
def verify_token(authorization: Optional[str] = Header(None)):
    expected_token = "254dbddb097c0ab2b89f6277bbbf7a7daae4b243cf2bc42d73ef4d5e16bba557"
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

# ------------------ Pinecone client ------------------
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index("quickstart")

# ------------------ Chunking -------------------------
def smart_sentence_chunks(
    text: str,
    doc_id: str,
    max_words: int = 140,
    stride: int = 2,
) -> List[Clause]:
    sents = [s.strip() for s in sent_tokenize_safe(text) if s.strip()]
    if not sents:
        return []

    chunks: List[Clause] = []
    i = 0
    chunk_id = 1
    while i < len(sents):
        cur = []
        w = 0
        j = i
        while j < len(sents) and w < max_words:
            cur.append(sents[j])
            w += len(sents[j].split())
            j += 1
        chunk_text = " ".join(cur).strip()
        meta = {"document_id": doc_id, "chunk_id": chunk_id, "start_sent": i, "end_sent": j - 1, "text": chunk_text}
        chunks.append(Clause(id=f"{doc_id}_{chunk_id}", text=chunk_text, metadata=meta))
        chunk_id += 1
        i = j - stride if (j - stride) > i else j
    return chunks

def extract_and_create_clauses(document_text: str, doc_id: str = "doc") -> List[Clause]:
    return smart_sentence_chunks(document_text, doc_id=doc_id, max_words=140, stride=2)

# ------------------ Document extraction ---------------
def extract_text_from_document(doc_url: str) -> str:
    local_path = download_file(doc_url)
    try:
        cleaned_name = re.sub(r"\?.*$", "", local_path)
        ext = os.path.splitext(cleaned_name)[1].lower()
        if ext == ".pdf":
            return extract_text_from_pdf_url(local_path)
        elif ext == ".docx":
            return extract_text_from_docx_file(local_path)
        elif ext == ".eml":
            return extract_text_from_eml_file(local_path)
        else:
            raise ValueError(f"Unsupported document type: {ext}")
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

# ------------------ Gemini reasoning ------------------
genai.configure(api_key=GEMINI_API_KEY)
reasoning_model = genai.GenerativeModel(model_name=os.environ.get("GEMINI_TEXT_MODEL", "gemini-1.5-pro"))

def llm_reasoning_with_gemini(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    hits: list of dicts like {"id": str, "score": float, "metadata": {"text": "...", ...}}
    """
    if not hits:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "explanation": "No matching chunks were found for this query.",
            "clause_references": []
        }

    parts = []
    for h in hits:
        cid = h.get("id", "")
        m = h.get("metadata", {}) or {}
        ctext = m.get("text", "")
        parts.append(f"Clause {cid}: {ctext}")
    context = "\n".join(parts)

    prompt = f"""
You are an expert at answering questions from retrieved document chunks.

Query:
{query}

Relevant Chunks:
{context}

Return ONLY valid JSON with exactly these keys:
- "answer": string
- "explanation": string
- "clause_references": list of objects with keys "id" and "text"
"""
    try:
        resp = reasoning_model.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                max_output_tokens=800
            )
        )
        raw = (resp.text or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in response.")
            data = json.loads(m.group(0))

        if isinstance(data.get("clause_references"), list) and data["clause_references"]:
            refs = [ClauseReference(id=str(c.get("id", "")), text=str(c.get("text", "")))
                    for c in data["clause_references"]]
        else:
            refs = [ClauseReference(id=str(h.get("id", "")), text=str((h.get("metadata") or {}).get("text", "")))
                    for h in hits]

        return {
            "answer": str(data.get("answer", "")).strip(),
            "explanation": str(data.get("explanation", "")).strip(),
            "clause_references": refs
        }
    except Exception as e:
        fallback_refs = [
            ClauseReference(id=str(h.get("id", "")), text=str((h.get("metadata") or {}).get("text", "")))
            for h in hits
        ]
        return {"answer": "Could not generate answer.", "explanation": str(e), "clause_references": fallback_refs}

# ------------------ Routes ----------------------------
@app.get("/")
async def root():
    return {"message": "✅ LLM Query API (Gemini + Pinecone Integrated) is live!"}

@app.get("/debug/pdf-functions")
async def debug_pdf_functions():
    functions = [name for name, obj in inspect.getmembers(pdf, inspect.isfunction)]
    return {"pdf_functions": functions}

@app.post("/api/v1/query", response_model=List[AnswerResponse])
async def handle_query(req: QueryRequest):
    try:
        # 1) Extract & chunk all docs
        all_chunks: List[Clause] = []
        for i, url in enumerate(req.documents):
            if not (isinstance(url, str) and url.startswith("http")):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
            print(f"📄 Processing document {i+1}: {url}")
            text = extract_text_from_document(url)
            chunks = extract_and_create_clauses(text, f"doc_{i+1}")
            all_chunks.extend(chunks)
            print(f"✅ {len(chunks)} chunks extracted from document {i+1}")

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No chunks extracted from any document.")

        # 2) Upsert via Pinecone Integrated Embeddings (top-level 'text' is embedded)
        print("⚙️ Upserting with Pinecone Integrated Embeddings (llama-text-embed-v2, dim=1024)...")
        index = get_pinecone_index()
        records = [{"id": c.id, "text": c.text, "metadata": dict(c.metadata or {})} for c in all_chunks]
        B = 200
        for s in range(0, len(records), B):
            index.upsert_records(namespace=PINECONE_NAMESPACE, records=records[s:s+B])

        # 3) Answer each question using Gemini
        results: List[AnswerResponse] = []
        for q in req.questions:
            print(f"❓ Query: {q}")
            res = index.search(
                namespace=PINECONE_NAMESPACE,
                query={"inputs": {"text": q}, "top_k": 5},
                include_metadata=True
            )
            hits = [{"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
                    for m in (res.matches or [])]
            r = llm_reasoning_with_gemini(q, hits)
            results.append(AnswerResponse(
                query=q,
                answer=r["answer"],
                explanation=r["explanation"],
                clause_references=r["clause_references"]
            ))
        return results

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Query handling failed:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, token: str = Depends(verify_token)):
    try:
        print(f"📄 Downloading document from: {req.documents}")
        full_text = extract_text_from_document(req.documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to extract document text: {str(e)}")

    # 1) Chunk the single doc
    chunks = extract_and_create_clauses(full_text, "submission_doc")
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks extracted from document.")

    # 2) Upsert to Pinecone (integrated embeddings)
    print(f"✂️ Extracted {len(chunks)} chunks. Upserting to Pinecone...")
    index = get_pinecone_index()
    records = [{"id": c.id, "text": c.text, "metadata": dict(c.metadata or {})} for c in chunks]
    B = 200
    for s in range(0, len(records), B):
        index.upsert_records(namespace=PINECONE_NAMESPACE, records=records[s:s+B])

    # 3) Search & answer
    answers: List[str] = []
    for question in req.questions:
        print(f"❓ Question: {question}")
        res = index.search(
            namespace=PINECONE_NAMESPACE,
            query={"inputs": {"text": question}, "top_k": 5},
            include_metadata=True
        )
        hits = [{"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
                for m in (res.matches or [])]
        result = llm_reasoning_with_gemini(question, hits)
        ans = str(result.get("answer", "")).strip() or "No relevant information found."
        answers.append(ans)

    return {"answers": answers}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedding_backend": "pinecone-integrated(llama-text-embed-v2)",
        "pinecone_index": PINECONE_INDEX,
        "namespace": PINECONE_NAMESPACE,
        "llm": os.environ.get("GEMINI_TEXT_MODEL", "gemini-1.5-pro"),
    }

