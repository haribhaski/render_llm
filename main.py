# === main.py (updated) ===
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import re
import inspect
import nltk
import google.generativeai as genai
from dotenv import load_dotenv

# ⬇️ FIX: use punkt (not punkt_tab)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ------------ Local modules ------------
from pdf import (
    extract_text_from_eml_file,
    extract_text_from_docx_file,
    extract_text_from_pdf_url,
    download_file,
)
import pdf

# ⬇️ NEW: import the Pinecone-ready engine we wrote earlier.
# Save that code as embeddings_engine.py next to this file.
from embeddings_engine import (
    create_search_engine,            # returns DocumentSearchEngine()
    extract_and_create_clauses,      # smarter sentence-window chunking
)

# ------------ Env & constants ------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Add it to your .env file.")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pinecone").lower()  # 'pinecone' (default) or 'faiss'
INDEX_PATH = "index/faiss.index"   # only used when VECTOR_BACKEND=faiss
CLAUSES_PATH = "index/clauses.pkl" # only used when VECTOR_BACKEND=faiss

# ------------ App ------------
app = FastAPI()

@app.get("/debug/pdf-functions")
async def debug_pdf_functions():
    functions = [name for name, obj in inspect.getmembers(pdf, inspect.isfunction)]
    return {"pdf_functions": functions}

# ------------ Gemini LLM for reasoning ------------
# Keep using Pro for reasoning; use Flash if you want it cheaper.
try:
    genai.configure(api_key=GEMINI_API_KEY)
    reasoning_model = genai.GenerativeModel(model_name=os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-pro"))
except Exception as e:
    raise RuntimeError(f"❌ Failed to initialize Gemini LLM: {str(e)}")

# ------------ Engine init ------------
# ⬇️ NEW: create the Pinecone/FAISS-capable engine
search_engine = create_search_engine()

# ⬇️ Only try FAISS disk load if you actually use FAISS
if VECTOR_BACKEND == "faiss" and os.path.exists(INDEX_PATH) and os.path.exists(CLAUSES_PATH):
    print("📂 Loading existing FAISS index and clause data...")
    # The new engine no longer exposes faiss_manager publicly; rebuild from disk is your responsibility.
    # If you want persistent FAISS across runs, keep a small helper that loads vectors + meta and re-adds to index.
    # For simplicity, we’ll defer and just rebuild per request when FAISS is used.
    print("ℹ️ Skipping legacy direct load; engine will rebuild on demand.")
else:
    if VECTOR_BACKEND == "faiss":
        print("⚠️ No saved FAISS index found. Will build from uploaded documents.")
    else:
        print("ℹ️ Using Pinecone backend — no local index to load.")

# ------------ Schemas ------------
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

# ------------ Auth ------------
def verify_token(authorization: Optional[str] = Header(None)):
    expected_token = "254dbddb097c0ab2b89f6277bbbf7a7daae4b243cf2bc42d73ef4d5e16bba557"
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

# ------------ Document text extraction ------------
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

# ------------ Reasoning ------------
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

    # Build a compact context payload
    clause_lines = []
    for h in hits:
        cid = h.get("id", "")
        m = h.get("metadata", {}) or {}
        ctext = m.get("text", "")
        clause_lines.append(f"Clause {cid}: {ctext}")

    clause_texts = "\n".join(clause_lines)
    prompt = f"""
You are an expert at answering questions from retrieved document chunks.

Query:
{query}

Relevant Chunks:
{clause_texts}

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
        # try strict json first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in response.")
            data = json.loads(m.group(0))

        # Normalize clause refs
        out_refs = []
        for h in hits:
            # prefer model-provided list, else fall back to hits
            pass
        # If the model provided refs, keep them; if not, synthesize from hits
        if isinstance(data.get("clause_references"), list) and data["clause_references"]:
            refs = []
            for c in data["clause_references"]:
                refs.append(ClauseReference(id=str(c.get("id", "")), text=str(c.get("text", ""))))
        else:
            refs = [
                ClauseReference(id=str(h.get("id", "")), text=str((h.get("metadata") or {}).get("text", "")))
                for h in hits
            ]

        return {
            "answer": str(data.get("answer", "")).strip(),
            "explanation": str(data.get("explanation", "")).strip(),
            "clause_references": refs
        }
    except Exception as e:
        print("❌ LLM Reasoning Failed:", str(e))
        # fall back: return concatenated chunk texts to help caller
        fallback_refs = [
            ClauseReference(id=str(h.get("id","")), text=str((h.get("metadata") or {}).get("text","")))
            for h in hits
        ]
        return {"answer": "Could not generate answer.", "explanation": str(e), "clause_references": fallback_refs}

# ------------ Routes ------------
@app.get("/")
async def root():
    return {"message": f"✅ LLM Query API (Gemini + {VECTOR_BACKEND.upper()}) is live!"}

@app.post("/api/v1/query", response_model=List[AnswerResponse])
async def handle_query(req: QueryRequest):
    try:
        # 1) Extract + chunk all docs
        all_chunks = []
        for i, url in enumerate(req.documents):
            if not (isinstance(url, str) and url.startswith("http")):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
            print(f"📄 Processing document {i+1}: {url}")
            text = extract_text_from_document(url)
            chunks = extract_and_create_clauses(text, f"doc_{i+1}")  # includes metadata["text"]
            all_chunks.extend(chunks)
            print(f"✅ {len(chunks)} chunks extracted from document {i+1}")

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No chunks extracted from any document.")

        # 2) Build index (Pinecone or FAISS depending on env)
        print(f"⚙️ Building {VECTOR_BACKEND.upper()} index with Gemini embeddings...")
        search_engine.build(all_chunks)  # ⬅️ NEW

        results: List[AnswerResponse] = []
        # 3) Answer each question
        for q in req.questions:
            print(f"❓ Query: {q}")
            # engine.search returns list of (id, score, metadata)
            raw_hits = search_engine.search(q, top_k=5, use_expansion=True, mmr_lambda=0.6)

            # adapt to reasoning input (id + metadata["text"])
            hits = [{"id": cid, "score": sc, "metadata": md} for (cid, sc, md) in raw_hits]
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

    # Chunk the single doc
    chunks = extract_and_create_clauses(full_text, "submission_doc")
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks extracted from document.")

    print(f"✂️ Extracted {len(chunks)} chunks. Embedding and indexing...")
    search_engine.build(chunks)  # ⬅️ NEW

    answers: List[str] = []
    for question in req.questions:
        print(f"❓ Question: {question}")
        raw_hits = search_engine.search(question, top_k=5, use_expansion=True, mmr_lambda=0.6)
        hits = [{"id": cid, "score": sc, "metadata": md} for (cid, sc, md) in raw_hits]
        result = llm_reasoning_with_gemini(question, hits)
        ans = str(result.get("answer", "")).strip() or "No relevant information found."
        answers.append(ans)

    return {"answers": answers}

@app.get("/health")
async def health():
    return {"status": "ok", "embedding_model": os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001"),
            "engine": VECTOR_BACKEND}
