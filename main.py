### ✅ Cleaned + FIXED Gemini-Faiss FastAPI with Robust JSON Parsing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from fastapi import Header, Depends
from typing import Optional
from pydantic import BaseModel
import google.generativeai as genai

INDEX_PATH = "index/faiss.index"
CLAUSES_PATH = "index/clauses.pkl"
from pdf import extract_text_from_eml_file, extract_text_from_docx_file, extract_text_from_pdf_url, download_file
from fastapi import FastAPI
import pdf
import inspect


from embeddings_engine import (
    create_gemini_search_engine,
    extract_and_create_clauses,
    DocumentSearchEngine,
    Clause
)
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

@app.get("/debug/pdf-functions")
async def debug_pdf_functions():
    functions = [name for name, obj in inspect.getmembers(pdf, inspect.isfunction)]
    return {"pdf_functions": functions}
# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Add it to your .env file.")

# Configure Gemini for LLM reasoning
try:
    genai.configure(api_key=GEMINI_API_KEY)
    reasoning_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
except Exception as e:
    raise RuntimeError(f"❌ Failed to initialize Gemini LLM: {str(e)}")

# Create search engine
search_engine: DocumentSearchEngine = create_gemini_search_engine(GEMINI_API_KEY)
if os.path.exists(INDEX_PATH) and os.path.exists(CLAUSES_PATH):
    print("📂 Loading existing FAISS index and clause data...")
    search_engine.faiss_manager.load_index(INDEX_PATH, CLAUSES_PATH)
    print(f"✅ Loaded {len(search_engine.faiss_manager.clauses)} clauses from disk.")
else:
    print("⚠️ No saved index found. Will build from uploaded documents.")

# API Schemas
class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class ClauseReference(BaseModel):
    id: str
    text: str

class AnswerResponse(BaseModel):
    query: str
    answer: str
    explanation: str
    clause_references: List[ClauseReference]

class RunRequest(BaseModel):
    documents: str  # single URL string
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# Document extractor helper
def verify_token(authorization: Optional[str] = Header(None)):
    expected_token = "254dbddb097c0ab2b89f6277bbbf7a7daae4b243cf2bc42d73ef4d5e16bba557"
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

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
        os.remove(local_path)

# Reasoning with Gemini Pro

def llm_reasoning_with_gemini(query: str, clauses: List[Clause]) -> dict:
    if not clauses:
        return {
            "answer": "I couldn't find relevant information in the documents.",
            "explanation": "No matching clauses were found for this query.",
            "clause_references": []
        }

    clause_texts = "\n".join([f"Clause {c.id}: {c.text}" for c in clauses])
    prompt = f"""
You are an expert in analyzing research papers.
Given a user query and relevant document clauses, provide a precise answer.

Query: {query}

Relevant Clauses:
{clause_texts}

Respond ONLY in valid JSON format with:
- answer (string)
- explanation (string)
- clause_references (list of objects with id and text)
"""
    try:
        response = reasoning_model.generate_content(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                top_p=0.95,
                max_output_tokens=1024
            )
        )
        print("🔎 Gemini response:\n", response.text)

        # Try full text parse first
        try:
            return_json = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # Fallback to regex extraction if needed
            match = re.search(r'{.*}', response.text, re.DOTALL)
            if not match:
                raise ValueError("Gemini response did not contain valid JSON object.")
            json_text = match.group(0)
            return_json = json.loads(json_text)

        return {
            "answer": return_json.get("answer", ""),
            "explanation": return_json.get("explanation", ""),
            "clause_references": [
                ClauseReference(id=c.get("id", ""), text=c.get("text", ""))
                for c in return_json.get("clause_references", [])
            ]
        }

    except Exception as e:
        print("❌ LLM Reasoning Failed:", str(e))
        return {
            "answer": "Could not generate answer.",
            "explanation": str(e),
            "clause_references": [ClauseReference(id=c.id, text=c.text) for c in clauses]
        }

@app.get("/")
async def root():
    return {"message": "✅ LLM Query API (Gemini + FAISS) is live!"}


@app.post("/api/v1/query", response_model=List[AnswerResponse])
async def handle_query(req: QueryRequest):
    try:
        all_clauses: List[Clause] = []
        for i, url in enumerate(req.documents):
            if not url.startswith("http"):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
            print(f"📄 Processing document {i+1}: {url}")
            doc_text = extract_text_from_document(url)
            clauses = extract_and_create_clauses(doc_text, f"doc_{i+1}")
            all_clauses.extend(clauses)
            print(f"✅ {len(clauses)} clauses extracted from document {i+1}")

        if not all_clauses:
            raise HTTPException(status_code=400, detail="No clauses extracted from any document.")

        print("⚙️ Building FAISS index with Gemini embeddings...")
        search_engine.build_search_index(all_clauses)
        if not search_engine.faiss_manager.index:
            print("🏗️ No index loaded yet. Building search index...")
            search_engine.build_search_index(all_clauses, batch_size=10)
            search_engine.faiss_manager.save_index(INDEX_PATH, CLAUSES_PATH)
        else:
            print("✅ Using already loaded FAISS index.")



        results = []
        for query in req.questions:
            print(f"❓ Query: {query}")
            search_hits = search_engine.search(query, top_k=3)
            matched = [clause for clause, _ in search_hits]
            result = llm_reasoning_with_gemini(query, matched)
            results.append(AnswerResponse(
                query=query,
                answer=result["answer"],
                explanation=result["explanation"],
                clause_references=result["clause_references"]
            ))
        return results

    except Exception as e:
        print("❌ Query handling failed:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(req: RunRequest, token: str = Depends(verify_token)):
    try:
        # 1. Extract full text from single document
        print(f"📄 Downloading document from: {req.documents}")
        full_text = extract_text_from_document(req.documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to extract document text: {str(e)}")

    # 2. Split and embed clauses using your Gemini clause engine
    clauses = extract_and_create_clauses(full_text, "submission_doc")
    if not clauses:
        raise HTTPException(status_code=400, detail="No clauses extracted from document.")

    print(f"✂️ Extracted {len(clauses)} clauses. Embedding and indexing...")
    search_engine.build_search_index(clauses)

    # 3. Use Gemini to answer each question
    answers = []
    for question in req.questions:
        print(f"❓ Question: {question}")
        top_hits = search_engine.search(question, top_k=3)
        matched_clauses = [clause for clause, _ in top_hits]
        result = llm_reasoning_with_gemini(question, matched_clauses)
        answers.append(result["answer"])

    return RunResponse(answers=answers)

@app.get("/health")
async def health():
    return {"status": "ok", "embedding_model": "Gemini", "engine": "FAISS"}





