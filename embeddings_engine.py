import os
import json
import time
import hashlib
import pickle
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import nltk
nltk.download("punkt", quiet=True)

# --- Optional local backend
try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

# --- Pinecone serverless backend (modern SDK)
# pip install pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None

import google.generativeai as genai  # pip install google-generativeai

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc-search")

# -------------------- Data model --------------------
@dataclass
class Clause:
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

# -------------------- Utils --------------------
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; safe for zeros."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b.T))

def now_ms() -> int:
    return int(time.time() * 1000)

# -------------------- Gemini embeddings --------------------
class GeminiEmbeddingGenerator:
    """
    Embedding generator with:
    - disk cache
    - rate limiting
    - dimension auto-detection
    - batch generation (looped, respects rate limit)
    """
    def __init__(
        self,
        api_key: str,
        cache_dir: str = os.getenv("EMBEDDINGS_CACHE_DIR", "./embeddings_cache"),
        model: str = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001"),
        min_interval_s: float = float(os.getenv("GEMINI_MIN_INTERVAL_S", "0.12")),
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        genai.configure(api_key=api_key)

        self._cache_lock = threading.Lock()
        self._last_ts = 0.0
        self._min_interval = min_interval_s
        self._dim: Optional[int] = None

        logger.info("✅ Gemini embedding generator ready")

    # ---------- caching ----------
    def _key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _get_cache(self, text: str) -> Optional[np.ndarray]:
        p = self._path(self._key(text))
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        return None

    def _set_cache(self, text: str, emb: np.ndarray):
        p = self._path(self._key(text))
        try:
            with self._cache_lock:
                with open(p, "wb") as f:
                    pickle.dump(emb, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _rate_limit(self):
        dt = time.time() - self._last_ts
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last_ts = time.time()

    def _infer_dim(self, sample_emb: np.ndarray):
        if self._dim is None:
            self._dim = int(sample_emb.shape[-1])
            logger.info(f"📏 Detected embedding dim={self._dim}")

    # ---------- public ----------
    def embed_one(self, text: str, use_cache=True) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(self._dim or 768, dtype=np.float32)

        if use_cache:
            cached = self._get_cache(text)
            if cached is not None:
                return cached.astype(np.float32)

        try:
            self._rate_limit()
            res = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
            vec = np.array(res["embedding"], dtype=np.float32)
            self._infer_dim(vec)
            self._set_cache(text, vec) if use_cache else None
            return vec
        except Exception as e:
            logger.error(f"Gemini embed error: {e}")
            # fall back to dim if known, else 768
            return np.zeros(self._dim or 768, dtype=np.float32)

    def embed_many(self, texts: List[str], use_cache=True, batch_log_every=50) -> List[np.ndarray]:
        out: List[Optional[np.ndarray]] = [None] * len(texts)
        to_do = []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = np.zeros(self._dim or 768, dtype=np.float32)
                continue
            if use_cache:
                cached = self._get_cache(t)
                if cached is not None:
                    out[i] = cached.astype(np.float32)
                    continue
            to_do.append((i, t))

        for k, (i, t) in enumerate(to_do, 1):
            out[i] = self.embed_one(t, use_cache=False)
            if use_cache:
                self._set_cache(t, out[i])
            if k % batch_log_every == 0:
                logger.info(f"🔁 Embedded {k}/{len(to_do)}")

        # Infer dim if still None
        if self._dim is None:
            for v in out:
                if v is not None and v.size > 0:
                    self._infer_dim(v)
                    break

        # Normalize (cosine-friendly, speeds similarity)
        out_np = [v if v is not None else np.zeros(self._dim or 768, dtype=np.float32) for v in out]
        if self._dim:
            return [v / (np.linalg.norm(v) + 1e-12) for v in out_np]
        return out_np

# -------------------- Vector Stores --------------------
class VectorStore:
    """Interface shared by Pinecone and FAISS backends."""
    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict]):
        raise NotImplementedError

    def query(self, vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Returns list of (id, score, metadata) in DESC score order."""
        raise NotImplementedError

    def ready(self) -> bool:
        return True

# --- Pinecone backend ---
class PineconeVectorStore(VectorStore):
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dim: int,
        metric: str = "cosine",
        namespace: str = "default",
        create_if_missing: bool = True,
        cloud: Optional[str] = os.getenv("PINECONE_CLOUD"),
        region: Optional[str] = os.getenv("PINECONE_REGION"),
    ):
        if Pinecone is None:
            raise RuntimeError("pinecone package not installed")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dim = dim
        self.metric = metric
        self.namespace = namespace

        if create_if_missing:
            existing = {i["name"] for i in self.pc.list_indexes()}
            if index_name not in existing:
                if ServerlessSpec is None:
                    raise RuntimeError("ServerlessSpec unavailable; update pinecone package")
                if not (cloud and region):
                    # sensible defaults (AWS us-east-1) if not provided
                    cloud, region = "aws", "us-east-1"
                logger.info(f"🧱 Creating Pinecone index '{index_name}' (dim={dim}, metric={metric})")
                self.pc.create_index(
                    name=index_name,
                    dimension=dim,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )

        self.index = self.pc.Index(index_name)
        logger.info(f"✅ Pinecone index ready: {index_name}")

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict]):
        # Pinecone expects list of dicts: {"id": "...", "values": [...], "metadata": {...}}
        # Ensure normalized vectors for cosine
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        items = [{"id": _id, "values": vecs[i].tolist(), "metadata": metadatas[i]} for i, _id in enumerate(ids)]
        # Chunk large upserts
        B = 200
        for s in range(0, len(items), B):
            self.index.upsert(vectors=items[s:s+B], namespace=self.namespace)

    def query(self, vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        v = l2_normalize(vector.reshape(1, -1).astype(np.float32))[0].tolist()
        res = self.index.query(
            namespace=self.namespace,
            vector=v,
            top_k=top_k,
            include_metadata=True
        )
        out = []
        for m in res.matches or []:
            out.append((m.id, float(m.score), dict(m.metadata or {})))
        return out

# --- FAISS backend (local) ---
class FaissVectorStore(VectorStore):
    def __init__(self, dim: int, index_type: str = "hnsw"):
        if faiss is None:
            raise RuntimeError("faiss not installed")
        self.dim = dim
        self.index_type = index_type
        self.index = self._create_index()
        self.ids: List[str] = []
        self.meta: List[Dict] = []

    def _create_index(self):
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dim)  # use IP on normalized vectors (=cosine)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32)
            index.hnsw.efSearch = 64
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported FAISS index_type: {self.index_type}")
        return index

    def upsert(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict]):
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self.index.add(vecs)
        self.ids.extend(ids)
        self.meta.extend(metadatas)

    def query(self, vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        v = l2_normalize(vector.reshape(1, -1).astype(np.float32))
        scores, idxs = self.index.search(v, top_k)
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1 or i >= len(self.ids):
                continue
            out.append((self.ids[i], float(s), self.meta[i]))
        return out

# -------------------- Chunking --------------------
def smart_sentence_chunks(
    text: str,
    doc_id: str,
    max_words: int = 140,
    stride: int = 2,
) -> List[Clause]:
    """
    Sentence chunking with a sliding window to preserve context.
    - max_words controls chunk size.
    - stride controls overlap (in sentences).
    """
    sents = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    if not sents:
        return []

    # build sentence windows
    chunks = []
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
        chunks.append(Clause(
            id=f"{doc_id}_{chunk_id}",
            text=chunk_text,
            metadata={"document_id": doc_id, "chunk_id": chunk_id, "start_sent": i, "end_sent": j-1}
        ))
        chunk_id += 1
        # advance with overlap
        i = j - stride if (j - stride) > i else j

    return chunks

# -------------------- Retrieval helpers --------------------
def mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    lambda_mult: float = 0.6,
    top_k: int = 5
) -> List[int]:
    """
    Maximal Marginal Relevance to increase diversity.
    Returns indices into cand_vecs in selection order.
    """
    if cand_vecs.shape[0] == 0:
        return []
    query_vec = l2_normalize(query_vec.reshape(1, -1))[0]
    cand_vecs = l2_normalize(cand_vecs)
    sim_to_query = (cand_vecs @ query_vec)  # cosine
    selected = []
    remaining = set(range(cand_vecs.shape[0]))
    while len(selected) < min(top_k, cand_vecs.shape[0]):
        if not selected:
            i = int(np.argmax(sim_to_query))
            selected.append(i)
            remaining.remove(i)
            continue
        max_score = -1e9
        best_i = None
        for i in list(remaining):
            max_sim_to_selected = np.max(cand_vecs[i:i+1] @ cand_vecs[selected].T)
            score = lambda_mult * sim_to_query[i] - (1 - lambda_mult) * max_sim_to_selected
            if score > max_score:
                max_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

def rrf_merge(rank_lists: List[List[Tuple[str, float, Dict]]], k: int = 60, top_k: int = 8):
    """
    Reciprocal Rank Fusion across multiple query variants.
    Input: list of lists of (id, score, meta) in descending score order.
    """
    scores: Dict[str, float] = {}
    meta_ref: Dict[str, Dict] = {}
    for lst in rank_lists:
        for rank, (cid, _, m) in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            meta_ref.setdefault(cid, m)
    # sort by fused score
    final = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(cid, float(sc), meta_ref[cid]) for cid, sc in final]

# -------------------- Query expansion (optionally) --------------------
def expand_query_gemini(generator: GeminiEmbeddingGenerator, query: str, n: int = 2) -> List[str]:
    """
    Use a lightweight prompt to paraphrase query n times.
    Cheap-ish and usually helps recall; keep n small (1-3).
    """
    prompt = (
        "Paraphrase the following search query into concise alternatives "
        f"(return exactly {n} bullet points, no explanations):\n\n{query}"
    )
    try:
        # Use a text model via genai for quick paraphrases
        # If you want fully offline, remove this; but it's usually worth it.
        genai.configure(api_key=generator.api_key)
        model = genai.GenerativeModel(model_name=os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-flash"))
        generator._rate_limit()
        resp = model.generate_content(prompt)
        text = resp.text or ""
        alts = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        alts = [a for a in alts if a and a.lower() != query.lower()]
        if len(alts) > n:
            alts = alts[:n]
        return alts
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return []

# -------------------- Engine --------------------
class DocumentSearchEngine:
    """
    Retrieval engine with:
    - Pinecone/FAISS backends
    - Chunking + embedding + upsert
    - Query expansion + RRF
    - MMR re-ranking
    """
    def __init__(
        self,
        gemini_api_key: str,
        backend: str = os.getenv("VECTOR_BACKEND", "pinecone").lower(),
        pinecone_index: Optional[str] = os.getenv("PINECONE_INDEX"),
        pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY"),
        pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default"),
        faiss_index_type: str = os.getenv("FAISS_INDEX_TYPE", "hnsw"),
    ):
        self.emb = GeminiEmbeddingGenerator(gemini_api_key)
        self.backend = backend
        self.vs: Optional[VectorStore] = None
        self._pinecone_kwargs = dict(
            api_key=pinecone_api_key,
            index_name=pinecone_index,
            dim=None,  # filled after first embedding
            metric="cosine",
            namespace=pinecone_namespace,
        )
        self._faiss_index_type = faiss_index_type

    # ---------- building ----------
    def build(self, clauses: List[Clause], batch_log_every=50):
        if not clauses:
            raise ValueError("No clauses provided")

        # generate embeddings
        texts = [c.text for c in clauses]
        vecs = self.emb.embed_many(texts, use_cache=True, batch_log_every=batch_log_every)
        # store back + normalize
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs = l2_normalize(vecs)
        for c, v in zip(clauses, vecs):
            c.embedding = v

        dim = vecs.shape[1]
        # init vector store
        if self.backend == "pinecone":
            kwargs = dict(self._pinecone_kwargs)
            if not kwargs["api_key"] or not kwargs["index_name"]:
                raise RuntimeError("Set PINECONE_API_KEY and PINECONE_INDEX for Pinecone backend")
            kwargs["dim"] = dim
            self.vs = PineconeVectorStore(**kwargs)
        elif self.backend == "faiss":
            if faiss is None:
                raise RuntimeError("faiss not installed")
            self.vs = FaissVectorStore(dim=dim, index_type=self._faiss_index_type)
        else:
            raise ValueError("backend must be 'pinecone' or 'faiss'")

        # upsert
        ids = [c.id for c in clauses]
        metas = [c.metadata or {} for c in clauses]
        self.vs.upsert(ids, vecs, metas)
        logger.info(f"✅ Indexed {len(clauses)} chunks via {self.backend.upper()}")

    # ---------- search ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_expansion: bool = True,
        mmr_lambda: float = 0.6
    ) -> List[Tuple[str, float, Dict]]:
        if not self.vs:
            raise RuntimeError("Vector store is not ready. Call build() first.")
        query = (query or "").strip()
        if not query:
            return []

        q_vec = self.emb.embed_one(query)
        # base results
        base = self.vs.query(q_vec, top_k=top_k * 3)  # overfetch for MMR

        lists_for_rrf = [base]

        # expansion (very small n for latency)
        if use_expansion:
            expansions = expand_query_gemini(self.emb, query, n=2)
            for ex in expansions:
                evec = self.emb.embed_one(ex)
                lists_for_rrf.append(self.vs.query(evec, top_k=top_k * 2))

        fused = rrf_merge(lists_for_rrf, top_k=top_k * 3)

        # MMR on fused set
        if not fused:
            return []
        cand_ids = [cid for cid, _, _ in fused]
        cand_meta = [m for _, _, m in fused]

        # get their vectors back when FAISS (we have them); when Pinecone, approximate by re-embedding text not available.
        # To keep it backend-agnostic and fast, we compute MMR on the *scores* via a proxy:
        # Here we re-query all candidates with the same query vector to fetch scores (already have).
        # If you want true MMR on vectors with Pinecone, store vectors in metadata (large) or round-trip to a local cache.
        # We'll approximate MMR using only scores diversity (fallback).
        # Better: run a small text-level redundancy penalty if you store 'text' in metadata.
        # If your metadata includes 'text', we can use embedding similarity among candidate texts:
        cand_texts = [m.get("text") or "" for m in cand_meta]
        if any(cand_texts):
            cand_vecs = self.emb.embed_many(cand_texts, use_cache=True)
            cand_vecs = np.asarray(cand_vecs, dtype=np.float32)
            sel_idx = mmr(q_vec, cand_vecs, lambda_mult=mmr_lambda, top_k=top_k)
            final = [(cand_ids[i], cosine_similarity(q_vec, cand_vecs[i]), cand_meta[i]) for i in sel_idx]
        else:
            # fallback: return fused top_k
            final = fused[:top_k]

        return final

# -------------------- Public helpers --------------------
def create_search_engine() -> DocumentSearchEngine:
    return DocumentSearchEngine(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        backend=os.getenv("VECTOR_BACKEND", "pinecone").lower()
    )

def extract_and_create_clauses(document_text: str, doc_id: str = "doc") -> List[Clause]:
    # smarter chunks vs single sentences -> better recall/precision
    chunks = smart_sentence_chunks(document_text, doc_id=doc_id, max_words=140, stride=2)
    # also store full text in metadata for better re-ranking / UI
    for c in chunks:
        meta = c.metadata or {}
        meta["text"] = c.text
        c.metadata = meta
    return chunks

