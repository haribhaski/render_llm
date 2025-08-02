import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import hashlib
import os
import json
import time
from dataclasses import dataclass
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import nltk
from dotenv import load_dotenv
load_dotenv()

# Download NLTK data for sentence tokenization
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Clause:
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class GeminiEmbeddingGenerator:
    """
    Handles embedding generation using Google Gemini Pro API
    """
    
    def __init__(self, api_key: str, cache_dir: str = os.getenv("EMBEDDINGS_CACHE_DIR", "./embeddings_cache")):
        """
        Initialize Gemini embedding generator
        
        Args:
            api_key: Google Gemini Pro API key
            cache_dir: Directory to cache embeddings
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.embedding_model = 'models/embedding-001'
        
        # For caching
        self._cache_lock = threading.Lock()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("✅ Gemini Pro embedding generator initialized")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"gemini_{text}".encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with self._cache_lock:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for single text using Gemini Pro
        
        Args:
            text: Input text
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(768, dtype=np.float32)  # Default dimension
        
        text = text.strip()
        
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(text)
            if cached is not None:
                return cached.astype(np.float32)
        
        try:
            self._rate_limit()
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            if use_cache:
                self._save_to_cache(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10, use_cache: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with caching and rate limiting
        
        Args:
            texts: List of input texts
            batch_size: Batch size for API calls
            use_cache: Whether to use caching
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        logger.info(f"🔍 Checking cache for {len(texts)} texts...")
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append(np.zeros(768, dtype=np.float32))
                continue
                
            text = text.strip()
            
            if use_cache:
                cached = self._load_from_cache(text)
                if cached is not None:
                    embeddings.append(cached.astype(np.float32))
                    continue
            
            embeddings.append(None)
            texts_to_process.append(text)
            indices_to_process.append(i)
        
        if texts_to_process:
            logger.info(f"🚀 Generating {len(texts_to_process)} embeddings using Gemini Pro...")
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i + batch_size]
                batch_indices = indices_to_process[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts_to_process)-1)//batch_size + 1}")
                
                for j, (text, orig_idx) in enumerate(zip(batch_texts, batch_indices)):
                    try:
                        embedding = self.generate_embedding(text, use_cache=False)
                        embeddings[orig_idx] = embedding
                        if use_cache:
                            self._save_to_cache(text, embedding)
                    except Exception as e:
                        logger.error(f"Failed to process text {orig_idx}: {e}")
                        embeddings[orig_idx] = np.zeros(768, dtype=np.float32)
                
                time.sleep(0.5)
        
        logger.info("✅ Batch embedding generation completed")
        return embeddings

class EnhancedFAISSManager:
    """
    Enhanced FAISS manager with better indexing and search capabilities
    """
    
    def __init__(self, embedding_dim: int = 768, index_type: str = "flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type.lower()
        self.index = None
        self.clauses = []
        
    def create_index(self, index_type: Optional[str] = None) -> faiss.Index:
        """Create FAISS index"""
        if index_type:
            self.index_type = index_type.lower()
            
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            nlist = min(100, max(10, len(self.clauses) // 10))
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"📊 Created {self.index_type.upper()} index (dim: {self.embedding_dim})")
        return index
    
    def build_index(self, clauses: List[Clause]) -> faiss.Index:
        """Build FAISS index from clauses"""
        if not clauses:
            raise ValueError("No clauses provided")
        
        valid_clauses = [c for c in clauses if c.embedding is not None and c.embedding.size > 0]
        
        if not valid_clauses:
            raise ValueError("No valid embeddings found")
        
        self.clauses = valid_clauses
        self.index = self.create_index()
        
        embeddings = np.array([c.embedding for c in valid_clauses], dtype=np.float32)
        
        logger.info(f"🏗️ Building index with {len(valid_clauses)} clauses...")
        
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("🎯 Training index...")
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        
        logger.info(f"✅ Index built! Total vectors: {self.index.ntotal}")
        return self.index
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Clause, float]]:
        """Search for similar clauses"""
        if self.index is None:
            raise ValueError("No index built")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.clauses):
                continue
            
            clause = self.clauses[idx]
            similarity = 1.0 / (1.0 + dist)
            results.append((clause, similarity))
        
        return results
    
    def save_index(self, index_path: str, clauses_path: str):
        """Save index and clauses"""
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, index_path)
        with open(clauses_path, 'wb') as f:
            pickle.dump(self.clauses, f)
        
        logger.info(f"💾 Saved index to {index_path} and clauses to {clauses_path}")
    
    def load_index(self, index_path: str, clauses_path: str):
        """Load index and clauses"""
        self.index = faiss.read_index(index_path)
        with open(clauses_path, 'rb') as f:
            self.clauses = pickle.load(f)
        
        logger.info(f"📂 Loaded index and {len(self.clauses)} clauses")

class DocumentSearchEngine:
    """
    Complete document search engine for your FastAPI integration
    """
    
    def __init__(self, gemini_api_key: str, index_type: str = "flat", cache_dir: str = os.getenv("EMBEDDINGS_CACHE_DIR", "./embeddings_cache")):
        self.embedding_generator = GeminiEmbeddingGenerator(gemini_api_key, cache_dir)
        self.faiss_manager = EnhancedFAISSManager(embedding_dim=768, index_type=index_type)
        
    def process_clauses_with_embeddings(self, clauses: List[Clause], batch_size: int = 10) -> List[Clause]:
        """Generate embeddings for clauses"""
        clauses_without_embeddings = [c for c in clauses if c.embedding is None]
        
        if clauses_without_embeddings:
            texts = [c.text for c in clauses_without_embeddings]
            embeddings = self.embedding_generator.generate_embeddings_batch(texts, batch_size)
            
            for clause, embedding in zip(clauses_without_embeddings, embeddings):
                clause.embedding = embedding
        
        return clauses
    
    def build_search_index(self, clauses: List[Clause], batch_size: int = 10) -> faiss.Index:
        """Build complete search index"""
        processed_clauses = self.process_clauses_with_embeddings(clauses, batch_size)
        return self.faiss_manager.build_index(processed_clauses)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Clause, float]]:
        """Search for relevant clauses"""
        if not query.strip():
            return []
        
        query_embedding = self.embedding_generator.generate_embedding(query)
        return self.faiss_manager.search(query_embedding, top_k)

def create_gemini_search_engine(api_key: str) -> DocumentSearchEngine:
    """Create search engine with Gemini Pro"""
    return DocumentSearchEngine(api_key)

def extract_and_create_clauses(document_text: str, doc_id: str = "doc") -> List[Clause]:
    """Convert document text to Clause objects using sentence-based splitting"""
    sentences = nltk.sent_tokenize(document_text)
    clauses = []
    
    for i, sentence in enumerate(sentences, 1):
        if sentence.strip():
            clause = Clause(
                id=f"{doc_id}_{i}",
                text=sentence,
                metadata={"document_id": doc_id, "clause_number": i}
            )
            clauses.append(clause)
    
    return clauses

def generate_embedding(text: str, search_engine: DocumentSearchEngine) -> np.ndarray:
    """Generate embedding using Gemini"""
    return search_engine.embedding_generator.generate_embedding(text)

def build_faiss_index(clauses: List[Clause], search_engine: DocumentSearchEngine) -> faiss.Index:
    """Build FAISS index"""
    return search_engine.build_search_index(clauses)

def semantic_search(search_engine: DocumentSearchEngine, query: str, top_k: int = 3) -> List[Clause]:
    """Semantic search"""
    results = search_engine.search(query, top_k)
    return [clause for clause, score in results]