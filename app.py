from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import logging
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stig-rag")

# Print system information
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

# Initialize FastAPI app
app = FastAPI(
    title="RHEL STIG RAG API",
    description="Production-Ready RAG System for RHEL STIG Data",
    version="3.0",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    rhel_version: Optional[str] = "9"
    top_k: Optional[int] = 15
    threshold: Optional[float] = 0.6

class STIGItem(BaseModel):
    stig_id: str
    title: str
    description: str
    severity: str
    check: str
    fix: str
    rhel_version: str
    relevance_score: float

class QueryResponse(BaseModel):
    question: str
    rhel_version: str
    results: List[STIGItem]
    count: int
    search_time: float
    model_name: str

class SystemStatus(BaseModel):
    status: str
    stigs: int
    has_index: bool
    model: str
    device: str
    cuda_available: bool
    initialization_time: float
    embedding_dimension: int
    binary_quantization: bool

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled exception: {str(exc)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": error_msg},
    )

class STIGSearchEngine:
    def __init__(self):
        """Initialize the STIG search engine with optimized embedding model."""
        self.model = None
        self.index = None
        self.stig_data = []
        self.embeddings = None
        
        # Configuration
        self.primary_model = "mixedbread-ai/mxbai-embed-large-v1"
        self.fallback_model = "all-MiniLM-L6-v2"
        self.model_name = self.primary_model
        self.embedding_dim = 512  # Using MRL optimization for primary model
        self.use_binary = True  # Use binary quantization for efficiency
        self.clean_cache = os.environ.get('CLEAN_CACHE', '0') == '1'
        self.load_on_init = True
        self.initialization_time = 0
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU detected, using CPU")
            
        # Load data during initialization
        if self.load_on_init:
            start_time = time.time()
            self.load_data()
            self.initialization_time = time.time() - start_time
            logger.info(f"Initialization completed in {self.initialization_time:.2f} seconds")

    def load_data(self):
    """Load STIG data and create/load embeddings with cache handling."""
    # Quick fix: disable caching if environment variable is set
    disable_cache = os.environ.get('DISABLE_CACHE', '0') == '1'

    if disable_cache:
        logger.info("Cache disabled by environment variable")
        # Skip all cache loading/saving logic and go straight to data loading
        # ... proceed with data loading and embedding creation
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove cache file: {e}")

        # Try to load from cache
        if cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Validate cache data
                required_keys = ['embeddings', 'processed_data', 'index', 'model_name', 'embedding_dim']
                if not all(key in cache_data for key in required_keys):
                    raise ValueError(f"Cache file is missing required data. Found keys: {list(cache_data.keys())}")
                
                # Check if the cache matches our current configuration
                if cache_data['model_name'] != self.model_name or cache_data['embedding_dim'] != self.embedding_dim:
                    logger.warning(f"Cache configuration mismatch: {cache_data['model_name']} vs {self.model_name}, {cache_data['embedding_dim']} vs {self.embedding_dim}")
                    raise ValueError("Cache configuration mismatch")
                    
                self.embeddings = cache_data['embeddings']
                self.stig_data = cache_data['processed_data']
                self.index = faiss.deserialize_index(cache_data['index'])
                logger.info(f"Loaded {len(self.stig_data)} STIGs from cache")
                return
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                logger.info("Will create new embeddings")
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                        logger.info(f"Removed invalid cache file: {cache_file}")
                    except:
                        pass

        # If we get here, we need to load raw data and create embeddings
        
        # Find and load STIG data files
        data_files = list(data_dir.glob("*.json"))
        if not data_files:
            logger.error(f"No STIG data files found in {data_dir}. Loading fallback data.")
            # Create a very minimal fallback dataset
            self.stig_data = [
                {
                    "stig_id": "FALLBACK-000001",
                    "title": "Fallback STIG Item",
                    "description": "This is a fallback STIG item used when no data files are found.",
                    "severity": "medium",
                    "check": "This is a fallback check procedure.",
                    "fix": "This is a fallback fix procedure.",
                    "rhel_version": "9",
                    "search_text": "Fallback STIG Item"
                }
            ]
            texts = ["Fallback STIG Item"]
        else:
            # Load data from all JSON files
            raw_data = {}
            for file_path in data_files:
                logger.info(f"Loading STIG data from {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        raw_data.update(file_data)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
            
            logger.info(f"Loaded {len(raw_data)} STIG items from {len(data_files)} files")
            
            # Process data
            self.stig_data = []
            texts = []
            
            for stig_id, info in raw_data.items():
                # Create search text
                search_text = f"{stig_id} {info.get('title', '')} {info.get('description', '')}"
                
                # Process and add to dataset
                self.stig_data.append({
                    "stig_id": stig_id,
                    "rhel_version": info.get("rhel_version", "9"),
                    "severity": info.get("severity", "medium"),
                    "title": info.get("title", ""),
                    "description": info.get("description", ""),
                    "check": info.get("check", ""),
                    "fix": info.get("fix", ""),
                    "search_text": search_text
                })
                texts.append(search_text)

        # Initialize the model
        logger.info(f"Loading sentence transformer model {self.model_name} on {self.device}")
        try:
            # Try to load the primary model
            if self.embedding_dim and self.model_name == "mixedbread-ai/mxbai-embed-large-v1":
                self.model = SentenceTransformer(self.model_name, truncate_dim=self.embedding_dim, device=self.device)
            else:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            # Fall back to the backup model
            self.model_name = self.fallback_model
            self.embedding_dim = None  # No MRL for fallback
            logger.info(f"Falling back to {self.model_name}")
            
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Successfully loaded fallback model")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                # Last resort - try on CPU
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded fallback model on CPU")

        # Create embeddings
        logger.info(f"Creating embeddings for {len(texts)} texts")
        batch_size = 16 if self.device == "cuda" else 32
        self.embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        # Get the actual embedding dimension
        self.embedding_dim = self.embeddings.shape[1]
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Convert to binary if requested
        if self.use_binary:
            logger.info("Converting embeddings to binary representation")
            self.embeddings = self._quantize_to_binary(self.embeddings)
            
            # Create binary index
            dimension = self.embeddings.shape[1] * 8  # Binary packing
            self.index = faiss.IndexBinaryFlat(dimension)
            self.index.add(self.embeddings)
        else:
            # Create standard FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))

        # Save cache
        logger.info(f"Saving cache to {cache_file}")
        try:
            # Create temporary file first
            temp_cache_file = cache_file.with_suffix('.tmp')
            with open(temp_cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'processed_data': self.stig_data,
                    'index': faiss.serialize_index(self.index),
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim
                }, f)
                
            # Rename temp file to final cache file
            temp_cache_file.rename(cache_file)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            # Clean up temp file if it exists
            try:
                if temp_cache_file.exists():
                    temp_cache_file.unlink()
            except:
                pass
        
        logger.info(f"Search engine initialized with {len(self.stig_data)} STIG entries")

    def _quantize_to_binary(self, embeddings):
        """Convert float embeddings to binary format for memory efficiency."""
        # Create binary vectors where values > 0 are 1, and values <= 0 are 0
        binary_vectors = (embeddings > 0).astype(np.uint8)
        
        # Pack 8 binary values into each byte
        num_vectors = binary_vectors.shape[0]
        num_dimensions = binary_vectors.shape[1]
        
        # Ensure dimensions are a multiple of 8
        padded_dimensions = ((num_dimensions + 7) // 8) * 8
        if padded_dimensions != num_dimensions:
            padding = np.zeros((num_vectors, padded_dimensions - num_dimensions), dtype=np.uint8)
            binary_vectors = np.hstack([binary_vectors, padding])
        
        # Reshape and pack bits into bytes
        binary_vectors = binary_vectors.reshape(num_vectors, -1, 8)
        packed_vectors = np.zeros((num_vectors, binary_vectors.shape[1]), dtype=np.uint8)
        
        # Pack bits using bit operations
        for i in range(8):
            packed_vectors |= binary_vectors[:, :, i] << i
            
        return packed_vectors

    def search(self, query: str, rhel_version: str = "9", top_k: int = 15, threshold: float = 0.6) -> List[Dict]:
        """
        Perform semantic search on STIG data.
        
        Args:
            query: The search query
            rhel_version: RHEL version to filter results ("all" for no filtering)
            top_k: Maximum number of results to return
            threshold: Minimum relevance score threshold (0-1)
            
        Returns:
            List of matching STIG items with relevance scores
        """
        start_time = time.time()
        
        # Make sure we have a model
        if not self.model:
            logger.warning("Model not loaded, initializing now")
            try:
                if self.embedding_dim and self.model_name == "mixedbread-ai/mxbai-embed-large-v1":
                    self.model = SentenceTransformer(self.model_name, truncate_dim=self.embedding_dim, device=self.device)
                else:
                    self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model_name = self.fallback_model
                self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Add the special prompt for mixedbread models
        search_query = query
        if self.model_name == "mixedbread-ai/mxbai-embed-large-v1":
            search_query = f"Represent this sentence for searching relevant passages: {query}"
            
        # Create query embedding
        query_embedding = self.model.encode([search_query])
        
        # Convert to binary if using binary quantization
        if self.use_binary:
            query_embedding = self._quantize_to_binary(query_embedding)
        
        # Perform search - request more results than needed for filtering
        search_top_k = max(50, top_k * 3)  # Get extra results for filtering
        D, I = self.index.search(
            query_embedding.astype('float32') if not self.use_binary else query_embedding, 
            search_top_k
        )
        
        # Process results
        results = []
        for idx, distance in zip(I[0], D[0]):
            if idx < len(self.stig_data):
                entry = self.stig_data[idx]
                
                # Filter by RHEL version if specified
                if rhel_version != "all" and entry["rhel_version"] != rhel_version:
                    continue
                
                # Calculate relevance score - normalize distance to 0-1 range
                # Binary and L2 distance need different scaling
                if self.use_binary:
                    # For binary, distance is Hamming distance
                    # Lower is better, max distance is embedding dimension
                    max_distance = self.embedding_dim * 8  # bits
                    relevance = 1 - (distance / max_distance)
                else:
                    # For L2, use a decay function
                    relevance = 1 / (1 + distance)
                
                # Apply threshold
                if relevance < threshold:
                    continue
                    
                # Add to results
                results.append({
                    "stig_id": entry["stig_id"],
                    "title": entry["title"],
                    "description": entry["description"],
                    "severity": entry["severity"],
                    "check": entry["check"],
                    "fix": entry["fix"],
                    "rhel_version": entry["rhel_version"],
                    "relevance_score": float(relevance)
                })
                
                # Stop if we have enough results after filtering
                if len(results) >= top_k:
                    break
                    
        # Measure search time
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        
        return results, search_time

# Initialize the search engine at startup
search_engine = STIGSearchEngine()

@app.get("/")
async def read_index():
    """Serve the web interface."""
    static_file = Path("/app/static/index.html")
    if static_file.exists():
        return FileResponse(static_file)
    return {"message": "Web UI not found. API is running at /docs"}

@app.get("/health")
async def health() -> SystemStatus:
    """Get system health and status information."""
    return {
        "status": "healthy",
        "stigs": len(search_engine.stig_data),
        "has_index": search_engine.index is not None,
        "model": search_engine.model_name,
        "device": search_engine.device,
        "cuda_available": torch.cuda.is_available(),
        "initialization_time": search_engine.initialization_time,
        "embedding_dimension": search_engine.embedding_dim,
        "binary_quantization": search_engine.use_binary
    }

@app.post("/api/query")
async def query(request: QueryRequest) -> QueryResponse:
    """Semantic search endpoint."""
    if not search_engine.index:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    # Validate parameters
    if request.top_k < 1:
        request.top_k = 15
    if request.top_k > 50:
        request.top_k = 50
    if request.threshold < 0 or request.threshold > 1:
        request.threshold = 0.6

    # Perform search
    results, search_time = search_engine.search(
        request.question, 
        request.rhel_version, 
        request.top_k,
        request.threshold
    )

    return {
        "question": request.question,
        "rhel_version": request.rhel_version,
        "results": results,
        "count": len(results),
        "search_time": search_time,
        "model_name": search_engine.model_name
    }

@app.get("/api/reload", response_model=dict)
async def reload_data(background_tasks: BackgroundTasks):
    """Reload STIG data and regenerate embeddings (async operation)."""
    background_tasks.add_task(search_engine.load_data)
    return {"message": "Data reload initiated in the background"}

# Mount static files if available
static_dir = Path("/app/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
