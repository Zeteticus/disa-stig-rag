#!/bin/bash
# Production-ready RHEL STIG RAG system with automatic data loading
# and improved embedding model for semantic search
# RHEL 8 and 9 support only
set -e

echo "=== Deploying Production-Ready RHEL STIG RAG System (RHEL 8/9 Only) ==="

# 1. Create full requirements.txt with ML dependencies
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
aiofiles==23.2.1
pydantic==2.4.2
numpy<2.0.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
torch==2.0.1
transformers==4.30.2
huggingface_hub==0.16.4
EOF

# 2. Create the advanced app.py with real RAG functionality
cat > app.py << 'EOF'
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
EOF

# 3. Create the static directory and polished UI
mkdir -p static

# 4. Create the index.html file in the static directory (RHEL 8/9 only)
cat > static/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RHEL STIG RAG Search</title>
    <style>
        :root {
            --primary-color: #0066cc;
            --primary-dark: #004080;
            --secondary-color: #cc0000;
            --light-gray: #f5f5f5;
            --medium-gray: #e0e0e0;
            --dark-gray: #333333;
            --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            font-family: var(--font-family);
            margin: 0;
            padding: 0;
            background-color: var(--light-gray);
            color: var(--dark-gray);
        }
        
        .header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        
        .header p {
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .system-info {
            font-size: 0.8rem;
            margin-top: 0.5rem;
            opacity: 0.8;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        .search-container {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .search-box {
            display: flex;
            gap: 10px;
        }
        
        .search-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid var(--medium-gray);
            border-radius: 4px;
            font-size: 1rem;
            font-family: var(--font-family);
        }
        
        .search-input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
        }
        
        .search-button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .search-button:hover {
            background-color: var(--primary-dark);
        }
        
        .filter-options {
            display: flex;
            gap: 16px;
            margin-top: 1rem;
        }
        
        .filter-options select, .filter-options input {
            padding: 8px 12px;
            border: 1px solid var(--medium-gray);
            border-radius: 4px;
            background-color: white;
            font-family: var(--font-family);
        }
        
        .results-container {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        
        .result-stats {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            color: #666;
            font-size: 0.9rem;
        }
        
        .result {
            border: 1px solid var(--medium-gray);
            border-radius: 6px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .result:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .result-title {
            margin: 0;
            font-size: 1.3rem;
            color: var(--primary-color);
        }
        
        .result-id {
            font-weight: bold;
        }
        
        .severity {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .severity-high {
            background-color: #ffebee;
            color: #c62828;
        }
        
        .severity-medium {
            background-color: #fff8e1;
            color: #f9a825;
        }
        
        .severity-low {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        
        .result-description {
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        
        .result-section {
            margin-bottom: 1rem;
        }
        
        .result-section h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            color: #555;
        }
        
        .result-section p {
            margin: 0;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .relevance {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            color: #666;
        }
        
        .relevance-bar {
            flex: 1;
            height: 6px;
            background-color: var(--medium-gray);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .relevance-fill {
            height: 100%;
            background-color: var(--primary-color);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .loading-spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: var(--primary-color);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
        
        .no-results {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .error-message {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            display: none;
        }
        
        .advanced-toggle {
            display: inline-block;
            margin-left: 1rem;
            color: var(--primary-color);
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .advanced-options {
            display: none;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--medium-gray);
        }
        
        .option-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .option-row label {
            width: 150px;
            font-size: 0.9rem;
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #666;
            font-size: 0.8rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .search-box {
                flex-direction: column;
            }
            
            .filter-options {
                flex-direction: column;
            }
            
            .result-header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .severity {
                margin-top: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>RHEL STIG RAG Search</h1>
            <p>Search for Security Technical Implementation Guides with semantic search</p>
            <div class="system-info" id="system-info"></div>
        </div>
    </div>
    
    <div class="container">
        <div class="search-container">
            <div class="search-box">
                <input type="text" id="search-input" class="search-input" placeholder="Search for STIG requirements, e.g., 'password complexity requirements'">
                <button id="search-button" class="search-button">Search</button>
            </div>
            <div class="filter-options">
                <div>
                    <label for="rhel-version">RHEL Version:</label>
                    <select id="rhel-version">
                        <option value="9">RHEL 9</option>
                        <option value="8">RHEL 8</option>
                        <option value="all">All Versions</option>
                    </select>
                </div>
                <div>
                    <label for="result-limit">Results:</label>
                    <select id="result-limit">
                        <option value="10">10</option>
                        <option value="15" selected>15</option>
                        <option value="20">20</option>
                        <option value="30">30</option>
                        <option value="50">50</option>
                    </select>
                </div>
                <div class="advanced-toggle" id="advanced-toggle">Show Advanced Options</div>
            </div>
            
            <div class="advanced-options" id="advanced-options">
                <div class="option-row">
                    <label for="threshold">Relevance Threshold:</label>
                    <input type="range" id="threshold" min="0" max="1" step="0.05" value="0.6">
                    <span id="threshold-value">0.6</span>
                </div>
                <div class="option-row">
                    <label for="reload-data">Reload Data:</label>
                    <button id="reload-data" class="search-button">Reload STIG Data</button>
                    <span id="reload-status"></span>
                </div>
            </div>
        </div>
        
        <div id="error-message" class="error-message"></div>
        
        <div id="loading" class="loading">
            <div class="loading-spinner"></div>
            <p>Searching STIG database...</p>
        </div>
        
        <div class="results-container">
            <div class="result-stats">
                <div id="result-count"></div>
                <div id="search-stats"></div>
            </div>
            <div id="results"></div>
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            <p>RHEL STIG RAG Search Engine | Using semantic search technology to find security requirements</p>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('search-input');
            const searchButton = document.getElementById('search-button');
            const rhelVersion = document.getElementById('rhel-version');
            const resultLimit = document.getElementById('result-limit');
            const threshold = document.getElementById('threshold');
            const thresholdValue = document.getElementById('threshold-value');
            const resultsContainer = document.getElementById('results');
            const resultCount = document.getElementById('result-count');
            const searchStats = document.getElementById('search-stats');
            const loading = document.getElementById('loading');
            const errorMessage = document.getElementById('error-message');
            const systemInfo = document.getElementById('system-info');
            const advancedToggle = document.getElementById('advanced-toggle');
            const advancedOptions = document.getElementById('advanced-options');
            const reloadDataButton = document.getElementById('reload-data');
            const reloadStatus = document.getElementById('reload-status');
            
            // Fetch system info on load
            fetchSystemInfo();
            
            // Set up event listeners
            searchButton.addEventListener('click', performSearch);
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
            
            threshold.addEventListener('input', function() {
                thresholdValue.textContent = this.value;
            });
            
            advancedToggle.addEventListener('click', function() {
                if (advancedOptions.style.display === 'block') {
                    advancedOptions.style.display = 'none';
                    advancedToggle.textContent = 'Show Advanced Options';
                } else {
                    advancedOptions.style.display = 'block';
                    advancedToggle.textContent = 'Hide Advanced Options';
                }
            });
            
            reloadDataButton.addEventListener('click', function() {
                reloadStatus.textContent = 'Reloading data...';
                fetch('/api/reload')
                    .then(response => response.json())
                    .then(data => {
                        reloadStatus.textContent = data.message;
                        setTimeout(() => {
                            fetchSystemInfo();
                            reloadStatus.textContent = 'Reload complete';
                        }, 2000);
                    })
                    .catch(error => {
                        reloadStatus.textContent = 'Error reloading data';
                        console.error('Error:', error);
                    });
            });
            
            function fetchSystemInfo() {
                fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        systemInfo.textContent = `Model: ${data.model} | Device: ${data.device} | STIGs: ${data.stigs} | Binary: ${data.binary_quantization ? 'Yes' : 'No'}`;
                    })
                    .catch(error => {
                        systemInfo.textContent = 'System info unavailable';
                        console.error('Error:', error);
                    });
            }
            
            function performSearch() {
                const query = searchInput.value.trim();
                if (!query) return;
                
                // Show loading
                loading.style.display = 'block';
                resultsContainer.innerHTML = '';
                resultCount.textContent = '';
                searchStats.textContent = '';
                errorMessage.style.display = 'none';
                
                // Prepare search parameters
                const searchParams = {
                    question: query,
                    rhel_version: rhelVersion.value,
                    top_k: parseInt(resultLimit.value),
                    threshold: parseFloat(threshold.value)
                };
                
                fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(searchParams),
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server error: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    loading.style.display = 'none';
                    
                    if (data.results.length === 0) {
                        resultCount.textContent = 'No results found';
                        searchStats.textContent = `Search time: ${(data.search_time * 1000).toFixed(0)}ms`;
                        resultsContainer.innerHTML = `
                            <div class="no-results">
                                <p>No matching STIG items found. Try broadening your search or using different keywords.</p>
                            </div>
                        `;
                        return;
                    }
                    
                    resultCount.textContent = `Found ${data.results.length} result${data.results.length !== 1 ? 's' : ''}`;
                    searchStats.textContent = `Search time: ${(data.search_time * 1000).toFixed(0)}ms | Model: ${data.model_name}`;
                    
                    let resultsHTML = '';
                    data.results.forEach(result => {
                        const severityClass = `severity-${result.severity.toLowerCase()}`;
                        const relevancePercent = Math.round(result.relevance_score * 100);
                        
                        resultsHTML += `
                            <div class="result">
                                <div class="result-header">
                                    <h3 class="result-title"><span class="result-id">${result.stig_id}</span>: ${result.title}</h3>
                                    <span class="severity ${severityClass}">${result.severity}</span>
                                </div>
                                <div class="result-description">${result.description}</div>
                                <div class="result-section">
                                    <h4>Check Procedure</h4>
                                    <p>${result.check}</p>
                                </div>
                                <div class="result-section">
                                    <h4>Fix Procedure</h4>
                                    <p>${result.fix}</p>
                                </div>
                                <div class="relevance">
                                    <span>Relevance:</span>
                                    <div class="relevance-bar">
                                        <div class="relevance-fill" style="width: ${relevancePercent}%"></div>
                                    </div>
                                    <span>${relevancePercent}%</span>
                                </div>
                            </div>
                        `;
                    });
                    
                    resultsContainer.innerHTML = resultsHTML;
                })
                .catch(error => {
                    loading.style.display = 'none';
                    errorMessage.style.display = 'block';
                    errorMessage.textContent = `Error: ${error.message}. Please try again.`;
                    console.error('Error:', error);
                });
            }
            
            // Add example searches
            const exampleSearches = [
                "password complexity requirements",
                "firewall configuration",
                "user account management",
                "audit logging",
                "SSH security settings",
                "network security",
                "system access controls",
                "file permissions",
                "authentication requirements"
            ];
            
            // Set random example as placeholder
            const randomExample = exampleSearches[Math.floor(Math.random() * exampleSearches.length)];
            searchInput.placeholder = `Search for STIG requirements, e.g., '${randomExample}'`;
        });
    </script>
</body>
</html>
EOF

# 5. Create a sample STIG data file (RHEL 8/9 only)
cat > stig_data.json << 'EOF'
{
  "RHEL-09-000001": {
    "rhel_version": "9",
    "severity": "high",
    "title": "Password Complexity Requirements",
    "description": "The RHEL 9 operating system must enforce a minimum password length of 15 characters to ensure strong authentication mechanisms.",
    "check": "Verify the operating system enforces a minimum 15-character password length.\n\nCheck for the value of the minimum password length with the following command:\n\n# grep -i minlen /etc/security/pwquality.conf\n\nminlen = 15\n\nIf the value of \"minlen\" is set to less than 15, this is a finding.",
    "fix": "Configure the operating system to enforce a minimum 15-character password length.\n\nAdd or modify the following line in the \"/etc/security/pwquality.conf\" file:\n\nminlen = 15"
  },
  "RHEL-09-000002": {
    "rhel_version": "9",
    "severity": "medium",
    "title": "Firewall Configuration",
    "description": "The RHEL 9 operating system must implement a host-based firewall to protect against unauthorized access and malicious attacks.",
    "check": "Verify that firewalld is installed and running on the system.\n\n# systemctl status firewalld\n\nIf firewalld is not active and enabled, this is a finding.",
    "fix": "Install and enable the firewalld service with the following commands:\n\n# dnf install -y firewalld\n# systemctl enable --now firewalld"
  },
  "RHEL-09-000003": {
    "rhel_version": "9",
    "severity": "medium",
    "title": "Audit Logging Configuration",
    "description": "The RHEL 9 operating system must be configured to generate audit records for successful account access events.",
    "check": "Verify the RHEL 9 operating system generates audit records showing successful account access events.\n\n# grep -i logins /etc/audit/audit.rules\n\nIf the command does not return the following output, this is a finding.\n\n-w /var/log/lastlog -p wa -k logins",
    "fix": "Configure the operating system to generate audit records for successful account access events by adding the following line to /etc/audit/rules.d/audit.rules:\n\n-w /var/log/lastlog -p wa -k logins\n\nRestart the audit daemon to apply the changes:\n\n# systemctl restart auditd"
  },
  "RHEL-09-000004": {
    "rhel_version": "9",
    "severity": "high",
    "title": "SSH Security Configuration",
    "description": "The RHEL 9 operating system must be configured to disable SSH root login to prevent unauthorized access to the system.",
    "check": "Verify the SSH daemon does not permit root login.\n\n# grep -i permitrootlogin /etc/ssh/sshd_config\n\nPermitRootLogin no\n\nIf the \"PermitRootLogin\" keyword is set to \"yes\", this is a finding.",
    "fix": "Configure SSH to prevent root login.\n\nEdit the /etc/ssh/sshd_config file to uncomment or add the line for PermitRootLogin and set it to \"no\":\n\nPermitRootLogin no\n\nRestart the SSH daemon:\n\n# systemctl restart sshd"
  },
  "RHEL-09-000005": {
    "rhel_version": "9",
    "severity": "medium",
    "title": "User Account Management",
    "description": "The RHEL 9 operating system must lock accounts after three unsuccessful logon attempts within a 15-minute time period.",
    "check": "Verify the operating system locks an account after three unsuccessful logon attempts within a period of 15 minutes.\n\n# grep pam_faillock.so /etc/pam.d/password-auth\n\nIf the \"deny\" option is not set to 3 or less, this is a finding.",
    "fix": "Configure the operating system to lock an account after three unsuccessful logon attempts within 15 minutes by adding the following line to /etc/pam.d/system-auth and /etc/pam.d/password-auth:\n\nauth required pam_faillock.so preauth silent deny=3 fail_interval=900 unlock_time=0"
  },
  "RHEL-08-000001": {
    "rhel_version": "8",
    "severity": "high",
    "title": "Password Policy",
    "description": "The RHEL 8 operating system must implement DoD-approved encryption to protect the confidentiality of remote access sessions.",
    "check": "Verify the operating system implements DoD-approved encryption to protect the confidentiality of remote access sessions.\n\n# grep -i ciphers /etc/ssh/sshd_config\n\nCiphers aes256-ctr,aes192-ctr,aes128-ctr\n\nIf any ciphers other than those listed are allowed, this is a finding.",
    "fix": "Configure the operating system to implement DoD-approved encryption by adding or modifying the following line in /etc/ssh/sshd_config:\n\nCiphers aes256-ctr,aes192-ctr,aes128-ctr\n\nRestart the SSH service:\n\n# systemctl restart sshd"
  },
  "RHEL-08-000002": {
    "rhel_version": "8",
    "severity": "medium",
    "title": "File System Mounting",
    "description": "The RHEL 8 operating system must prevent direct root logins.",
    "check": "Verify the operating system prevents direct root logins.\n\n# grep -i securetty /etc/securetty\n\nIf the file exists and is not empty, this is a finding.",
    "fix": "Configure the operating system to prevent direct root logins by removing the /etc/securetty file or ensuring it is empty:\n\n# echo > /etc/securetty\n\nor\n\n# rm /etc/securetty"
  }
}
EOF

# 6. Create the Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py /app/
COPY static/ /app/static/
COPY requirements.txt /app/

# Create directories
RUN mkdir -p /app/data /app/cache

# Install Python dependencies properly
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV CACHE_DIR=/app/cache

# Start the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
EOF

# 7. Build the container
echo "Building production-ready STIG RAG container (RHEL 8/9 only)..."
podman build -t rhel-stig-rag:production-rhel89 .

# 8. Stop any existing container
echo "Stopping old containers..."
podman stop stig-rag 2>/dev/null || true
podman rm stig-rag 2>/dev/null || true

# 8.5. Create local cache directory
mkdir -p cache

# 9. Start the new container
echo "Starting production container..."
podman run -d \
    --name stig-rag \
    -p 8000:8000 \
    -v $PWD/stig_data.json:/app/data/stig_data.json:ro,Z \
    -v $PWD/cache:/app/cache:Z \
    rhel-stig-rag:production-rhel89

echo "Waiting for initialization..."
sleep 10

# 10. Test if it's working
echo "Testing health endpoint:"
curl http://localhost:8000/health
echo -e "\n\nProduction-ready STIG RAG system is now available at http://localhost:8000"
echo -e "\nThe system will:"
echo -e "- Support RHEL 8 and 9 only (as requested)"
echo -e "- Automatically load STIG data at startup"
echo -e "- Use the mxbai-embed-large-v1 model for improved semantic search"
echo -e "- Return up to 15 results by default (configurable)"
echo -e "- Cache embeddings for faster startup"
echo -e "- Use memory-efficient binary quantization"
echo -e "- Provide a production-quality UI with advanced options"
