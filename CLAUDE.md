# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SCBProject is a Korean university notice chatbot system with a microservices architecture consisting of:
- vLLM service (port 22222) - Korean-optimized LLM server using A.X 4.0 Light model
- API server (port 8001) - FastAPI server with RAG system for context-aware responses
- Frontend (port 3001) - React TypeScript chat interface

## Architecture

The system implements a RAG (Retrieval-Augmented Generation) pipeline:
1. User queries are embedded using Qwen3-Embedding-8B
2. FAISS vector search retrieves relevant university notices
3. Context is sent to vLLM service for response generation
4. Responses are streamed back to the frontend

Key architectural decisions:
- GPU allocation: vLLM on GPU 2, embeddings on GPU 1
- Streaming responses via Server-Sent Events (SSE)
- FAISS index with 8192-dimension embeddings
- Caching system for embeddings to improve performance

## Common Commands

### Running the System
```bash
# Start all services (vLLM, API server, frontend)
./start_all_services.sh

# Stop all services gracefully
./stop_all_services.sh

# Check service status
ps aux | grep -E "(vllm|api_server|npm)"
```

### Frontend Development
```bash
cd frontend
npm install  # Install dependencies
npm start    # Run development server (port 3001)
npm build    # Build for production
npm test     # Run tests (if any)
```

### Backend Development
```bash
# Run API server manually (requires vLLM service running)
cd APIServer
python api_server.py

# Run vLLM service manually
cd vLLM
./launch_a.x.sh
```

### Testing API Endpoints
```bash
# Health check
curl http://localhost:8001/health

# Chat endpoint (streaming)
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "창원대학교 장학금 안내"}'
```

## Key Code Patterns

### Adding New API Endpoints
New endpoints should be added to `APIServer/api_server.py`. Follow the existing pattern:
- Use FastAPI decorators
- Include proper error handling
- Support streaming responses where appropriate

### Modifying RAG Behavior
RAG system configuration is in `APIServer/RAGSystem.py`:
- `similarity_threshold`: Controls relevance filtering (default: 0.3)
- `k`: Number of similar documents to retrieve (default: 5)
- `process_query()`: Main entry point for RAG queries

### Frontend Chat Integration
Chat functionality is in `frontend/src/components/ChatWidget.tsx`:
- Uses EventSource for SSE streaming
- Implements error handling and reconnection logic
- Maintains conversation history in component state

## Environment Configuration

The `.env` file controls service configuration:
```
VLLM_PORT=22222
API_PORT=8001
FRONTEND_PORT=3001
VLLM_GPU=2
API_GPU=1
```

## Data Management

University notices are stored in `APIServer/data/output.csv` with columns:
- title: Notice title
- content: Full notice content
- reg_date: Registration date
- view_count: Number of views
- notice: Notice ID

Embeddings are cached in `APIServer/cache/` to avoid recomputation.

## Debugging Tips

1. Check service logs in respective directories
2. Monitor GPU usage: `nvidia-smi -l 1`
3. API server logs show streaming chunks and processing times
4. Frontend console shows SSE connection status
5. Use `start_all_services.sh` output to verify service startup order

## Model Information

- **LLM**: A.X 4.0 Light (7B parameters)
  - Max context: 16,384 tokens
  - Optimized for Korean language (KMMLU: 64.15)
  - Path: `/home/jskim/.cache/huggingface/hub/models--hpcai-tech--ax-4.0-light`

- **Embeddings**: Qwen3-Embedding-8B
  - Dimension: 8192
  - Used for semantic search in RAG pipeline