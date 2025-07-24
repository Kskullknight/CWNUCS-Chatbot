from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator
import json
import sys
import os
from dotenv import load_dotenv

# 프로젝트 루트에서 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAGSystem import AdvancedSchoolNoticeRAG

# 환경 변수에서 설정 읽기
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3001"))
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "./data/output.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

app = FastAPI()

print(f"Simple API Server")
print(f"VLLM_PORT: {VLLM_PORT}")
print(f"API_PORT: {API_PORT}")
print(f"DATA_FILE_PATH: {DATA_FILE_PATH}")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{FRONTEND_PORT}", "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 시스템 초기화
print("RAG 시스템 초기화 중...")
rag_system = AdvancedSchoolNoticeRAG(DATA_FILE_PATH, EMBEDDING_MODEL)
print("RAG 시스템 초기화 완료")

class ChatRequest(BaseModel):
    message: str

async def generate_simple_streaming_response(query: str) -> AsyncGenerator[str, None]:
    """간단한 스트리밍 응답 생성 (동기 버전 사용)"""
    try:
        print(f"쿼리 처리 중: {query}")
        
        # 동기 버전 사용
        result = rag_system.answer_question(query)
        
        print(f"응답 생성 완료")
        print(f"답변 길이: {len(result.get('answer', ''))}")
        
        # 응답을 청크로 나누어 스트리밍
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
        chunk_size = 20
        
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield f"data: {json.dumps({'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)
        
        # 완료 신호
        yield f"data: {json.dumps({'content': '', 'done': True, 'metadata': result.get('metadata', {})}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트 - SSE 스트리밍"""
    print(f"채팅 요청 받음: {request.message}")
    return StreamingResponse(
        generate_simple_streaming_response(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "type": "simple_api_server"}

if __name__ == "__main__":
    import uvicorn
    print(f"Simple API Server 시작: {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)