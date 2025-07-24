from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import AsyncGenerator, Dict, Any
import json
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

# 프로젝트 루트에서 .env 파일 로드
load_dotenv(os.path.join(Path(__file__).parent.parent, '.env'))

from RAGSystem import AdvancedSchoolNoticeRAG
from langchain_core.messages import HumanMessage
import queue

# 환경 변수에서 설정 읽기
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
VLLM_PORT = int(os.getenv("VLLM_SERVER_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3001"))
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "./data/output.csv")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

app = FastAPI()
print(f"VLLM_PORT : {VLLM_PORT}")
print(f"API_PORT : {API_PORT}")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{FRONTEND_PORT}"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 시스템 초기화
rag_system = AdvancedSchoolNoticeRAG("./data/output.csv", EMBEDDING_MODEL)

class ChatRequest(BaseModel):
    message: str

class StreamingChatBotGraph:
    """스트리밍을 위한 ChatBotGraph 래퍼"""
    def __init__(self, chatbot_graph):
        self.graph = chatbot_graph
        self.streaming_queue = queue.Queue()
    
    def streaming_callback(self, chunk: str):
        """스트리밍 콜백 함수"""
        self.streaming_queue.put(chunk)
    
    async def start_streaming_graph(self, query: str) -> Dict[str, Any]:
        """스트리밍 그래프 실행"""
        # 기존 StreamingDisplayManager를 오버라이드
        original_manager = self.graph.streamingDisplayManager
        
        # 임시 스트리밍 매니저 생성
        class TempStreamingManager:
            def __init__(self, callback):
                self.callback = callback
                self.content = ""
            
            def update(self, chunk):
                self.content += chunk
                self.callback(chunk)
            
            def finish(self):
                pass
        
        # 임시 매니저로 교체
        self.graph.streamingDisplayManager = TempStreamingManager(self.streaming_callback)
        
        # 그래프 실행
        result = await self.graph.start_graph(query)
        
        # 원래 매니저로 복원
        self.graph.streamingDisplayManager = original_manager
        
        # 종료 신호
        self.streaming_queue.put(None)
        
        return result

streaming_wrapper = StreamingChatBotGraph(rag_system.graph)

async def generate_streaming_response(query: str) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성"""
    try:
        # 비동기 태스크로 그래프 실행
        result_container = {"result": None}
        
        async def run_graph():
            result_container["result"] = await streaming_wrapper.start_streaming_graph(query)
        
        # 비동기 태스크 생성
        task = asyncio.create_task(run_graph())
        
        # 큐에서 청크를 읽어 스트리밍
        while True:
            try:
                # 비동기적으로 큐 확인
                await asyncio.sleep(0.01)
                
                try:
                    chunk = streaming_wrapper.streaming_queue.get_nowait()
                except queue.Empty:
                    # 태스크가 완료되었는지 확인
                    if task.done():
                        # 남은 청크들을 모두 처리
                        while True:
                            try:
                                chunk = streaming_wrapper.streaming_queue.get_nowait()
                                if chunk is None:
                                    break
                                yield f"data: {json.dumps({'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"
                            except queue.Empty:
                                break
                        break
                    continue
                
                if chunk is None:  # 종료 신호
                    break
                
                # SSE 형식으로 전송
                yield f"data: {json.dumps({'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                print(f"스트리밍 중 에러: {e}")
                continue
        
        # 태스크 완료 대기
        await task
        
        # 완료 신호
        result = result_container["result"]
        yield f"data: {json.dumps({'content': '', 'done': True, 'metadata': result.get('metadata', {})}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트 - SSE 스트리밍"""
    return StreamingResponse(
        generate_streaming_response(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)