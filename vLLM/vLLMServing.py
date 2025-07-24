import asyncio
import os
import time
import logging
import os
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# 프로젝트 루트에서 .env 파일 로드

load_dotenv(os.path.join(Path(__file__).parent.parent, '.env'))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU 설정
VLLM_GPU_DEVICES = os.getenv("VLLM_GPU_DEVICES", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = VLLM_GPU_DEVICES

# 환경변수에서 설정 읽기
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
PORT = int(os.getenv("VLLM_SERVER_PORT"))
HOST = os.getenv("VLLM_SERVER_HOST")

# Pydantic 모델 정의
class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="대화 메시지 목록")
    max_tokens: Optional[int] = Field(default=2048, description="최대 토큰 수")
    temperature: Optional[float] = Field(default=0.7, description="온도 설정")
    top_p: Optional[float] = Field(default=0.9, description="Top-p 설정")
    frequency_penalty: Optional[float] = Field(default=0.0, description="빈도 페널티")
    stream: Optional[bool] = Field(default=False, description="스트리밍 여부")


class SimpleRequest(BaseModel):
    prompt: str = Field(..., description="프롬프트")
    max_tokens: Optional[int] = Field(default=1024, description="최대 토큰 수")
    temperature: Optional[float] = Field(default=0.7, description="온도 설정")
    top_p: Optional[float] = Field(default=0.9, description="Top-p 설정")
    frequency_penalty: Optional[float] = Field(default=0.0, description="빈도 페널티")


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class SimpleResponse(BaseModel):
    choices: List[Dict[str, Any]]
    text: str
    generated_text: str
    model: str
    created: int


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


# 전역 변수
llm_engine: Optional[AsyncLLMEngine] = None


async def create_llm_engine():
    """vLLM 엔진 초기화"""
    global llm_engine
    
    try:
        logger.info(f"LLM 엔진 초기화 시작 - 모델: {MODEL_NAME}")
        
        # AsyncEngineArgs 설정
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            dtype="auto",
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=False,
            disable_log_requests=True,
            tokenizer_mode="auto",
            load_format="auto",
        )
        
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"LLM 엔진이 성공적으로 로드되었습니다: {MODEL_NAME}")
        
    except Exception as e:
        logger.error(f"LLM 엔진 초기화 실패: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 라이프사이클 관리"""
    # 시작 시 실행
    try:
        await create_llm_engine()
        logger.info("서버가 성공적으로 시작되었습니다")
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        raise
    
    yield
    
    # 종료 시 실행
    logger.info("서버가 종료됩니다")


# FastAPI 앱 생성
app = FastAPI(
    title="vLLM 한국어 챗봇 API",
    description="vLLM과 FastAPI를 사용한 한국어 챗봇 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """메시지를 한국어 대화 형식의 프롬프트로 변환"""
    prompt_parts = []
    
    # 시스템 메시지가 있으면 먼저 추가
    system_msg = None
    for message in messages:
        if message.role == "system":
            system_msg = message.content
            break
    
    if system_msg:
        prompt_parts.append(f"[시스템] {system_msg}")
        prompt_parts.append("")
    
    # 대화 기록 추가
    for message in messages:
        if message.role == "user":
            prompt_parts.append(f"사용자: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"어시스턴트: {message.content}")
    
    # 마지막에 어시스턴트 응답 유도
    prompt_parts.append("어시스턴트:")
    
    return "\n".join(prompt_parts)


def clean_response_text(text: str) -> str:
    """응답 텍스트 정리"""
    # 불필요한 프롬프트 부분 제거
    text = text.strip()
    
    # 응답에서 '어시스턴트:', '사용자:' 등 제거
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('사용자:', '어시스턴트:', 'User:', 'Assistant:')):
            cleaned_lines.append(line)
        elif line.startswith('어시스턴트:'):
            # '어시스턴트:' 이후의 텍스트만 추출
            content = line[5:].strip()
            if content:
                cleaned_lines.append(content)
    
    return '\n'.join(cleaned_lines).strip()


async def generate_stream_response(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성"""
    try:
        async for request_output in llm_engine.generate(prompt, sampling_params, request_id):
            for output in request_output.outputs:
                if output.text:
                    cleaned_text = clean_response_text(output.text)
                    if cleaned_text:
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=int(time.time()),
                            model=MODEL_NAME,
                            choices=[{
                                "index": 0,
                                "delta": {"content": cleaned_text},
                                "finish_reason": None
                            }]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
        
        # 스트림 종료 신호
        final_chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"스트리밍 응답 생성 중 오류: {e}")
        error_chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[{
                "index": 0,
                "delta": {"content": f"오류가 발생했습니다: {str(e)}"},
                "finish_reason": "error"
            }]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


def create_sampling_params(request):
    """개선된 샘플링 파라미터 생성"""
    return SamplingParams(
        temperature=max(0.1, request.temperature),  # 최소값 보장
        top_p=request.top_p,
        max_tokens=min(request.max_tokens, 512),  # 최대 토큰 제한
        repetition_penalty=1.1,  # 반복 방지
        frequency_penalty=max(0.1, request.frequency_penalty),  # 빈도 페널티 최소값
        presence_penalty=0.1,  # 존재 페널티 추가
        stop=["사용자:", "User:", "\n\n사용자", "\n\nUser", "질문:", "\n\n질문", "답변:", "\n답변:", "어시스턴트:", "\n어시스턴트:"],
        skip_special_tokens=True
    )

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "vLLM 한국어 챗봇 API가 실행 중입니다",
        "model": MODEL_NAME,
        "endpoints": [
            "/v1/chat/completions",
            "/health"
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 호환 챗 완성 API"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM 엔진이 아직 로드되지 않았습니다")
    
    try:
        # 메시지를 프롬프트로 변환
        prompt = format_messages_to_prompt(request.messages)
        logger.info(f"생성된 프롬프트: {prompt[:200]}...")
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["사용자:", "User:", "\n\n사용자", "\n\nUser"]
        )
        
        request_id = f"chatreq-{int(time.time())}-{hash(prompt) % 10000}"
        
        
        return StreamingResponse(
            generate_stream_response(prompt, sampling_params, request_id),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
            
    except Exception as e:
        logger.error(f"챗 완성 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"응답 생성 중 오류 발생: {str(e)}")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM 엔진이 아직 로드되지 않았습니다")
    
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "engine_loaded": llm_engine is not None,
        "timestamp": int(time.time())
    }


if __name__ == "__main__":
    # 서버 실행
    logger.info(f"서버 시작: {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        workers=1,
        log_level="info"
    )