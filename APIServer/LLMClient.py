import os
from pydantic import BaseModel, Field
import requests
import time
from typing import Iterator
import json
from dotenv import load_dotenv
from pathlib import Path

# 프로젝트 루트에서 .env 파일 로드
load_dotenv(os.path.join(Path(__file__).parent.parent, '.env'))

# GPU 설정은 start_all_services.sh에서 관리
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")


class LLMAPIClient:
    """LLM API 클라이언트 (스트리밍 지원)"""

    def __init__(
        self, 
        base_url=None,
        timeout: int = 3000, 
        max_retries: int = 3):
        
        # 환경 변수에서 포트 읽기
        if base_url is None:
            vllm_port = os.getenv('VLLM_SERVER_PORT', '8000')
            base_url = f"http://localhost:{vllm_port}"
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.max_tokens = 4096
        self.temperature = 0.37
        self.top_p = 0.9

        
        # 기본 헤더 설정
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def call_llm_stream(self, prompt: str) -> Iterator[str]:
        """스트리밍 LLM API 호출"""
        print(prompt)
        request_data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        for attempt in range(self.max_retries):
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=self.timeout,
                stream=True
            )

            if response.status_code == 200:
                content_len = 0
                # 스트리밍 응답 처리
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 'data: ' 제거

                            if data_str.strip() == '[DONE]':
                                break

                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content'][content_len:]
                                    content_len += len(content)
                                    if content:
                                        yield content

                return

            elif response.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            else:
                yield f"LLM 서버 오류 (HTTP {response.status_code})"
                return
