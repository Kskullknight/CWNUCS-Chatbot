#!/bin/bash

# 색상 코드 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# .env 파일 로드
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}창원대학교 공지사항 챗봇 통합 서비스 시작${NC}"
echo -e "${GREEN}========================================${NC}"

# PID 파일 경로
PID_DIR="/home/jskim/SCBProject/.pids"
mkdir -p $PID_DIR

# 1. vLLM 서버 시작 (백그라운드)
echo -e "\n${YELLOW}[1/3] vLLM 서버 시작 중...${NC}"

# 기존 vLLM 서버가 실행 중인지 확인
# 먼저 포트가 사용 중인지 확인
if lsof -Pi :${VLLM_SERVER_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ vLLM 서버가 이미 실행 중입니다 (포트 ${VLLM_SERVER_PORT} 사용 중).${NC}"
    # health 체크로 정상 작동 확인
    if curl -s http://localhost:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; then
        echo -e "  서버가 정상적으로 응답하고 있습니다."
    else
        echo -e "  ${YELLOW}주의: 포트는 사용 중이지만 health 체크에 실패했습니다.${NC}"
        echo -e "  기존 프로세스를 확인하거나 포트를 변경하세요."
    fi
else
    echo "vLLM 서버를 시작합니다..."
    cd /home/jskim/SCBProject/vLLM
    
    # GPU 환경 설정
    export CUDA_VISIBLE_DEVICES="${VLLM_GPU_DEVICES}"
    
    # vLLM 서버 실행 (백그라운드)
    nohup python vLLMServing.py > vllm_server.log 2>&1 &
    VLLM_PID=$!
    echo $VLLM_PID > $PID_DIR/vllm.pid
    
    echo "vLLM 서버 시작 대기 중..."
    # 서버가 준비될 때까지 대기 (최대 60초)
    for i in {1..60}; do
        if curl -s http://localhost:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ vLLM 서버가 성공적으로 시작되었습니다. (PID: $VLLM_PID)${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    if ! curl -s http://localhost:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; then
        echo -e "${RED}✗ vLLM 서버 시작 실패. 로그를 확인하세요: vLLL/vllm_server.log${NC}"
        exit 1
    fi
fi

# 2. API 서버 시작 (백그라운드)
echo -e "\n${YELLOW}[2/3] API 서버 시작 중...${NC}"

# 기존 API 서버가 실행 중인지 확인
# 먼저 포트가 사용 중인지 확인
if lsof -Pi :${API_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ API 서버가 이미 실행 중입니다 (포트 ${API_PORT} 사용 중).${NC}"
    # health 체크로 정상 작동 확인
    if curl -s http://localhost:${API_PORT}/api/health > /dev/null 2>&1; then
        echo -e "  서버가 정상적으로 응답하고 있습니다."
    else
        echo -e "  ${YELLOW}주의: 포트는 사용 중이지만 health 체크에 실패했습니다.${NC}"
    fi
else
    cd /home/jskim/SCBProject/APIServer
    
    # GPU 환경 설정 (RAG 시스템용)
    export CUDA_VISIBLE_DEVICES="${API_GPU_DEVICES}"
    
    # API 서버 실행 (백그라운드)
    nohup python api_server.py > api_server.log 2>&1 &
    API_PID=$!
    echo $API_PID > $PID_DIR/api.pid
    
    echo "API 서버 시작 대기 중..."
    # 서버가 준비될 때까지 대기 (최대 30초)
    for i in {1..30}; do
        if curl -s http://localhost:${API_PORT}/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API 서버가 성공적으로 시작되었습니다. (PID: $API_PID)${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    if ! curl -s http://localhost:${API_PORT}/api/health > /dev/null 2>&1; then
        echo -e "${RED}✗ API 서버 시작 실패. 로그를 확인하세요: api_server.log${NC}"
        exit 1
    fi
fi

# 3. React 개발 서버 시작
echo -e "\n${YELLOW}[3/3] React 개발 서버 시작 중...${NC}"
cd /home/jskim/SCBProject/frontend

# npm 패키지가 설치되어 있는지 확인
if [ ! -d "node_modules" ]; then
    echo "npm 패키지를 설치합니다..."
    npm install
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}모든 서비스가 시작되었습니다!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n서비스 상태:"
echo -e "  - vLLM 서버: ${GREEN}http://localhost:${VLLM_SERVER_PORT}${NC}"
echo -e "  - API 서버: ${GREEN}http://localhost:${API_PORT}${NC}"
echo -e "  - 웹 인터페이스: ${GREEN}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "\n종료하려면 Ctrl+C를 누르세요."

# React 서버 실행 (포그라운드)
PORT=${FRONTEND_PORT} npm start

# 종료 시 모든 서버 정리
echo -e "\n${YELLOW}서비스를 종료합니다...${NC}"

# PID 파일에서 프로세스 종료
if [ -f "$PID_DIR/api.pid" ]; then
    API_PID=$(cat $PID_DIR/api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo "API 서버를 종료했습니다."
    fi
    rm $PID_DIR/api.pid
fi

# vLLM 서버는 수동으로 종료하도록 안내
echo -e "\n${YELLOW}주의: vLLM 서버는 다른 작업에서도 사용될 수 있으므로 자동으로 종료되지 않습니다.${NC}"
echo -e "vLLM 서버를 종료하려면 다음 명령을 실행하세요:"
echo -e "  ${GREEN}./stop_vllm_server.sh${NC}"