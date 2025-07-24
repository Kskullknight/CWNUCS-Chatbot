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

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}모든 서비스 종료${NC}"
echo -e "${YELLOW}========================================${NC}"

# PID 파일 경로
PID_DIR="/home/jskim/SCBProject/.pids"

# 1. React 서버 종료 (포트 기반)
echo -e "\n${YELLOW}[1/3] React 서버 종료 중...${NC}"
REACT_PID=$(lsof -Pi :${FRONTEND_PORT} -sTCP:LISTEN -t)
if [ ! -z "$REACT_PID" ]; then
    kill $REACT_PID 2>/dev/null
    echo -e "${GREEN}✓ React 서버를 종료했습니다. (PID: $REACT_PID)${NC}"
else
    echo -e "  React 서버가 실행되고 있지 않습니다."
fi

# 2. API 서버 종료
echo -e "\n${YELLOW}[2/3] API 서버 종료 중...${NC}"
# PID 파일 확인
if [ -f "$PID_DIR/api.pid" ]; then
    API_PID=$(cat $PID_DIR/api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo -e "${GREEN}✓ API 서버를 종료했습니다. (PID: $API_PID)${NC}"
    else
        echo -e "  PID 파일의 프로세스가 이미 종료되었습니다."
    fi
    rm $PID_DIR/api.pid
fi

# 포트 기반 확인
API_PID=$(lsof -Pi :${API_PORT} -sTCP:LISTEN -t)
if [ ! -z "$API_PID" ]; then
    kill $API_PID 2>/dev/null
    echo -e "${GREEN}✓ API 서버를 종료했습니다. (포트 ${API_PORT}, PID: $API_PID)${NC}"
fi

# 3. vLLM 서버 종료
echo -e "\n${YELLOW}[3/3] vLLM 서버 종료 중...${NC}"
echo -e "${YELLOW}주의: vLLM 서버를 종료하시겠습니까? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    # PID 파일 확인
    if [ -f "$PID_DIR/vllm.pid" ]; then
        VLLM_PID=$(cat $PID_DIR/vllm.pid)
        if kill -0 $VLLM_PID 2>/dev/null; then
            kill $VLLM_PID
            echo -e "${GREEN}✓ vLLM 서버를 종료했습니다. (PID: $VLLM_PID)${NC}"
        else
            echo -e "  PID 파일의 프로세스가 이미 종료되었습니다."
        fi
        rm $PID_DIR/vllm.pid
    fi
    
    # 포트 기반 확인
    VLLM_PID=$(lsof -Pi :${VLLM_SERVER_PORT} -sTCP:LISTEN -t)
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null
        echo -e "${GREEN}✓ vLLM 서버를 종료했습니다. (포트 ${VLLM_SERVER_PORT}, PID: $VLLM_PID)${NC}"
    fi
else
    echo -e "  vLLM 서버 종료를 건너뜁니다."
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}서비스 종료가 완료되었습니다.${NC}"
echo -e "${GREEN}========================================${NC}"