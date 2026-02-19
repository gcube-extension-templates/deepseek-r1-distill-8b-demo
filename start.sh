#!/bin/bash
# ==============================================
#  DeepSeek-R1-Distill-Llama-8B — Start Script
#  실행: bash start.sh
# ==============================================

echo "======================================"
echo "  🧠 DeepSeek-R1-Distill-Llama-8B 시작"
echo "======================================"

# Ollama 서비스 시작
echo ""
echo "[1/2] Ollama 서비스 시작 중..."
ollama serve &
sleep 5
echo "✅ Ollama 실행 중 (port 11434)"

# Open WebUI 시작
echo ""
echo "[2/2] Open WebUI 시작 중..."
echo "      잠시 기다려주세요..."
echo ""

OLLAMA_BASE_URL=http://localhost:11434 open-webui serve --port 8080 &
WEBUI_PID=$!

# Open WebUI가 실제로 뜰 때까지 대기
until curl -s http://localhost:8080 > /dev/null 2>&1; do
    sleep 2
done

echo "======================================"
echo "  🌐 Open WebUI 준비 완료!"
echo "  브라우저에서 아래 주소로 접속하세요:"
echo "  GCUBE 서비스 URL (포트 8080)"
echo "======================================"

# 프로세스 유지
wait $WEBUI_PID