#!/bin/bash
# ============================================
# JMT-YOLO Web Demo 停止脚本
# ============================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}正在停止 JMT-YOLO Web Demo 服务...${NC}"
echo ""

# 停止后端
if [ -f /tmp/jmt-yolo-backend.pid ]; then
    BACKEND_PID=$(cat /tmp/jmt-yolo-backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID 2>/dev/null
        echo -e "${GREEN}✓ 后端服务器已停止 (PID: $BACKEND_PID)${NC}"
    fi
    rm -f /tmp/jmt-yolo-backend.pid
fi

# 停止前端
if [ -f /tmp/jmt-yolo-frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/jmt-yolo-frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID 2>/dev/null
        echo -e "${GREEN}✓ 前端服务器已停止 (PID: $FRONTEND_PID)${NC}"
    fi
    rm -f /tmp/jmt-yolo-frontend.pid
fi

# 确保所有相关进程都被杀掉
pkill -f "python server.py" 2>/dev/null && echo -e "${GREEN}✓ 清理残留后端进程${NC}" || true
pkill -f "vite" 2>/dev/null && echo -e "${GREEN}✓ 清理残留前端进程${NC}" || true

# 清理日志
rm -f /tmp/jmt-yolo-backend.log /tmp/jmt-yolo-frontend.log

echo ""
echo -e "${GREEN}🛑 所有服务已停止${NC}"
