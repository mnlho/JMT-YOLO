#!/bin/bash
# ============================================
# JMT-YOLO Web Demo 一键启动脚本
# ============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    JMT-YOLO Web Demo 启动中...${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 检查 conda 环境
echo -e "${YELLOW}[1/4] 检查 Conda 环境...${NC}"
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo -e "${RED}错误: 未找到 Conda 安装${NC}"
    exit 1
fi

# 激活环境
conda activate zjh_SSDA_web 2>/dev/null || {
    echo -e "${RED}错误: 无法激活 zjh_SSDA_web 环境${NC}"
    echo -e "${YELLOW}请先创建环境: conda create -n zjh_SSDA_web python=3.8${NC}"
    exit 1
}
echo -e "${GREEN}✓ Conda 环境已激活: zjh_SSDA_web${NC}"

# 检查并安装后端依赖
echo -e "${YELLOW}[2/4] 检查后端依赖...${NC}"
cd "$SCRIPT_DIR"
pip show fastapi > /dev/null 2>&1 || {
    echo -e "${YELLOW}  安装 FastAPI...${NC}"
    pip install fastapi uvicorn python-multipart -q
}
echo -e "${GREEN}✓ 后端依赖已就绪${NC}"

# 检查前端依赖
echo -e "${YELLOW}[3/4] 检查前端依赖...${NC}"
if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo -e "${YELLOW}  安装前端依赖 (首次运行)...${NC}"
    npm install
fi
echo -e "${GREEN}✓ 前端依赖已就绪${NC}"

# 杀掉可能存在的旧进程
echo -e "${YELLOW}[4/4] 清理旧进程...${NC}"
pkill -f "python server.py" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1
echo -e "${GREEN}✓ 进程清理完成${NC}"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}    启动服务...${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 启动后端服务器 (后台运行)
echo -e "${YELLOW}启动后端服务器 (端口 8000)...${NC}"
cd "$SCRIPT_DIR"
nohup python server.py > /tmp/jmt-yolo-backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

# 检查后端是否启动成功
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 后端服务器启动成功 (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}✗ 后端服务器启动失败，请检查日志: /tmp/jmt-yolo-backend.log${NC}"
    cat /tmp/jmt-yolo-backend.log
    exit 1
fi

# 启动前端服务器 (后台运行)
echo -e "${YELLOW}启动前端服务器 (端口 5173)...${NC}"
cd "$SCRIPT_DIR"
nohup npm run dev > /tmp/jmt-yolo-frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 3

# 检查前端是否启动成功
if curl -s http://localhost:5173/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 前端服务器启动成功 (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${YELLOW}! 前端服务器可能需要更长时间启动...${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}    🎉 JMT-YOLO Web Demo 启动完成!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "  ${GREEN}前端地址:${NC} http://localhost:5173"
echo -e "  ${GREEN}后端地址:${NC} http://localhost:8000"
echo ""
echo -e "  ${YELLOW}日志文件:${NC}"
echo -e "    后端: /tmp/jmt-yolo-backend.log"
echo -e "    前端: /tmp/jmt-yolo-frontend.log"
echo ""
echo -e "  ${YELLOW}停止服务:${NC} ./stop.sh 或 pkill -f 'server.py|vite'"
echo ""

# 保存 PID 供 stop.sh 使用
echo "$BACKEND_PID" > /tmp/jmt-yolo-backend.pid
echo "$FRONTEND_PID" > /tmp/jmt-yolo-frontend.pid

# 尝试自动打开浏览器
if command -v xdg-open &> /dev/null; then
    sleep 1
    xdg-open http://localhost:5173 2>/dev/null &
elif command -v open &> /dev/null; then
    sleep 1
    open http://localhost:5173 2>/dev/null &
fi

echo -e "${GREEN}按 Ctrl+C 退出此脚本 (服务将继续在后台运行)${NC}"
echo ""

# 显示实时日志
tail -f /tmp/jmt-yolo-backend.log /tmp/jmt-yolo-frontend.log
