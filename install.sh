#!/bin/bash
#===============================================================================
# 螺丝检测系统 - Ubuntu/香橙派 一键部署脚本
# 
# 使用方法:
#   1. 将整个 newls 目录复制到 香橙派
#   2. chmod +x install.sh
#   3. ./install.sh
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}螺丝检测系统 - Ubuntu 部署脚本${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 1. 检测环境
echo -e "${GREEN}[1/7]${NC} 检测系统环境..."
OS=$(cat /etc/os-release | grep '^NAME=' | cut -d'"' -f2)
ARCH=$(uname -m)
echo "  系统: $OS"
echo "  架构: $ARCH"
echo "  项目: $PROJECT_DIR"

# 2. 安装系统依赖
echo -e "${GREEN}[2/7]${NC} 安装系统依赖..."
sudo apt update -qq
sudo apt install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopencv-dev \
    cmake \
    pkg-config

# 3. 创建虚拟环境
echo -e "${GREEN}[3/7]${NC} 创建虚拟环境..."
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate

# 4. 升级 pip
echo -e "${GREEN}[4/7]${NC} 安装 Python 依赖..."
pip install --upgrade pip -q

# 安装核心依赖 (使用清华镜像)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -q \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    opencv-python==4.9.0.80 \
    numpy==1.26.3 \
    pillow==10.2.0 \
    pyyaml==6.0.1 \
    ultralytics==8.1.0

# PyTorch
if [[ "$ARCH" == "aarch64" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 -q
else
    pip install torch==2.1.0 torchvision==0.16.0 -q
fi

# 5. 初始化目录
echo -e "${GREEN}[5/7]${NC} 初始化目录..."
mkdir -p runtime_data/live
mkdir -p runtime_data/data
mkdir -p uploads
mkdir -p models
mkdir -p data
mkdir -p templates
mkdir -p static
mkdir -p cameras

# 检查文件
echo -e "${GREEN}[6/7]${NC} 检查文件..."
REQUIRED="app.py config.py templates/index.html cameras/hik_camera.py cameras/usb_camera.py hik_sdk_adapter.py"
OK=true
for f in $REQUIRED; do
    if [ ! -f "$f" ]; then
        echo -e "  ${RED}缺少: $f${NC}"
        OK=false
    fi
done

if [ "$OK" = true ]; then
    echo -e "  ${GREEN}所有核心文件完整${NC}"
fi

# 6. 测试加载
echo -e "${GREEN}[7/7]${NC} 测试模型加载..."
source venv/bin/activate
python3 -c "from ultralytics import YOLO; print('  ultralytics OK')"
python3 -c "import cv2; print('  opencv OK')"
python3 -c "from app import app; print('  app OK')"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}安装完成!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "启动命令:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python3 app.py"
echo ""
echo "或使用启动脚本:"
echo "  ./start.sh"
echo ""
echo "访问地址: http://localhost:5000"
echo ""