#!/bin/bash
#===============================================================================
# 清理临时文件 - 删除测试产生的缓存
#===============================================================================
echo "清理临时文件..."

cd "$(dirname "$0")"

# 清理 runtime_data/live 中的临时图片
if [ -d "runtime_data/live" ]; then
    echo "  清理 runtime_data/live..."
    rm -f runtime_data/live/*.jpg
fi

# 清理 runtime_data 中的测试图
rm -f runtime_data/*.jpg 2>/dev/null

# 清理 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 清理 .pyc 文件
find . -name "*.pyc" -delete 2>/dev/null

echo "清理完成!"