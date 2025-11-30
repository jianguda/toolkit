#!/bin/bash
# Streamlit Cloud 构建脚本
# 用于在部署时构建前端组件

set -e

echo "🔨 开始构建前端组件..."

# 检查 Node.js 是否安装
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，无法构建前端组件"
    exit 1
fi

# 检查 npm 是否安装
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，无法构建前端组件"
    exit 1
fi

# 进入前端目录
cd lm_lens/components/frontend

# 安装依赖
echo "📦 安装前端依赖..."
npm install --legacy-peer-deps

# 构建前端组件
echo "🏗️  构建前端组件..."
npm run build

# 检查构建结果
if [ -d "build" ]; then
    echo "✅ 前端组件构建成功！"
    echo "📁 构建目录: $(pwd)/build"
else
    echo "❌ 前端组件构建失败：build 目录不存在"
    exit 1
fi

