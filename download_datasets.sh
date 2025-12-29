#!/bin/bash

# 下载训练所需的数据集

set -e

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCHMARK_DIR"

echo "=========================================="
echo "下载训练数据集"
echo "=========================================="

# 创建数据目录
mkdir -p data/squad
mkdir -p data/glue/sst2/SST-2

# 下载 SQuAD v2.0 数据集（用于 BERT 和 RoBERTa）
echo ""
echo "下载 SQuAD v2.0 数据集..."
# 检查文件是否存在且大小大于 0
if [ ! -f "data/squad/train-v2.0.json" ] || [ ! -s "data/squad/train-v2.0.json" ]; then
    if [ -f "data/squad/train-v2.0.json" ] && [ ! -s "data/squad/train-v2.0.json" ]; then
        echo "检测到空文件，删除并重新下载..."
        rm -f data/squad/train-v2.0.json
    fi
    echo "下载 train-v2.0.json (~40MB)..."
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll --timeout=30 --tries=3 \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json \
            -O data/squad/train-v2.0.json
    elif command -v curl &> /dev/null; then
        curl -# -L --max-time 300 --retry 3 \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json \
            -o data/squad/train-v2.0.json
    else
        echo "错误: 需要 wget 或 curl 来下载文件"
        exit 1
    fi
    
    # 验证下载是否成功
    if [ ! -s "data/squad/train-v2.0.json" ]; then
        echo "✗ 下载失败：文件为空"
        exit 1
    fi
    
    file_size=$(stat -f%z "data/squad/train-v2.0.json" 2>/dev/null || stat -c%s "data/squad/train-v2.0.json" 2>/dev/null || echo "0")
    if [ "$file_size" -lt 1000000 ]; then
        echo "✗ 下载失败：文件大小异常 ($file_size bytes)"
        exit 1
    fi
    
    echo ""
    echo "✓ SQuAD train-v2.0.json 下载完成 ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$file_size bytes"))"
else
    file_size=$(stat -f%z "data/squad/train-v2.0.json" 2>/dev/null || stat -c%s "data/squad/train-v2.0.json" 2>/dev/null || echo "0")
    echo "✓ SQuAD train-v2.0.json 已存在 ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$file_size bytes"))"
fi

# 检查文件是否存在且大小大于 0
if [ ! -f "data/squad/dev-v2.0.json" ] || [ ! -s "data/squad/dev-v2.0.json" ]; then
    if [ -f "data/squad/dev-v2.0.json" ] && [ ! -s "data/squad/dev-v2.0.json" ]; then
        echo "检测到空文件，删除并重新下载..."
        rm -f data/squad/dev-v2.0.json
    fi
    echo "下载 dev-v2.0.json (~4MB)..."
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll --timeout=30 --tries=3 \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
            -O data/squad/dev-v2.0.json
    elif command -v curl &> /dev/null; then
        curl -# -L --max-time 300 --retry 3 \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
            -o data/squad/dev-v2.0.json
    else
        echo "错误: 需要 wget 或 curl 来下载文件"
        exit 1
    fi
    
    # 验证下载是否成功
    if [ ! -s "data/squad/dev-v2.0.json" ]; then
        echo "✗ 下载失败：文件为空"
        exit 1
    fi
    
    file_size=$(stat -f%z "data/squad/dev-v2.0.json" 2>/dev/null || stat -c%s "data/squad/dev-v2.0.json" 2>/dev/null || echo "0")
    if [ "$file_size" -lt 100000 ]; then
        echo "✗ 下载失败：文件大小异常 ($file_size bytes)"
        exit 1
    fi
    
    echo ""
    echo "✓ SQuAD dev-v2.0.json 下载完成 ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$file_size bytes"))"
else
    file_size=$(stat -f%z "data/squad/dev-v2.0.json" 2>/dev/null || stat -c%s "data/squad/dev-v2.0.json" 2>/dev/null || echo "0")
    echo "✓ SQuAD dev-v2.0.json 已存在 ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "$file_size bytes"))"
fi

# 下载 GLUE SST-2 数据集（用于 BART 和 GPT2）
echo ""
echo "下载 GLUE SST-2 数据集..."
if [ ! -d "data/glue/sst2/SST-2" ] || [ -z "$(ls -A data/glue/sst2/SST-2 2>/dev/null)" ]; then
    echo "下载 SST-2 (~5MB)..."
    
    # 查找 Python 可执行文件（优先使用虚拟环境）
    PYTHON_CMD=""
    if [ -f ".venv/bin/python" ]; then
        PYTHON_CMD=".venv/bin/python"
        echo "使用虚拟环境中的 Python: $PYTHON_CMD"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "使用系统 Python: $PYTHON_CMD"
    else
        echo "错误: 未找到 Python"
        exit 1
    fi
    
    # 使用 Hugging Face datasets 下载（如果可用）
    if [ -n "$PYTHON_CMD" ]; then
        $PYTHON_CMD << 'EOF'
import os
import sys
try:
    import datasets
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("")
    print("请安装缺失的库:")
    print("  如果使用虚拟环境: source .venv/bin/activate && pip install datasets tqdm")
    print("  或者直接运行: pip install datasets tqdm")
    print("")
    sys.exit(1)

os.makedirs('data/glue/sst2/SST-2', exist_ok=True)

# 启用 datasets 库的进度条和详细输出
import datasets
datasets.config.HF_DATASETS_OFFLINE = False
datasets.config.HF_DATASETS_DISABLE_PROGRESS_BARS = False
datasets.config.HF_DATASETS_VERBOSITY = "info"  # 显示详细信息

print("=" * 60)
print("正在下载 GLUE SST-2 训练集...")
print("=" * 60)
print("(下载进度会显示在下方，请耐心等待...)")
print("")

# load_dataset 会自动显示下载进度条
# 如果数据集已存在则复用，不存在则下载
train_dataset = load_dataset('glue', 'sst2', split='train')
print(f"\n✓ 训练集下载完成，共 {len(train_dataset)} 条数据")

# 保存为 TSV 格式（GLUE 标准格式）
print("正在写入训练集到文件...")
with open('data/glue/sst2/SST-2/train.tsv', 'w', encoding='utf-8') as f:
    f.write('sentence\tlabel\n')
    for item in tqdm(train_dataset, desc="写入训练集", unit="条", ncols=80):
        # 转义制表符和换行符
        sentence = item['sentence'].replace('\t', ' ').replace('\n', ' ')
        f.write(f"{sentence}\t{item['label']}\n")

print("\n" + "=" * 60)
print("正在下载 GLUE SST-2 验证集...")
print("=" * 60)
print("(下载进度会显示在下方，请耐心等待...)")
print("")

val_dataset = load_dataset('glue', 'sst2', split='validation')
print(f"\n✓ 验证集下载完成，共 {len(val_dataset)} 条数据")

print("正在写入验证集到文件...")
with open('data/glue/sst2/SST-2/dev.tsv', 'w', encoding='utf-8') as f:
    f.write('sentence\tlabel\n')
    for item in tqdm(val_dataset, desc="写入验证集", unit="条", ncols=80):
        sentence = item['sentence'].replace('\t', ' ').replace('\n', ' ')
        f.write(f"{sentence}\t{item['label']}\n")

print("\n✓ GLUE SST-2 下载完成")
EOF
    else
        echo ""
        echo "错误: 无法下载 GLUE SST-2 数据集"
        echo ""
        echo "解决方案："
        echo "1. 确保在虚拟环境中: source .venv/bin/activate"
        echo "2. 安装必要的库: pip install datasets tqdm"
        echo "3. 或者手动下载: https://gluebenchmark.com/tasks"
        echo ""
        exit 1
    fi
else
    echo "✓ GLUE SST-2 已存在"
fi

# CIFAR-100 会自动下载（通过 torchvision）
echo ""
echo "=========================================="
echo "数据集下载完成！"
echo "=========================================="
echo ""
echo "数据集位置:"
echo "  SQuAD:     $BENCHMARK_DIR/data/squad/"
echo "  GLUE SST-2: $BENCHMARK_DIR/data/glue/sst2/SST-2/"
echo "  CIFAR-100: 会自动下载到 $BENCHMARK_DIR/data/"
echo ""

