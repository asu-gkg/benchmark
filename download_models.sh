#!/bin/bash

# 预先下载所有训练所需的模型
# 这样训练时就可以直接使用缓存，不需要等待下载

set -e

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCHMARK_DIR"

echo "=========================================="
echo "预先下载训练模型"
echo "=========================================="
echo ""

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

# 使用 Python 下载所有模型
$PYTHON_CMD << 'EOF'
import os
import sys
from tqdm import tqdm

try:
    from transformers import (
        BertForQuestionAnswering, BertTokenizerFast,
        RobertaForQuestionAnswering, RobertaTokenizerFast,
        BartForSequenceClassification, BartTokenizer,
        GPT2ForSequenceClassification, GPT2Tokenizer
    )
except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("")
    print("请安装缺失的库:")
    print("  如果使用虚拟环境: source .venv/bin/activate && pip install transformers tqdm")
    print("  或者直接运行: pip install transformers tqdm")
    print("")
    sys.exit(1)

models_to_download = [
    ("BERT", "bert-base-uncased", BertForQuestionAnswering, BertTokenizerFast),
    ("RoBERTa", "roberta-base", RobertaForQuestionAnswering, RobertaTokenizerFast),
    ("BART", "facebook/bart-base", BartForSequenceClassification, BartTokenizer),
    ("GPT2", "gpt2", GPT2ForSequenceClassification, GPT2Tokenizer),
]

print("=" * 60)
print("开始下载模型...")
print("=" * 60)
print("模型将下载到: ~/.cache/huggingface/")
print("")

for model_name, model_id, model_class, tokenizer_class in models_to_download:
    print(f"\n{'=' * 60}")
    print(f"下载 {model_name} ({model_id})...")
    print(f"{'=' * 60}")
    
    try:
        # 下载模型（会自动显示进度条）
        print(f"正在下载模型权重...")
        model = model_class.from_pretrained(model_id)
        print(f"✓ {model_name} 模型下载完成")
        
        # 下载 tokenizer（通常很快）
        print(f"正在下载 tokenizer...")
        tokenizer = tokenizer_class.from_pretrained(model_id)
        print(f"✓ {model_name} tokenizer 下载完成")
        
        # 对于 GPT2 和 BART，需要特殊处理
        if model_name == "GPT2":
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        print(f"✓ {model_name} 下载完成！")
        
    except Exception as e:
        print(f"✗ {model_name} 下载失败: {e}")
        print("继续下载其他模型...")
        continue

print("\n" + "=" * 60)
print("所有模型下载完成！")
print("=" * 60)
print("")
print("模型缓存位置: ~/.cache/huggingface/")
print("现在可以运行训练，模型会直接从缓存加载，无需等待下载。")
print("")
EOF

if [ $? -eq 0 ]; then
    echo "✓ 模型下载脚本执行完成"
else
    echo "✗ 模型下载脚本执行失败"
    exit 1
fi

