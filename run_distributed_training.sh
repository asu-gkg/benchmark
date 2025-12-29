#!/bin/bash

# 分布式训练启动脚本
# 自动在多个节点上启动训练，避免手动在每个节点上运行

# 配置
MASTER_ADDR="${1:-172.1.1.3}"  # 默认使用h2的实验IP作为MASTER_ADDR
MODEL="${2:-vgg19}"
RUN_ALL="${RUN_ALL:-1}"
USE_HADAMARD="${USE_HADAMARD:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # 每个节点使用1个GPU

# 节点配置（8个节点）
# 格式: "管理IP:RANK" 或 "local:RANK" 表示本地节点
# 自动检测本地 IP
LOCAL_IP=$(hostname -I | awk '{print $1}' | head -1)
NODES=(
    "10.30.10.163:0"  # h1 (rank 0)
    "10.30.9.188:1"   # h2 (rank 1)
    "10.30.11.134:2"  # h3 (rank 2)
    "10.30.10.158:3"  # h4 (rank 3)
    "10.30.9.80:4"    # h5 (rank 4)
    "10.30.9.169:5"   # h6 (rank 5)
    "10.30.11.6:6"    # h7 (rank 6)
    "10.30.11.122:7"  # h8 (rank 7)
)

SSH_USER="${SSH_USER:-asu}"
BENCHMARK_DIR="/home/asu/Desktop/benchmark"

# 检查参数
if [ -z "$1" ]; then
    echo "Usage: $0 <MASTER_ADDR> <MODEL>"
    echo ""
    echo "Examples:"
    echo "  $0 172.1.1.3 vgg19                    # 正常训练（8个节点）"
    echo "  $0 172.1.1.3 bert                     # 训练BERT模型"
    echo ""
    echo "Environment variables:"
    echo "  RUN_ALL=1              Run all communication schemes (default: 1)"
    echo "  USE_HADAMARD=1         Use Hadamard transform (default: 0)"
    echo "  CUDA_VISIBLE_DEVICES   GPU device ID (default: 0)"
    exit 1
fi

# 验证 MASTER_ADDR 是否是有效的 IP 地址格式
if ! echo "$MASTER_ADDR" | grep -qE '^([0-9]{1,3}\.){3}[0-9]{1,3}$'; then
    echo "错误: MASTER_ADDR 必须是有效的 IP 地址，而不是 '$MASTER_ADDR'"
    echo ""
    echo "正确的用法:"
    echo "  $0 <IP地址> <模型名称>"
    echo ""
    echo "例如:"
    echo "  $0 172.1.1.3 bert"
    echo "  $0 172.1.1.3 vgg19"
    exit 1
fi

# 验证 MODEL 是否是有效的模型名称
valid_models=("vgg19" "bert" "bart" "roberta" "gpt2")
if [[ ! " ${valid_models[@]} " =~ " ${MODEL} " ]]; then
    echo "错误: 无效的模型名称 '$MODEL'"
    echo ""
    echo "可用的模型: ${valid_models[*]}"
    exit 1
fi

echo "=========================================="
echo "启动分布式训练"
echo "=========================================="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MODEL: $MODEL"
echo "RUN_ALL: $RUN_ALL"
echo "USE_HADAMARD: $USE_HADAMARD"
echo "Nodes: ${#NODES[@]}"
echo "=========================================="
echo ""

# 函数：在节点上运行训练
run_on_node() {
    local node_info=$1
    local ip=$(echo $node_info | cut -d: -f1)
    local rank=$(echo $node_info | cut -d: -f2)
    
    # 日志文件
    local log_file="/tmp/training_rank${rank}_$(date +%Y%m%d_%H%M%S).log"
    
    # 构建命令，确保在正确的目录下执行，并设置所有必要的环境变量
    local cmd="cd $BENCHMARK_DIR && "
    cmd+="source .venv/bin/activate && "
    cmd+="export MASTER_ADDR=$MASTER_ADDR && "
    cmd+="export MASTER_PORT=12355 && "
    cmd+="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "
    cmd+="RUN_ALL=$RUN_ALL "
    cmd+="USE_HADAMARD=$USE_HADAMARD "
    cmd+="./run_training.sh $MASTER_ADDR $rank ${#NODES[@]} $MODEL"
    
    if [ "$ip" = "local" ] || [ "$ip" = "localhost" ] || [ "$ip" = "127.0.0.1" ] || [ "$ip" = "$LOCAL_IP" ]; then
        # 本地节点 - 使用 tee 同时显示和保存日志
        echo "[本地节点 rank $rank] 启动训练..."
        echo "日志文件: $log_file"
        (
            cd $BENCHMARK_DIR
            eval "$cmd" 2>&1 | tee "$log_file"
        ) &
        local pid=$!
        echo "本地进程 PID: $pid"
    else
        # 远程节点 - SSH 输出保存到日志文件，同时显示
        echo "[$ip rank $rank] 通过 SSH 启动训练..."
        echo "日志文件: $log_file"
        if command -v sshpass &> /dev/null; then
            sshpass -p " " ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
                "$SSH_USER@$ip" "$cmd" 2>&1 | tee "$log_file" &
        else
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
                "$SSH_USER@$ip" "$cmd" 2>&1 | tee "$log_file" &
        fi
        local pid=$!
        echo "远程进程 PID: $pid"
    fi
    
    echo "$pid:$log_file" >> /tmp/distributed_training_pids.txt
    echo "$log_file" >> /tmp/distributed_training_logs.txt
}

# 清理旧的 PID 和日志文件
rm -f /tmp/distributed_training_pids.txt
rm -f /tmp/distributed_training_logs.txt

# 在所有节点上启动训练
echo "正在启动所有节点的训练..."
echo "注意: 确保所有节点可以互相访问 MASTER_ADDR ($MASTER_ADDR)"
echo ""

# 先启动所有节点（后台运行）
for node in "${NODES[@]}"; do
    run_on_node "$node"
    sleep 0.5  # 短暂延迟，避免同时启动造成冲突
done

# 等待一下，确保所有进程都启动了
sleep 2
echo "所有节点进程已启动，等待连接..."

echo ""
echo "=========================================="
echo "所有节点已启动"
echo "=========================================="
echo "训练进程 PID 保存在: /tmp/distributed_training_pids.txt"
echo "训练日志文件保存在: /tmp/distributed_training_logs.txt"
echo ""
echo "实时查看日志（所有节点）:"
echo "  tail -f \$(cat /tmp/distributed_training_logs.txt | tr '\n' ' ')"
echo ""
echo "查看特定 rank 的日志:"
echo "  tail -f /tmp/training_rank0_*.log  # rank 0"
echo "  tail -f /tmp/training_rank1_*.log  # rank 1"
echo ""
echo "停止所有训练:"
echo "  kill \$(cat /tmp/distributed_training_pids.txt | cut -d: -f1)"
echo ""
echo "=========================================="
echo "训练输出（实时显示）:"
echo "=========================================="
echo ""

# 使用 tail -f 实时显示所有日志文件
if [ -f /tmp/distributed_training_logs.txt ]; then
    # 等待一下让日志文件创建
    sleep 2
    # 实时显示所有日志（使用 tail -f 合并多个文件）
    tail -f $(cat /tmp/distributed_training_logs.txt | tr '\n' ' ') &
    TAIL_PID=$!
    
    # 等待所有训练进程完成
    wait
    
    # 停止 tail
    kill $TAIL_PID 2>/dev/null
else
    # 如果没有日志文件，直接等待
    wait
fi

echo ""
echo "=========================================="
echo "所有训练已完成"
echo "=========================================="

