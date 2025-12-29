#!/bin/bash

# Function to auto-detect network device by IP address
auto_detect_device() {
    local target_ip=$1
    local device=""
    
    # Try to find device by IP address
    if [ -n "$target_ip" ]; then
        # Priority 1: Check if any interface directly has this IP (exclude lo)
        device=$(ip -o addr show | grep -E "inet $target_ip/" | awk '{print $2}' | grep -v "^lo$" | head -1)
        
        # Priority 2: Get the network interface from route table (exclude lo)
        if [ -z "$device" ]; then
            device=$(ip route get "$target_ip" 2>/dev/null | grep -oP 'dev \K\S+' | grep -v "^lo$" | head -1)
        fi
        
        # Priority 3: Find interface with IP in same subnet (exclude lo)
        if [ -z "$device" ]; then
            # Extract subnet (first 3 octets)
            local subnet=$(echo "$target_ip" | cut -d. -f1-3)
            device=$(ip -o addr show | grep -E "inet $subnet\." | awk '{print $2}' | grep -v "^lo$" | head -1)
        fi
        
        # Priority 4: Try to find any active interface from default route (exclude lo)
        if [ -z "$device" ]; then
            device=$(ip route | grep default | awk '{print $5}' | grep -v "^lo$" | head -1)
        fi
        
        # Priority 5: Find any non-lo interface with an IP address
        if [ -z "$device" ]; then
            device=$(ip -o addr show | awk '{print $2}' | grep -v "^lo$" | grep -v "^$" | head -1)
        fi
    fi
    
    echo "$device"
}

# Check if all required arguments are provided
if [ -z "$4" ]; then
    echo "Usage: $0 <MASTER_ADDR> <RANK> <NODES> <MODEL> [DEV]"
    echo "Available models: vgg19, bert, bart, roberta, gpt2"
    echo "Set RUN_ALL=1 to run all communication schemes"
    echo "Set USE_HADAMARD=1 to use Hadamard transform"
    echo ""
    echo "Note: DEV is optional. If not specified, it will be auto-detected based on MASTER_ADDR"
    exit 1
fi

# Set environment variables
export MASTER_ADDR=$1
export RANK=$2
export NODES=$3
export MODEL=$4
export DEV=${5:-""}  # Optional, will be auto-detected if not provided
export CUBLAS_WORKSPACE_CONFIG=:16:8
export RUN_ALL=${RUN_ALL:-0}  # Default to 0 if not set
export USE_HADAMARD=${USE_HADAMARD:-0}  # Default to 0 (don't use hadamard)
export MASTER_PORT=${MASTER_PORT:-12355}  # Default master port

# Auto-detect network device if not specified
if [ -z "$DEV" ]; then
    echo "Auto-detecting network device based on MASTER_ADDR ($MASTER_ADDR)..."
    DEV=$(auto_detect_device "$MASTER_ADDR")
    
    if [ -z "$DEV" ]; then
        echo "Warning: Could not auto-detect network device. Trying common names..."
        # Try common device names
        for dev in enp6s27f0np0 ens17 eno1 eth0 mlx5_0; do
            if ip link show "$dev" &>/dev/null; then
                DEV="$dev"
                echo "Found device: $DEV"
                break
            fi
        done
        
        if [ -z "$DEV" ]; then
            echo "Error: Could not auto-detect network device. Please specify it manually."
            echo "Usage: $0 <MASTER_ADDR> <RANK> <NODES> <MODEL> <DEV>"
            exit 1
        fi
    else
        echo "Auto-detected device: $DEV"
    fi
fi

# Print the environment variables
echo "Environment variables set:"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "RANK=$RANK"
echo "NODES=$NODES"
echo "DEV=$DEV"
echo "MODEL=$MODEL"
echo "RUN_ALL=$RUN_ALL"
echo "USE_HADAMARD=$USE_HADAMARD"
echo "MASTER_PORT=$MASTER_PORT"

# Assign variables based on the model
case $MODEL in
    vgg19)
        BATCH_SIZE=128
        EPOCHS=150
        ;;
    bert)
        BATCH_SIZE=16
        EPOCHS=5
        ;;
    roberta)
        BATCH_SIZE=16
        EPOCHS=5
        ;;
    bart)
        BATCH_SIZE=8
        EPOCHS=6
        ;;
    gpt2)
        BATCH_SIZE=8
        EPOCHS=6
        ;;
    *)
        echo "Invalid model specified. Available models: vgg19, bert, bart, roberta, gpt2"
        exit 1
        ;;
esac

# Clean up any existing processes using the port or running training
echo "Cleaning up any existing processes..."
MASTER_PORT=12355
# Kill processes using the master port
if command -v lsof &> /dev/null; then
    lsof -ti:$MASTER_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
fi
# Kill any existing training processes
pkill -f "examples/train.py" 2>/dev/null || true
# Wait a moment for processes to terminate
sleep 2

# Set LD_LIBRARY_PATH for hadamard_cuda if needed
if [ "$USE_HADAMARD" = "1" ]; then
    TORCH_LIB_PATH="/home/asu/Desktop/benchmark/.venv/lib/python3.12/site-packages/torch/lib"
    if [ -d "$TORCH_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB_PATH:${LD_LIBRARY_PATH:-}"
        echo "Set LD_LIBRARY_PATH for hadamard_cuda: $TORCH_LIB_PATH"
    fi
fi

# Find Python executable (prefer venv, then python3, then python)
PYTHON_CMD=""
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found!"
    exit 1
fi

# Construct the base command
HADAMARD_FLAG=""
if [ "$USE_HADAMARD" = "1" ]; then
    HADAMARD_FLAG="--hadamard True"
fi
BASE_COMMAND="$PYTHON_CMD examples/train.py --nr $RANK --nodes $NODES --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --dev $DEV $HADAMARD_FLAG"

echo "Executing commands..."
echo "Python command: $PYTHON_CMD"
echo "Base command: $BASE_COMMAND"
echo ""

# Run all communication schemes if RUN_ALL is set
if [ "$RUN_ALL" = "1" ]; then
    echo "Running all communication schemes..."
    echo "Waiting for all nodes to connect (this may take a moment)..."
    echo ""

    # NCCL schemes
    echo "=== Starting NCCL Ring algorithm ==="
    NCCL_IB_DISABLE=1 $BASE_COMMAND --algo ring --comm nccl
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: NCCL Ring failed with exit code $exit_code"
    fi
    sleep 5

    echo "=== Starting NCCL Tree algorithm ==="
    NCCL_IB_DISABLE=1 $BASE_COMMAND --algo tree --comm nccl
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: NCCL Tree failed with exit code $exit_code"
    fi
    sleep 5
fi

echo "All commands executed."
