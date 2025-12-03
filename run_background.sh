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

# Usage function
usage() {
    echo "Usage: $0 -s SIZE -r STARTING_RANK -t TIME [-m MASTER_ADDR] [-p MASTER_PORT] [-d DEVICE] [-b BACKEND]"
    echo "Options:"
    echo "  -s SIZE           Number of processes"
    echo "  -r STARTING_RANK  Starting rank"
    echo "  -t TIME           Iteration time in seconds"
    echo "  -m MASTER_ADDR    Master address (default: 10.0.2.253)"
    echo "  -p MASTER_PORT    Master port (default: 12355)"
    echo "  -d DEVICE         Network device (default: auto-detect from MASTER_ADDR)"
    echo "  -b BACKEND        Backend: 'nccl' (GPU) or 'gloo' (CPU) (default: nccl)"
    echo ""
    echo "Note: If DEVICE is not specified, it will be auto-detected based on MASTER_ADDR"
    echo "      Use -b gloo to run on CPU without using GPU resources"
    exit 1
}

# Paths and directories for the benchmark
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MASTER_ADDR="10.0.2.253"
MASTER_PORT="12355"
TCP_DEVICE=""  # Will be auto-detected if not specified
BACKEND="nccl"  # Default to NCCL

# Parse input arguments
while getopts ":s:r:t:m:p:d:b:" opt; do
  case $opt in
    s) SIZE="$OPTARG"
    ;;
    r) STARTING_RANK="$OPTARG"
    ;;
    t) ITERATION_TIME="$OPTARG"
    ;;
    m) MASTER_ADDR="$OPTARG"
    ;;
    p) MASTER_PORT="$OPTARG"
    ;;
    d) TCP_DEVICE="$OPTARG"
    ;;
    b) BACKEND="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
  esac
done

# Validate backend
if [ "$BACKEND" != "nccl" ] && [ "$BACKEND" != "gloo" ]; then
    echo "Error: Backend must be 'nccl' or 'gloo'"
    usage
fi

# Set benchmark script based on backend
if [ "$BACKEND" = "gloo" ]; then
    BENCHMARK_SCRIPT="$SCRIPT_DIR/gloo_benchmark.py"
else
    BENCHMARK_SCRIPT="$SCRIPT_DIR/nccl_benchmark.py"
fi

# Check if the necessary arguments are provided
if [ -z "$SIZE" ] || [ -z "$STARTING_RANK" ] || [ -z "$ITERATION_TIME" ]; then
    usage
fi

# Auto-detect network device if not specified
if [ -z "$TCP_DEVICE" ]; then
    echo "Auto-detecting network device based on MASTER_ADDR ($MASTER_ADDR)..."
    TCP_DEVICE=$(auto_detect_device "$MASTER_ADDR")
    
    if [ -z "$TCP_DEVICE" ]; then
        echo "Warning: Could not auto-detect network device. Trying common names..."
        # Try common device names
        for dev in enp6s27f0np0 ens17 eno1 eth0; do
            if ip link show "$dev" &>/dev/null; then
                TCP_DEVICE="$dev"
                echo "Found device: $TCP_DEVICE"
                break
            fi
        done
        
        if [ -z "$TCP_DEVICE" ]; then
            echo "Error: Could not find network device. Please specify with -d option."
            echo "Available network interfaces:"
            ip -o link show | awk -F': ' '{print "  " $2}'
            exit 1
        fi
    else
        echo "Auto-detected device: $TCP_DEVICE"
    fi
fi

# Verify device exists
if ! ip link show "$TCP_DEVICE" &>/dev/null; then
    echo "Error: Network device '$TCP_DEVICE' not found."
    echo "Available network interfaces:"
    ip -o link show | awk -F': ' '{print "  " $2}'
    exit 1
fi

# Set other constant variables
ELEMENTS="2048000"

# Check GPU availability and limit ranks to avoid conflicts (only for NCCL backend)
if [ "$BACKEND" = "nccl" ]; then
    if command -v python3 &> /dev/null; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            # Since ranks increment by 2, we can start at most GPU_COUNT ranks
            # Maximum rank that can be started without conflict: (GPU_COUNT - 1) * 2
            # For 4 GPUs: max rank is 6 (ranks 0, 2, 4, 6), so max SIZE is 8
            MAX_RANK_WITHOUT_CONFLICT=$(((GPU_COUNT - 1) * 2))
            MAX_SIZE=$((MAX_RANK_WITHOUT_CONFLICT + 1))
            
            if [ $SIZE -gt $MAX_SIZE ]; then
                echo "Warning: You have only $GPU_COUNT GPUs available."
                echo "Limiting SIZE from $SIZE to $MAX_SIZE to avoid GPU conflicts."
                echo "Ranks will be: $STARTING_RANK, $((STARTING_RANK + 2)), ..., $MAX_RANK_WITHOUT_CONFLICT"
                SIZE=$MAX_SIZE
            fi
        fi
    fi
fi

# Check if benchmark script exists
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "Error: Benchmark script not found at $BENCHMARK_SCRIPT"
    exit 1
fi

# Clean up any existing processes using the port or running benchmark
echo "Cleaning up any existing processes..."
# Kill processes using the master port
if command -v lsof &> /dev/null; then
    lsof -ti:$MASTER_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
fi
# Kill any existing benchmark processes
pkill -f "nccl_benchmark.py" 2>/dev/null || true
pkill -f "gloo_benchmark.py" 2>/dev/null || true
# Wait a moment for processes to terminate
sleep 1

# Print configuration
echo "Running $BACKEND benchmark with configuration:"
echo "Backend: $BACKEND"
echo "Size: $SIZE"
echo "Starting Rank: $STARTING_RANK"
echo "Iteration Time: $ITERATION_TIME seconds"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Network Device: $TCP_DEVICE"
echo "Elements: $ELEMENTS"

# Execute the commands
# For gloo backend, ranks increment by 1 (0, 1, 2, 3...) since no GPU conflicts
# For nccl backend, ranks increment by 2 (0, 2, 4, 6...) to avoid GPU conflicts
RANK=$STARTING_RANK
if [ "$BACKEND" = "gloo" ]; then
  # Gloo: start processes sequentially from STARTING_RANK, increment by 1
  # Each node runs SIZE/2 processes (assuming 2-node setup)
  # Node 0 (STARTING_RANK=0): runs ranks 0..(SIZE/2-1)
  # Node 1 (STARTING_RANK=1): runs ranks (SIZE/2)..(SIZE-1)
  PROCESSES_PER_NODE=$((SIZE / 2))
  if [ $STARTING_RANK -eq 0 ]; then
    # First node: ranks 0 to SIZE/2-1
    END_RANK=$PROCESSES_PER_NODE
  else
    # Second node: ranks SIZE/2 to SIZE-1
    RANK=$PROCESSES_PER_NODE
    END_RANK=$SIZE
  fi
  while [ $RANK -lt $END_RANK ]; do
    python3 -u "$BENCHMARK_SCRIPT" \
      $RANK \
      $SIZE \
      $MASTER_ADDR \
      $MASTER_PORT \
      $ELEMENTS \
      $ITERATION_TIME \
      $TCP_DEVICE 2>&1 | sed "s/^/[rank$RANK]: /" &
    RANK=$((RANK + 1))
  done
else
  # NCCL: start multiple processes per node (ranks increment by 2)
  while [ $RANK -lt $SIZE ]; do
    python3 -u "$BENCHMARK_SCRIPT" \
      $RANK \
      $SIZE \
      $MASTER_ADDR \
      $MASTER_PORT \
      $ELEMENTS \
      $ITERATION_TIME \
      $TCP_DEVICE 2>&1 | sed "s/^/[rank$RANK]: /" &
    RANK=$((RANK + 2))
  done
fi

echo "All benchmark processes started. Waiting for completion..."
echo "You should see output from each rank as they run."

# Wait for all background processes to complete
wait

echo "All benchmark processes completed."