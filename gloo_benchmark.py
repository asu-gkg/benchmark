#!/usr/bin/env python3
"""
Gloo AllReduce Benchmark Script
Runs AllReduce operations using Gloo backend for benchmarking on CPU
"""

import os
import sys
import time
import torch
import torch.distributed as dist
from datetime import timedelta

def benchmark_allreduce(rank, world_size, master_addr, master_port, elements, iteration_time, device_name):
    """Run AllReduce benchmark using Gloo backend on CPU"""
    
    # Set environment variables for Gloo
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    if device_name:
        os.environ['GLOO_SOCKET_IFNAME'] = device_name
    
    # Initialize process group with Gloo backend
    try:
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=300)
        )
    except Exception as e:
        print(f"Rank {rank}: Failed to initialize process group: {e}")
        if rank == 0:
            print("Rank 0 (master) initialization failed. This will cause all other ranks to fail.")
        raise
    
    # Force CPU device for Gloo
    device = torch.device('cpu')
    print(f"Rank {rank}: Using CPU device (Gloo backend)")
    
    # Create tensor for AllReduce on CPU
    tensor = torch.ones(elements, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Benchmark
    print(f"Rank {rank}: Starting benchmark (will run for {iteration_time} seconds)...", flush=True)
    start_time = time.time()
    iterations = 0
    last_print_time = start_time
    
    while time.time() - start_time < iteration_time:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        iterations += 1
        
        # Print progress every 10 seconds
        current_time = time.time()
        if current_time - last_print_time >= 10:
            elapsed = current_time - start_time
            print(f"Rank {rank}: Progress - {iterations} iterations in {elapsed:.1f}s ({iterations/elapsed:.2f} ops/sec)", flush=True)
            last_print_time = current_time
    
    elapsed_time = time.time() - start_time
    
    print(f"Rank {rank}: Completed {iterations} iterations in {elapsed_time:.2f}s "
          f"({iterations/elapsed_time:.2f} ops/sec)")
    
    # Cleanup
    dist.destroy_process_group()

def main():
    if len(sys.argv) < 8:
        print("Usage: python gloo_benchmark.py <rank> <world_size> <master_addr> <master_port> "
              "<elements> <iteration_time> <device_name>")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = int(sys.argv[4])
    elements = int(sys.argv[5])
    iteration_time = float(sys.argv[6])
    device_name = sys.argv[7] if len(sys.argv) > 7 else None
    
    benchmark_allreduce(rank, world_size, master_addr, master_port, 
                       elements, iteration_time, device_name)

if __name__ == '__main__':
    main()

