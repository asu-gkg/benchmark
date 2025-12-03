import os
import torch
import pandas as pd
from datetime import timedelta
import torch.distributed as dist

try:
    from hadamard_cuda import hadamard_transform
except ImportError:
    hadamard_transform = None

_initialized = False
_random_diag_encode = None
_random_diag_decode = None

def is_hadamard_available():
    return hadamard_transform is not None

def _initialize_hadamard_matrices():
    global _initialized, _random_diag_encode, _random_diag_decode
    if not _initialized:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sgen = torch.Generator(device=device)
        rgen = torch.Generator(device=device)
        sgen.manual_seed(0)
        rgen.manual_seed(0)
        _random_diag_encode = 2 * torch.bernoulli(torch.ones(250000000, device=device) / 2, generator=sgen) - 1
        _random_diag_decode = 2 * torch.bernoulli(torch.ones(250000000, device=device) / 2, generator=rgen) - 1
        _initialized = True

def hadamard_hook_cuda(process_group, bucket):
    # Initialize the matrices if not already done
    _initialize_hadamard_matrices()
    
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    tensor = bucket.buffer()
    tensor.div_(group_to_use.size())
    torch.cuda.synchronize()
    encoded_allreduced_tensor = dist.all_reduce(
        hadamard_transform(tensor * _random_diag_encode[:tensor.numel()]),
        group=group_to_use, 
        async_op=True
    )
    def decode(fut):
        decoded = hadamard_transform(fut.value()[0]) / len(fut.value()[0])
        return decoded * _random_diag_decode[:tensor.numel()]
    return encoded_allreduced_tensor.get_future().then(decode)

def setup_distributed_env(args):
    file_prefix = f"{args.comm}_{args.algo}_{args.model}_{args.epochs}_{args.batch_size}"
    os.environ['MASTER_PORT'] = '12355'
    if args.comm == 'nccl':
        os.environ['NCCL_SOCKET_IFNAME'] = args.dev
    return file_prefix + ".log"
    
def initialize_process_group(args):
    dist.init_process_group(
        backend=args.comm, 
        rank=int(args.nr), 
        world_size=int(args.nodes), 
        timeout=timedelta(seconds=200)
    )

def log_training_metrics(epoch_times, epoch_acc, epoch_loss, file_path):
    df1 = pd.DataFrame(list(zip(epoch_times, epoch_acc, epoch_loss)),
        columns=['Time', 'Train Acc', 'Train Loss'])
    df1.to_csv(file_path)

def calculate_classification_accuracy(predictions, labels):
    preds = torch.argmax(predictions, dim=1)
    return torch.mean((preds == labels).float())


def calculate_span_prediction_accuracy(outputs, start_positions, end_positions):
    start_preds = torch.argmax(outputs.start_logits, dim=1)
    end_preds = torch.argmax(outputs.end_logits, dim=1)
    
    accuracy = ((start_preds == start_positions).float() + 
               (end_preds == end_positions).float()) / 2.0
    
    return torch.mean(accuracy)
