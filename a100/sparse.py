import torch
import torch.nn as nn
from apex.contrib.sparsity import ASP

# 설정 (Dense와 동일하게 유지)
ITERATIONS = 500
BATCH_SIZE = 128
DIM = 8192

model = nn.Sequential(nn.Linear(DIM, DIM, bias=False)).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda')

# [핵심] Structural Sparsity 활성화
asp = ASP()
asp.init_model_for_pruning(model, mask_calculator="24", verbosity=2)
asp.compute_sparse_masks()

def run_bench():
    # Warm-up
    for _ in range(50): model(input_data)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERATIONS):
        # L2 Cache Flush
        _ = cache_cleaner.zero_()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print(f"=== SPARSE (2:4) MODE RESULT ===")
    print(f"Avg Latency: {sum(times)/len(times):.4f} ms")

if __name__ == "__main__":
    run_bench()