import torch
import torch.nn as nn
import sys
from apex.contrib.sparsity import ASP
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

DIM = 8192
BATCH_SIZE = 4096
ITERATIONS = 500

use_apex = True if "apex=1" in sys.argv else False

# dense weight 준비
weight = torch.randn(DIM, DIM).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda')

if use_apex:
    print("[ASP] A100 Structural Sparsity 가속 모드 활성화...")
    # ASP로 2:4 마스크 생성
    model_tmp = nn.Linear(DIM, DIM, bias=False).cuda().half()
    model_tmp.weight.data.copy_(weight)
    asp = ASP()
    asp.init_model_for_pruning(model_tmp, mask_calculator="m4n2_1d", verbosity=0)
    asp.compute_sparse_masks()
    masked_weight = model_tmp.weight.data  # 50% 0으로 만들어진 dense weight

    # 실제 sparse 포맷으로 변환 → cuSPARSELt 커널 직접 호출
    SparseSemiStructuredTensor._FORCE_CUTLASS = False
    sparse_weight = to_sparse_semi_structured(masked_weight)
    weight_t = masked_weight.t().contiguous() 
    sparse_weight_t = to_sparse_semi_structured(weight_t)
    print(f"[ASP] Sparse tensor type: {type(sparse_weight).__name__}")

    def forward_fn():
        return torch.mm(input_data, sparse_weight_t)

    mode_label = "SPARSE (cuSPARSELt 2:4)"
else:
    print("[DENSE] 일반 모드 연산 수행...")
    def forward_fn():
        return torch.mm(input_data, weight.t())

    mode_label = "DENSE (cuBLAS)"

def run_bench():
    # Warm-up
    for _ in range(50):
        forward_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERATIONS):
        _ = cache_cleaner.zero_()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = forward_fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    tflops = (2 * BATCH_SIZE * DIM * DIM) / (avg_ms * 1e-3) / 1e12

    print(f"\n[{mode_label} 결과]")
    print(f"평균 지연 시간: {avg_ms:.4f} ms")
    print(f"실효 연산 성능: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    run_bench()
