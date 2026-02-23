import torch
import torch.nn as nn
import sys
from apex.contrib.sparsity import ASP

# 성능 차이를 확실히 보기 위해 행렬 크기를 크게 잡습니다.
DIM = 8192
BATCH_SIZE = 128
ITERATIONS = 500

# 1. 모델 및 데이터 준비
use_apex = True if "apex=1" in sys.argv else False
model = nn.Linear(DIM, DIM, bias=False).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda') # 128MB L2 Flush

if use_apex:
    print("[ASP] A100 Structural Sparsity 가속 모드 활성화...")
    asp = ASP()
    # 하드웨어가 인식할 수 있도록 모델을 준비하고 2:4 패턴 마스크를 생성/적용합니다.
    # 이 과정에서 내부적으로 가속을 위한 'Sparse 커널'로 교체됩니다.
    asp.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2)
    asp.compute_sparse_masks()
    mode_label = "SPARSE (Apex)"
else:
    print("[DENSE] 일반 모드 연산 수행...")
    mode_label = "DENSE (Normal)"

def run_bench():
    # Warm-up (초기 오버헤드 제거)
    for _ in range(50): model(input_data)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERATIONS):
        # L2 캐시 플러시 (가중치를 DRAM에서 새로 읽어오도록 강제)
        _ = cache_cleaner.zero_()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_ms = sum(times) / len(times)
    # 2*M*N*K 연산량 기준 TFLOPS 계산
    tflops = (2 * BATCH_SIZE * DIM * DIM) / (avg_ms * 1e-3) / 1e12
    
    print(f"\n[{mode_label} 결과]")
    print(f"평균 지연 시간: {avg_ms:.4f} ms")
    print(f"실효 연산 성능: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    run_bench()