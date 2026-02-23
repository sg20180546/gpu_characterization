import torch
import torch.nn as nn
import sys
import time

# 설정
ITERATIONS = 500
DIM = 8192
BATCH_SIZE = 128

# 실행 인자 확인
use_apex = False
if len(sys.argv) > 1 and sys.argv[1] == "apex=1":
    use_apex = True
    from apex.contrib.sparsity import ASP
    mode_label = "SPARSE (Apex ASP)"
else:
    mode_label = "DENSE (Standard)"

# 1. 모델 및 데이터 준비
model = nn.Linear(DIM, DIM, bias=False).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda') # 128MB L2 Cache Flush용

# 2. Apex ASP 적용 (모드 1일 때만)
if use_apex:
    asp = ASP()
    # 2:4 패턴 적용 및 하드웨어 가속 커널 준비
    asp.init_model_for_pruning(model, mask_calculator="24", verbosity=2)
    asp.compute_sparse_masks()
    # 주의: ASP는 내부적으로 커널을 교체하여 하드웨어 가속을 활성화합니다.

def run_benchmark():
    print(f"[{mode_label}] 벤치마크 시작...")
    
    # Warm-up
    for _ in range(50):
        model(input_data)
    torch.cuda.synchronize()

    times = []
    for i in range(ITERATIONS):
        # L2 캐시 플러시 (DRAM에서 가중치를 새로 읽어오도록 강제)
        _ = cache_cleaner.zero_()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg_latency = sum(times) / len(times)
    print(f"\n결과:")
    print(f" - 모드: {mode_label}")
    print(f" - 평균 지연 시간: {avg_latency:.4f} ms")
    print(f" - TFLOPS(추정): {(2 * BATCH_SIZE * DIM * DIM) / (avg_latency * 1e-3) / 1e12:.2f}")

if __name__ == "__main__":
    run_benchmark()