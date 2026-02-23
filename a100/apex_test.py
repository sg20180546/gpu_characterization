import torch
import torch.nn as nn
import sys
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# 1. 설정
DIM = 8192
BATCH_SIZE = 128
ITERATIONS = 500

# 2. 모델 준비
mode = "SPARSE" if "apex=1" in sys.argv else "DENSE"
model = nn.Linear(DIM, DIM, bias=False).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda') # L2 Flush용

if mode == "SPARSE":
    print("[A100 가속 모드] 2:4 Sparse 커널을 적용합니다.")
    # 가중치를 2:4 패턴으로 마스킹 (4개 중 2개 0)
    mask = torch.tensor([1, 1, 0, 0], device='cuda', dtype=torch.bool).repeat(DIM * DIM // 4)
    with torch.no_grad():
        model.weight.data.view(-1).mul_(mask)
        # 중요: 이 함수가 A100의 하드웨어 Sparse 가속을 실제로 호출하는 핵심입니다.
        model.weight.data = to_sparse_semi_structured(model.weight.data)
else:
    print("[일반 모드] Dense 연산을 수행합니다.")

def run_benchmark():
    # Warm-up
    for _ in range(50): model(input_data)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERATIONS):
        # L2 Cache Flush (DRAM에서 새로 읽어오도록 강제)
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
    tflops = (2 * BATCH_SIZE * DIM * DIM) / (avg_ms * 1e-3) / 1e12
    
    print(f"\n[{mode} 결과]")
    print(f"평균 지연 시간: {avg_ms:.4f} ms")
    print(f"연산 성능: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    run_benchmark()