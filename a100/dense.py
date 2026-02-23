import torch
import torch.nn as nn
import time

# 설정
ITERATIONS = 500
BATCH_SIZE = 128
DIM = 8192  # A100의 연산 유닛을 충분히 쓰기 위해 큰 사이즈 권장

# 모델 및 데이터 준비
model = nn.Sequential(nn.Linear(DIM, DIM, bias=False)).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda') # 128MB 캐시 비우기용

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

    print(f"=== DENSE MODE RESULT ===")
    print(f"Avg Latency: {sum(times)/len(times):.4f} ms")

if __name__ == "__main__":
    run_bench()