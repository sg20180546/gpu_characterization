import torch
import time

# 1. Compute-bound 커널 (Matrix Multiplication)
# 연산량이 매우 많아 GPU의 산술 성능 한계까지 밀어붙입니다.
def compute_heavy_task():
    size = 8192
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    # Nsight Compute가 캡처할 수 있도록 이름을 지정한 커널 실행
    print("Running Compute-bound kernel...")
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

# 2. Memory-bound 커널 (Vector Addition)
# 연산은 단순하지만 데이터 이동량이 많아 메모리 대역폭을 주로 사용합니다.
def memory_heavy_task():
    size = 1024 * 1024 * 256 # 대용량 벡터
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    print("Running Memory-bound kernel...")
    for _ in range(10):
        z = x + y
    torch.cuda.synchronize()

if __name__ == "__main__":
    # Warm up
    compute_heavy_task()
    memory_heavy_task()