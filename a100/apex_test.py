import torch
import torch.nn as nn
import sys

# 8192 x 8192 행렬은 A100의 성능을 보기에 아주 적합한 크기입니다.
ITERATIONS = 500
DIM = 8192
BATCH_SIZE = 128

def create_24_mask(tensor):
    """
    A100 2:4 구조적 희소성 패턴 수동 생성: 
    연속된 4개 원소 중 2개를 0으로 만듭니다.
    """
    shape = tensor.shape
    t = tensor.view(-1, 4)
    mask = torch.zeros_like(t, dtype=torch.bool)
    # 각 4개 묶음에서 앞의 2개만 살립니다 (2:4 패턴)
    mask[:, :2] = True
    return mask.view(shape)

# 1. 모델 준비
mode = "SPARSE" if "apex=1" in sys.argv else "DENSE"
model = nn.Linear(DIM, DIM, bias=False).cuda().half()
input_data = torch.randn(BATCH_SIZE, DIM).cuda().half()
cache_cleaner = torch.randn(128 * 1024 * 1024 // 4, device='cuda') # 128MB

if mode == "SPARSE":
    print("[A100 가속 모드] 2:4 패턴을 가중치에 수동 적용합니다.")
    with torch.no_grad():
        mask = create_24_mask(model.weight.data)
        model.weight.data.mul_(mask)
    # 중요: A100은 FP16 연산 시 가중치에 2:4 패턴이 있으면 내부적으로 
    # 'Sparse Tensor Core'를 사용하려고 시도합니다.
else:
    print("[일반 모드] Dense 연산을 수행합니다.")

def run_benchmark():
    # Warm-up
    for _ in range(50): model(input_data)
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERATIONS):
        # L2 Cache Flush (DRAM 대역폭 효율까지 측정에 포함)
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