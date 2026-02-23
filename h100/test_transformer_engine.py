import torch
import time
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# 1. 환경 설정
device = torch.device("cuda")
batch_size = 4096
in_features = 8192
out_features = 8192
num_iters = 100

# 테스트 데이터 생성
input_data = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)

# 2. 모델 정의
# Native PyTorch BF16 Linear
model_bf16 = torch.nn.Linear(in_features, out_features).to(device).to(torch.bfloat16)

# Transformer Engine FP8 Linear
model_fp8 = te.Linear(in_features, out_features).to(device)

def benchmark(model, data, mode_name, use_fp8=False):
    # Warm-up (GPU 예열)
    for _ in range(10):
        if use_fp8:
            with te.fp8_autocast(enabled=True):
                _ = model(data)
        else:
            _ = model(data)
    torch.cuda.synchronize()

    # 성능 측정 시작
    start_time = time.perf_counter()
    
    for _ in range(num_iters):
        if use_fp8:
            # FP8 Recipe 설정 (H100 전용 가속)
            with te.fp8_autocast(enabled=True):
                output = model(data)
        else:
            output = model(data)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_iters) * 1000
    ips = num_iters / total_time

    print(f"[{mode_name}]")
    print(f" - Avg Latency: {avg_latency:.3f} ms")
    print(f" - Throughput: {ips:.2f} iters/sec")
    print("-" * 30)
    return avg_latency

# 3. 테스트 실행
print(f"Starting Benchmark on H100...\n")

# BF16 테스트
latency_bf16 = benchmark(model_bf16, input_data, "Native PyTorch BF16", use_fp8=False)

# FP8 테스트
latency_fp8 = benchmark(model_fp8, input_data, "Transformer Engine FP8", use_fp8=True)

# 4. 결과 분석
speedup = latency_bf16 / latency_fp8
print(f"H100 FP8 is {speedup:.2f}x faster than BF16 in this workload.")