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

# 입력 데이터 (BF16)
input_data = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)

# 2. 모델 정의 및 타입 일치 (이 부분이 수정되었습니다)
model_bf16 = torch.nn.Linear(in_features, out_features).to(device).to(torch.bfloat16)

# TE 레이어도 입력과 똑같이 bfloat16으로 초기화해야 에러가 안 납니다.
model_fp8 = te.Linear(in_features, out_features).to(device).to(torch.bfloat16)

# FP8 Recipe 설정
fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=16, amax_compute_algo='max')

def benchmark(model, data, mode_name, use_fp8=False):
    # Warm-up
    for _ in range(10):
        if use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = model(data)
        else:
            _ = model(data)
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iters):
        if use_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = model(data)
        else:
            _ = model(data)
    torch.cuda.synchronize()
    
    total_time = time.perf_counter() - start_time
    avg_latency = (total_time / num_iters) * 1000
    print(f"[{mode_name}] Avg Latency: {avg_latency:.3f} ms")
    return avg_latency

# 3. 실행
print("Starting Validated Benchmark on H100...\n")
latency_bf16 = benchmark(model_bf16, input_data, "Native PyTorch BF16")
latency_fp8 = benchmark(model_fp8, input_data, "Transformer Engine FP8", use_fp8=True)

print(f"\nSpeedup: {latency_bf16/latency_fp8:.2f}x")