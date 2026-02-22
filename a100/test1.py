import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertForSequenceClassification
import time
import pynvml

# NVML 초기화 (NVIDIA 관리 라이브러리)
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
except:
    print("[Error] NVML 초기화 실패. NVIDIA 드라이버 상태를 확인하세요.")
    exit()

def get_power():
    """현재 전력 사용량 (Watt) 반환"""
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

def run_benchmark(precision, iterations=50, batch_size=64):
    # 정밀도 설정 제어
    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.float32
    elif precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    # 모델/데이터 생성 (A100 80GB 부하용)
    config = BertConfig()
    model = BertForSequenceClassification(config).to("cuda").to(dtype)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    inputs = torch.randint(0, config.vocab_size, (batch_size, 128)).to("cuda")
    labels = torch.zeros(batch_size, dtype=torch.long).to("cuda")

    # Warm-up (정확한 측정을 위해 GPU 가열)
    for _ in range(10):
        model(inputs, labels=labels).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # 실제 측정 시작
    start_time = time.time()
    power_samples = []
    
    for _ in range(iterations):
        power_samples.append(get_power())
        outputs = model(inputs, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_throughput = (iterations * batch_size) / total_time
    avg_power = sum(power_samples) / len(power_samples)
    perf_per_watt = avg_throughput / avg_power
    
    return avg_throughput, avg_power, perf_per_watt

# 실행 및 데이터 출력
precisions = ["fp32", "tf32", "fp16", "bf16"]
results = {}

print("\n" + "="*75)
print(f" GPU Model: {gpu_name}")
print(f" Test Scenario: BERT-Base Training (Batch Size: 64)")
print("="*75)
print(f"{'Mode':<10} | {'Throughput':<15} | {'Avg Power':<12} | {'Perf / Watt'}")
print("-" * 75)

for p in precisions:
    thr, pwr, efc = run_benchmark(p)
    results[p] = (thr, pwr, efc)
    # 실시간 행 출력
    print(f"{p.upper():<10} | {thr:>10.2f} s/s | {pwr:>10.2f} W | {efc:>12.2f}")

print("="*75)

# 분석 요약 (Summary)
fp32_thr = results['fp32'][0]
tf32_thr = results['tf32'][0]
bf16_thr = results['bf16'][0]
max_efc_mode = max(results, key=lambda x: results[x][2]).upper()

print("\n[ Performance Characterization Summary ]")
print(f"▶ TF32 Acceleration: FP32 대비 약 {tf32_thr/fp32_thr:.2f}배 빠름")
print(f"▶ BF16 Acceleration: FP32 대비 약 {bf16_thr/fp32_thr:.2f}배 빠름")
print(f"▶ Energy Efficiency: '{max_efc_mode}' 모드가 전력 대비 가장 많은 연산을 수행함")
print("="*75 + "\n")


# (sj) elicer@d2abfd222339:~/gpu_characterization/a100$ python ./test1.py 

# ===========================================================================
#  GPU Model: NVIDIA A100 80GB PCIe
#  Test Scenario: BERT-Base Training (Batch Size: 64)
# ===========================================================================
# Mode       | Throughput      | Avg Power    | Perf / Watt
# ---------------------------------------------------------------------------
# FP32       |     230.56 s/s |     299.89 W |         0.77
# TF32       |     863.29 s/s |     300.79 W |         2.87
# FP16       |    1588.79 s/s |     209.69 W |         7.58
# BF16       |    1575.21 s/s |     272.09 W |         5.79
# ===========================================================================

# [ Performance Characterization Summary ]
# ▶ TF32 Acceleration: FP32 대비 약 3.74배 빠름
# ▶ BF16 Acceleration: FP32 대비 약 6.83배 빠름
# ▶ Energy Efficiency: 'FP16' 모드가 전력 대비 가장 많은 연산을 수행함
# ===========================================================================

# (sj) elicer@d2abfd222339:~/gpu_characterization/a100$ python ./test1.py 

# ===========================================================================
#  GPU Model: NVIDIA A100 80GB PCIe
#  Test Scenario: BERT-Base Training (Batch Size: 64)
# ===========================================================================
# Mode       | Throughput      | Avg Power    | Perf / Watt
# ---------------------------------------------------------------------------
# FP32       |     230.64 s/s |     304.91 W |         0.76
# TF32       |     858.95 s/s |     301.50 W |         2.85
# FP16       |    1627.00 s/s |     225.44 W |         7.22
# BF16       |    1607.20 s/s |     275.57 W |         5.83
# ===========================================================================

# [ Performance Characterization Summary ]
# ▶ TF32 Acceleration: FP32 대비 약 3.72배 빠름
# ▶ BF16 Acceleration: FP32 대비 약 6.97배 빠름
# ▶ Energy Efficiency: 'FP16' 모드가 전력 대비 가장 많은 연산을 수행함
# ===========================================================================