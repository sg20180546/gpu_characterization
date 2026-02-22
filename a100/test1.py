
# 포맷,비트 구성 (S+E+M),특징,용도
# FP32 (Single),1+8+23,표준 정밀도. 가장 정확하지만 메모리와 전력 소모가 큼.,"가중치 업데이트, 마스터 웨이트 저장"
# FP16 (Half),1+5+10,메모리 절반 사용. 표현 범위가 좁아 Loss Scaling 필수.,"일반적인 가속 학습, 추론"
# BF16 (Bfloat16),1+8+7,FP32와 지수 비트가 같아 범위가 넓음. 스케일링 불필요.,LLM 학습의 표준 (안정성 높음)
# TF32 (NVIDIA 전용),1+8+10,내부 연산 시에만 사용. FP32의 범위 + FP16의 속도.,A100/H100 기본 가속 모드
# FP8 (H100 핵심),1+4+3 / 1+5+2,극단적인 다이어트. Transformer Engine 필수.,H100 초고속 학습 및 추론

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertForSequenceClassification
import time
import pynvml
import matplotlib.pyplot as plt
import numpy as np

# NVML 초기화 (전력 측정용)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_power():
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

def run_benchmark(precision, iterations=50, batch_size=32):
    print(f"Testing {precision.upper()}...")
    
    # 정밀도 설정
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

    # 모델 및 데이터 설정
    config = BertConfig()
    model = BertForSequenceClassification(config).to("cuda").to(dtype)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    inputs = torch.randint(0, config.vocab_size, (batch_size, 128)).to("cuda")
    labels = torch.zeros(batch_size, dtype=torch.long).to("cuda")

    # Warm-up (GPU 예열)
    for _ in range(10):
        model(inputs, labels=labels).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # 측정 시작
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
    
    return avg_throughput, avg_power

# 2. 데이터 수집
precisions = ["fp32", "tf32", "fp16", "bf16"]
results = {p: run_benchmark(p) for p in precisions}

# 3. 시각화 (Matplotlib)
names = [p.upper() for p in precisions]
throughputs = [results[p][0] for p in precisions]
powers = [results[p][1] for p in precisions]
eff_ratio = [t / p for t, p in zip(throughputs, powers)] # Perf per Watt

fig, ax1 = plt.subplots(figsize=(10, 6))

# 바 차트: Throughput
color1 = 'tab:blue'
ax1.set_xlabel('Precision Type')
ax1.set_ylabel('Throughput (samples/sec)', color=color1)
ax1.bar(names, throughputs, color=color1, alpha=0.6, label='Throughput')
ax1.tick_params(axis='y', labelcolor=color1)

# 라인 차트: Avg Power
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Avg Power (Watts)', color=color2)
ax2.plot(names, powers, color=color2, marker='o', linewidth=2, label='Avg Power')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('A100 Performance vs Power Characterization')
fig.tight_layout()
plt.savefig('a100_characterization.png')
print("\n[성공] 'a100_characterization.png' 파일로 결과가 저장되었습니다.")
plt.show()