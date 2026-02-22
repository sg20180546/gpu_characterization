# 2. 데이터 수집
precisions = ["fp32", "tf32", "fp16", "bf16"]
results = {p: run_benchmark(p) for p in precisions}

# 3. 시각화 (Matplotlib)
names = [p.upper() for p in precisions]
throughputs = [results[p][0] for p in precisions]
powers = [results[p][1] for p in precisions]
eff_ratio = [t / p for t, p in zip(throughputs, powers)] # Perf per Watt

