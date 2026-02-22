import torch
import time

def profile_custom_roofline(name, func, iters=50):
    # A100 80GB PCIe 이론적 스펙 (기준점)
    PEAK_FLOPS = 19.5 * 1e12  # FP32 Peak (19.5 TFLOPS)
    PEAK_BW = 1935 * 1e9      # Memory BW (1.935 TB/s)
    
    # Warm up
    func()
    torch.cuda.synchronize()
    
    # 시간 측정 및 전력 측정 준비
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    avg_time = elapsed_time / iters
    
    return avg_time

# --- 테스트할 커널들 ---
def compute_kernel():
    # Matrix Mul: Ops = 2 * N^3
    N = 4096
    a = torch.randn(N, N, device='cuda')
    b = torch.randn(N, N, device='cuda')
    ops = 2 * (N ** 3)
    bytes_moved = 3 * (N ** 2) * 4 # A, B, C 읽고 쓰기 (float32=4bytes)
    
    t = profile_custom_roofline("Compute", lambda: torch.matmul(a, b))
    
    tflops = (ops / t) / 1e12
    bw_used = (bytes_moved / t) / 1e9
    intensity = ops / bytes_moved # 산술 강도
    
    print(f"\n[{name.upper()} KERNEL RESULT]")
    print(f"Throughput: {tflops:.2f} TFLOPS (Peak 대비 {tflops/19.5*100:.1f}%)")
    print(f"Mem BW Used: {bw_used:.2f} GB/s (Peak 대비 {bw_used/1935*100:.1f}%)")
    print(f"Arithmetic Intensity: {intensity:.2f} FLOPs/Byte")
    
    # 병목 진단
    if tflops / 19.5 > bw_used / 1935:
        print(">> Diagnosis: COMPUTE-BOUND (연산기 성능 한계에 근접)")
    else:
        print(">> Diagnosis: MEMORY-BOUND (메모리 대역폭 한계에 근접)")

def memory_kernel():
    # Vector Add: Ops = N, Bytes = 3 * N * 4
    N = 1024 * 1024 * 128
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    ops = N
    bytes_moved = 3 * N * 4
    
    t = profile_custom_roofline("Memory", lambda: x + y)
    
    tflops = (ops / t) / 1e12
    bw_used = (bytes_moved / t) / 1e9
    intensity = ops / bytes_moved
    
    print(f"\n[{name.upper()} KERNEL RESULT]")
    print(f"Throughput: {tflops:.4f} TFLOPS")
    print(f"Mem BW Used: {bw_used:.2f} GB/s (Peak 대비 {bw_used/1935*100:.1f}%)")
    print(f"Arithmetic Intensity: {intensity:.2f} FLOPs/Byte")
    print(">> Diagnosis: MEMORY-BOUND (데이터 옮기느라 바쁨)")

if __name__ == "__main__":
    name = "Compute-bound"
    compute_kernel()
    name = "Memory-bound"
    memory_kernel()