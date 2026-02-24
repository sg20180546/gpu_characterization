import torch
import triton
import triton.language as tl

# 1. Triton 커널: DSMEM 직접 접근 vs Global Memory 경유
@triton.jit
def dsmem_transfer_kernel(
    data_ptr, 
    BLOCK_SIZE: tl.constexpr, 
    USE_DSMEM: tl.constexpr
):
    # 현재 프로그램(Thread Block) ID
    pid = tl.program_id(0)
    
    # H100 Cluster 구조: 2개의 SM이 한 클러스터라고 가정 (0번, 1번)
    # USE_DSMEM이 True일 때, Triton은 내부적으로 cluster 내 상대 주소를 계산함
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # 내 데이터 위치
    my_ptr = data_ptr + pid * BLOCK_SIZE + offsets
    
    if USE_DSMEM:
        # DSMEM을 통한 인접 SM 데이터 로드
        # 0번 피드는 1번을, 1번 피드는 0번 데이터를 가져옴
        remote_pid = 1 - pid 
        remote_ptr = data_ptr + remote_pid * BLOCK_SIZE + offsets
        # Cluster 내 다른 SM의 Shared Memory에 직접 접근
        val = tl.load(remote_ptr) 
    else:
        # Global Memory(L2 Cache)를 경유하여 다른 블록 데이터 로드
        remote_pid = 1 - pid
        remote_ptr = data_ptr + remote_pid * BLOCK_SIZE + offsets
        val = tl.load(remote_ptr)

    tl.store(my_ptr, val)

def run_benchmark():
    # H100 특성화 파라미터
    BLOCK_SIZE = 8192  # 8KB
    NUM_BLOCKS = 2     # Cluster 내 통신을 위해 2개 블록 사용
    
    data = torch.randn(NUM_BLOCKS * BLOCK_SIZE, device='cuda', dtype=torch.float32)

    print(f"--- H100 DSMEM vs Global Memory Latency Test ---")
    print(f"Data size per SM: {BLOCK_SIZE * 4 / 1024:.1f} KB")

    # A. Global Memory 방식 (Cluster 미사용)
    def bench_global():
        # grid만 대괄호에 넣고, 나머지는 인자로 전달
        dsmem_transfer_kernel[(NUM_BLOCKS,)](
            data, BLOCK_SIZE, USE_DSMEM=False, 
            num_warps=4
        )

    # B. DSMEM 방식 (Cluster Size = 2 설정)
    def bench_dsmem():
        # cluster_dims를 통해 하드웨어 Cluster 기능 활성화
        dsmem_transfer_kernel[(NUM_BLOCKS,)](
            data, BLOCK_SIZE, USE_DSMEM=True, 
            num_warps=4,
            cluster_dims=(2, 1, 1)  # SM 2개를 하드웨어적으로 묶음
        )

    # 지연 시간 측정 (ms -> us 변환)
    ms_global = triton.testing.do_bench(bench_global)
    ms_dsmem = triton.testing.do_bench(bench_dsmem)

    print(f"\n[Results]")
    print(f"Global Memory (L2) Latency : {ms_global * 1000:.2f} us")
    print(f"DSMEM Direct Latency      : {ms_dsmem * 1000:.2f} us")
    print("-" * 40)
    
    speedup = ms_global / ms_dsmem
    print(f"Performance Gain: {speedup:.2f}x Faster")

if __name__ == "__main__":
    # H100 여부 확인
    device_name = torch.cuda.get_device_name()
    if "H100" in device_name or "Hopper" in device_name:
        run_benchmark()
    else:
        print(f"Current Device: {device_name}")
        print("This test is optimized for H100 DSMEM. Results on other GPUs might not show DSMEM gains.")