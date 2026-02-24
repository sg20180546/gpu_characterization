import torch
import triton
import triton.language as tl

# 1. DSMEM 활용 커널 (Thread Block Cluster 사용)
@triton.jit
def dsmem_transfer_kernel(
    data_ptr, 
    BLOCK_SIZE: tl.constexpr, 
    USE_DSMEM: tl.constexpr
):
    pid = tl.program_id(0)
    # H100의 Cluster ID와 내부 ID 확인
    cluster_id = pid // 2  # 2개의 SM을 하나의 클러스터로 묶음
    local_id = pid % 2
    
    # 각 SM이 담당할 메모리 주소 계산
    offsets = tl.arange(0, BLOCK_SIZE)
    ptr = data_ptr + pid * BLOCK_SIZE + offsets
    
    if USE_DSMEM:
        # [DSMEM 방식] 
        # 옆 SM(remote)의 데이터를 직접 가져옴 (L2/HBM을 거치지 않음)
        # Triton은 cluster_id가 정의되면 자동으로 DSMEM 경로를 활성화 시도함
        remote_pid = cluster_id * 2 + (1 - local_id)
        remote_ptr = data_ptr + remote_pid * BLOCK_SIZE + offsets
        val = tl.load(remote_ptr) # Cluster 내 SM 간 직접 통신
    else:
        # [Global Memory 방식]
        # 일반적인 load/store로 다른 블록의 데이터를 참조 (L2 캐시 강제 경유)
        remote_pid = (pid + 1) % 2 
        remote_ptr = data_ptr + remote_pid * BLOCK_SIZE + offsets
        val = tl.load(remote_ptr)

    tl.store(ptr, val)

# 2. 벤치마크 함수
def run_benchmark():
    BLOCK_SIZE = 4096  # 4KB 단위 전송
    NUM_BLOCKS = 2     # 두 SM 간의 통신 테스트
    
    data = torch.randn(NUM_BLOCKS * BLOCK_SIZE, device='cuda', dtype=torch.float32)

    print(f"--- H100 DSMEM vs Global Memory Latency Test ---")

    # A. Global Memory 방식 (Cluster Size 1 = No DSMEM)
    def bench_global():
        return dsmem_transfer_kernel[(NUM_BLOCKS,)](
            data, BLOCK_SIZE, USE_DSMEM=False, 
            num_warps=4, grid=(NUM_BLOCKS,)
        )

    # B. DSMEM 방식 (Cluster Size 2 = DSMEM Enabled)
    # Triton에서 cluster_2d 속성을 사용하여 하드웨어 클러스터링 활성화
    def bench_dsmem():
        return dsmem_transfer_kernel[(NUM_BLOCKS,)](
            data, BLOCK_SIZE, USE_DSMEM=True, 
            num_warps=4, grid=(NUM_BLOCKS,),
            cluster_dims=(2, 1, 1) # 2개 SM을 Cluster로 묶음 (H100 전용)
        )

    # 시간 측정
    ms_global = triton.testing.do_bench(bench_global)
    ms_dsmem = triton.testing.do_bench(bench_dsmem)

    print(f"Global Memory Latency: {ms_global * 1000:.4f} us")
    print(f"DSMEM Direct Latency : {ms_dsmem * 1000:.4f} us")
    print("-" * 40)
    
    speedup = ms_global / ms_dsmem
    print(f"DSMEM Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        reduction = (1 - (ms_dsmem / ms_global)) * 100
        print(f"Latency Reduced by: {reduction:.2f}%")

if __name__ == "__main__":
    if "H100" in torch.cuda.get_device_name():
        run_benchmark()
    else:
        print("This script requires an NVIDIA H100 GPU to measure DSMEM performance.")