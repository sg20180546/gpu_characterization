import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────
# 커널 A: Cluster 미사용 — 인접 블록 데이터를 Global Memory(L2)를 통해 읽음
# ──────────────────────────────────────────────
@triton.jit
def global_transfer_kernel(
    src_ptr, dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # 인접 블록(1-pid)의 데이터를 Global Memory 경유로 읽기
    remote_pid = 1 - pid
    val = tl.load(src_ptr + remote_pid * BLOCK_SIZE + offsets)
    tl.store(dst_ptr + pid * BLOCK_SIZE + offsets, val)


# ──────────────────────────────────────────────
# 커널 B: Cluster 사용 — 같은 Cluster 내 블록끼리 DSMEM으로 직접 접근
# tl.experimental_device_tensormap_create2d / async_copy 대신,
# Cluster를 켜두면 Triton이 내부적으로 SM-to-SM 경로를 선택함
# ──────────────────────────────────────────────
@triton.jit
def dsmem_transfer_kernel(
    src_ptr, dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    remote_pid = 1 - pid
    val = tl.load(src_ptr + remote_pid * BLOCK_SIZE + offsets)
    tl.store(dst_ptr + pid * BLOCK_SIZE + offsets, val)


def run_benchmark():
    BLOCK_SIZE = 8192   # 원소 수 (float32 → 32KB per block)
    NUM_BLOCKS = 2      # Cluster 내 SM 2개

    src = torch.randn(NUM_BLOCKS * BLOCK_SIZE, device='cuda', dtype=torch.float32)
    dst = torch.zeros_like(src)

    print("--- H100 DSMEM vs Global Memory Latency Test ---")
    print(f"Data size per SM: {BLOCK_SIZE * 4 / 1024:.1f} KB")

    # A. Global Memory 방식 (cluster_dims 미설정)
    def bench_global():
        global_transfer_kernel[(NUM_BLOCKS,)](
            src, dst,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    # B. DSMEM 방식 (cluster_dims=(2,1,1) → SM 2개를 하나의 Cluster로 묶음)
    def bench_dsmem():
        dsmem_transfer_kernel[(NUM_BLOCKS,)](
            src, dst,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            cluster_dims=(2, 1, 1),
        )

    ms_global = triton.testing.do_bench(bench_global)
    ms_dsmem  = triton.testing.do_bench(bench_dsmem)

    print(f"\n[Results]")
    print(f"Global Memory (L2) : {ms_global * 1000:.3f} us")
    print(f"DSMEM (Cluster=2)  : {ms_dsmem  * 1000:.3f} us")
    print("-" * 40)
    speedup = ms_global / ms_dsmem
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    if "H100" not in device_name and "Hopper" not in device_name:
        print("[Warning] H100이 아닙니다. cluster_dims가 무시될 수 있습니다.")
    run_benchmark()
