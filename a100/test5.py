import torch
import time
import multiprocessing as mp

def gpu_heavy_load(proc_id, counter, stop_event):
    """GPU를 쉴 틈 없이 갈구는 연산 프로세스"""
    # 1. 행렬 크기 확대 (메모리 대역폭 타격용)
    size = 1024 * 16 * 2
    
    # 2. Tensor Core 가속 활성화 (TF32) - 전력 소모 극대화
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 여러 개의 행렬을 메모리에 올려 데이터 전송 부하 유도
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    local_count = 0
    last_sync_time = time.time()

    while not stop_event.is_set():
        # 실제 연산 수행
        torch.matmul(a, b)
        local_count += 1
        
        # 매번 Lock을 걸지 않고, 0.1초마다 한 번씩만 메인 카운터에 보고 (CPU 병목 방지)
        if time.time() - last_sync_time > 0.1:
            with counter.get_lock():
                counter.value += local_count
            local_count = 0
            last_sync_time = time.time()

def monitor_process(counter, stop_event, num_procs):
    """처리율 및 변화율 실시간 모니터링"""
    size = 1024 * 16
    ops_per_matmul = 2 * (size ** 3)
    
    baseline_tflops = None
    history = []
    
    print(f"[{time.strftime('%H:%M:%S')}] A100 가혹 부하 테스트 (프로세스 {num_procs}개)")
    print("-" * 80)
    print(f"{'Time':<10} | {'TFLOPS':<10} | {'Change (%)':<12} | {'Total Ops'}")
    print("-" * 80)
    
    last_count = 0
    last_time = time.time()

    while not stop_event.is_set():
        time.sleep(1)
        
        curr_time = time.time()
        with counter.get_lock():
            curr_count = counter.value
            
        dt = curr_time - last_time
        dc = curr_count - last_count
        
        tflops = (dc * ops_per_matmul / dt) / 1e12
        
        if baseline_tflops is None:
            history.append(tflops)
            if len(history) >= 5:
                baseline_tflops = max(history)
            change_str = "Wait..."
        else:
            change_pct = (tflops / baseline_tflops) * 100
            change_str = f"{change_pct:>8.2f} %"

        print(f"{time.strftime('%H:%M:%S'):<10} | {tflops:>10.2f} | {change_str:<12} | {curr_count:>10}")
        
        last_count = curr_count
        last_time = curr_time

if __name__ == "__main__":
    num_procs = 4  # A100을 압착하기 위한 병렬 프로세스 수
    counter = mp.Value('i', 0)
    stop_event = mp.Event()
    
    # 1. 모니터링 시작
    p_mon = mp.Process(target=monitor_process, args=(counter, stop_event, num_procs))
    p_mon.start()
    
    # 2. 다중 연산 프로세스 시작
    workers = []
    for i in range(num_procs):
        p = mp.Process(target=gpu_heavy_load, args=(i, counter, stop_event))
        p.start()
        workers.append(p)
    
    try:
        p_mon.join()
    except KeyboardInterrupt:
        print("\n[!] 중단 요청. 프로세스 정리 중...")
        stop_event.set()
        for w in workers:
            w.terminate()
        p_mon.terminate()