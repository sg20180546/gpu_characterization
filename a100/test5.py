import torch
import time
import multiprocessing as mp

def gpu_hell_fire(proc_id, counter, stop_event):
    # 1. 캐시 크기보다 훨씬 큰 데이터셋 준비 (A100 L2 캐시는 40MB에 불과함)
    # 16K 행렬 여러 개를 리스트에 담아 매번 다른 행렬을 연산하게 함
    size = 1024 * 16 * 2
    num_buffers = 8 # 총 12GB 이상의 메모리 점유하여 캐시 순환 유도
    
    # 텐서 코어 가속 활성화
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 여러 개의 버퍼를 생성 (캐시 히트 방지용)
    buffers = [torch.randn(size, size, device='cuda', dtype=torch.float32) for _ in range(num_buffers)]
    output = torch.empty(size, size, device='cuda', dtype=torch.float32)
    
    local_count = 0
    idx = 0
    last_sync_time = time.time()

    while not stop_event.is_set():
        # 2. 매 사이클마다 다른 버퍼 조합을 사용하여 캐시 무력화
        # A = buffer[0], B = buffer[1] -> 다음은 A = buffer[2], B = buffer[3] ...
        idx_a = (idx) % num_buffers
        idx_b = (idx + 1) % num_buffers
        
        # 실제 연산 (HBM에서 데이터를 강제로 읽어오게 만듦)
        torch.matmul(buffers[idx_a], buffers[idx_b], out=output)
        
        local_count += 1
        idx += 2
        
        # 0.1초마다 카운트 보고
        if time.time() - last_sync_time > 0.1:
            with counter.get_lock():
                counter.value += local_count
            local_count = 0
            last_sync_time = time.time()

def monitor_process(counter, stop_event, num_procs):
    size = 1024 * 16
    ops_per_matmul = 2 * (size ** 3)
    
    baseline_tflops = None
    print(f"[{time.strftime('%H:%M:%S')}] A100 Cache-Busting Stress Test")
    print("-" * 85)
    print(f"{'Time':<10} | {'TFLOPS':<10} | {'Change (%)':<12} | {'Temp'} | {'Power'}")
    print("-" * 85)
    
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
        
        # 기준 설정
        if baseline_tflops is None and len(history := []) < 5:
            history.append(tflops)
            if len(history) == 5: baseline_tflops = max(history)
            change_str = "Wait..."
        else:
            change_pct = (tflops / baseline_tflops) * 100 if baseline_tflops else 100
            change_str = f"{change_pct:>8.2f} %"

        print(f"{time.strftime('%H:%M:%S'):<10} | {tflops:>10.2f} | {change_str:<12} | Monitoring via smi...")
        last_count = curr_count
        last_time = curr_time

if __name__ == "__main__":
    # 프로세스 수를 GPU 코어 수에 맞춰 8개 정도로 증설 (압착 극대화)
    num_procs = 8 
    counter = mp.Value('i', 0)
    stop_event = mp.Event()
    
    workers = []
    for i in range(num_procs):
        p = mp.Process(target=gpu_hell_fire, args=(i, counter, stop_event))
        p.start()
        workers.append(p)
    
    p_mon = mp.Process(target=monitor_process, args=(counter, stop_event, num_procs))
    p_mon.start()
    
    try:
        p_mon.join()
    except KeyboardInterrupt:
        stop_event.set()
        for w in workers: w.terminate()
        p_mon.terminate()