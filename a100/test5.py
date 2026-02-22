import torch
import time
import multiprocessing as mp

def gpu_stress_process(counter, stop_event):
    """GPU에 쉴 틈 없이 매트릭스 연산을 퍼붓는 프로세스"""
    size = 1024 * 12
    # 공유 메모리를 쓰지 않고 각 프로세스에서 독립적으로 연산
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    while not stop_event.is_set():
        torch.matmul(a, b)
        # 연산이 끝날 때마다 공유 카운터를 1씩 증가
        with counter.get_lock():
            counter.value += 1
        # 가속을 위해 synchronize를 빼고 스트리밍으로 연산

def monitor_process(counter, stop_event):
    """카운터를 읽어서 초당 처리율과 변화율을 계산하고 출력하는 프로세스"""
    size = 1024 * 12
    ops_per_matmul = 2 * (size ** 3)
    
    history = []
    baseline_tflops = None
    
    print(f"[{time.strftime('%H:%M:%S')}] A100 지속 부하 테스트 및 실시간 모니터링 시작...")
    print("-" * 75)
    print(f"{'Time':<10} | {'TFLOPS':<10} | {'Change':<12} | {'Total Ops'}")
    print("-" * 75)
    
    start_time = time.time()
    last_count = 0
    last_check_time = start_time

    try:
        while not stop_event.is_set():
            time.sleep(1) # 1초 간격으로 리포트
            
            current_time = time.time()
            with counter.get_lock():
                current_count = counter.value
            
            # 인터벌 동안의 연산 횟수 계산
            interval_count = current_count - last_count
            interval_time = current_time - last_check_time
            
            # TFLOPS 계산
            curr_tflops = (interval_count * ops_per_matmul / interval_time) / 1e12
            
            # Baseline 설정 (첫 5초)
            if baseline_tflops is None:
                history.append(curr_tflops)
                if len(history) >= 5:
                    baseline_tflops = sum(history) / len(history)
                change_str = "Calculating..."
            else:
                change_rate = (curr_tflops / baseline_tflops) * 100
                change_str = f"{change_rate:>6.2f} %"

            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:<10} | {curr_tflops:>8.2f} | {change_str:<12} | {current_count:>10}")
            
            last_count = current_count
            last_check_time = current_time
            
    except KeyboardInterrupt:
        stop_event.set()

if __name__ == "__main__":
    # 공유 카운터와 중지 이벤트 생성
    counter = mp.Value('i', 0)
    stop_event = mp.Event()
    
    # 두 프로세스 시작
    p_stress = mp.Process(target=gpu_stress_process, args=(counter, stop_event))
    p_monitor = mp.Process(target=monitor_process, args=(counter, stop_event))
    
    p_stress.start()
    p_monitor.start()
    
    try:
        p_stress.join()
        p_monitor.join()
    except KeyboardInterrupt:
        stop_event.set()
        p_stress.terminate()
        p_monitor.terminate()
        print("\n테스트를 강제 종료합니다.")