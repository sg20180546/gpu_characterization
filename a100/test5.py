import torch
import time

def run_throttling_test():
    # A100의 부하를 극대화하기 위한 설정
    size = 1024 * 12  # Matrix Size (메모리와 연산을 동시에 갈구기 위함)
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    # FP32 연산량 계산: 2 * N^3
    ops_per_matmul = 2 * (size ** 3)
    
    print(f"[{time.strftime('%H:%M:%S')}] 테스트 시작...")
    print(f"터미널 B에서 'nvidia-smi' 모니터링 명령어를 실행해 주세요.")
    print("-" * 50)
    print(f"{'Time':<10} | {'TFLOPS':<12} | {'Status'}")
    print("-" * 50)

    try:
        while True:
            start_time = time.time()
            count = 0
            
            # 1초 동안 연산을 반복하고 횟수 측정
            while time.time() - start_time < 1.0:
                torch.matmul(a, b)
                count += 1
            
            torch.cuda.synchronize()
            actual_duration = time.time() - start_time
            
            # 초당 TFLOPS 계산
            total_ops = count * ops_per_matmul
            tflops = (total_ops / actual_duration) / 1e12
            
            curr_time = time.strftime("%H:%M:%S")
            print(f"{curr_time:<10} | {tflops:>10.2f} | Running...")
            
    except KeyboardInterrupt:
        print("\n[종료] 테스트를 중단합니다.")

if __name__ == "__main__":
    run_throttling_test()