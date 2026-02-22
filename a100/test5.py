import torch
import time

def run_throttling_profile():
    # A100 부하를 위한 매트릭스 크기 (기존보다 더 키워 부하를 극대화)
    size = 1024 * 14  
    a = torch.randn(size, size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    # 1회 연산당 FLOPs: 2 * N^3
    ops_per_matmul = 2 * (size ** 3)
    
    baseline_tflops = None
    print(f"[{time.strftime('%H:%M:%S')}] A100 쓰로틀링 프로파일링 시작...")
    print(f"기준 성능 설정 중 (초기 5초)...")
    print("-" * 70)
    print(f"{'Time':<10} | {'TFLOPS':<10} | {'Change':<12} | {'Status'}")
    print("-" * 70)

    history = []

    try:
        while True:
            start_time = time.time()
            count = 0
            
            # 1초 동안 반복 실행
            while time.time() - start_time < 1.0:
                torch.matmul(a, b)
                count += 1
            
            torch.cuda.synchronize()
            actual_duration = time.time() - start_time
            
            # 현재 TFLOPS 계산
            curr_tflops = (count * ops_per_matmul / actual_duration) / 1e12
            
            # 첫 5회 평균을 기준 성능(Baseline)으로 설정
            if baseline_tflops is None:
                history.append(curr_tflops)
                if len(history) >= 5:
                    baseline_tflops = sum(history) / len(history)
                change_str = "Setting..."
                status = "Calibrating"
            else:
                # 기준 대비 변화율 계산
                change_rate = (curr_tflops / baseline_tflops) * 100
                change_str = f"{change_rate:>6.2f} %"
                
                if change_rate > 98:
                    status = "Stable"
                elif change_rate > 90:
                    status = "Slight Drop"
                else:
                    status = "THROTTLING!"

            curr_time = time.strftime("%H:%M:%S")
            print(f"{curr_time:<10} | {curr_tflops:>8.2f} | {change_str:<12} | {status}")
            
    except KeyboardInterrupt:
        print("\n" + "-" * 70)
        print("테스트 종료. 터미널 B의 nvidia-smi 로그와 비교해 보세요.")

if __name__ == "__main__":
    run_throttling_profile()