"""
Jetson AGX Xavier — nvpmodel 전력 모드별 성능/전력 효율 특성 분석

Xavier는 nvpmodel로 전력 모드를 전환할 수 있다:
  MODE 0: MAXN (30W, 8코어, GPU 최대 클럭)
  MODE 1: 10W (2코어, GPU 제한)
  MODE 2: 15W (4코어, GPU 제한)
  MODE 3: 30W (4코어, GPU 최대)
  ... (JetPack 버전에 따라 다름)

이 코드는:
1. 현재 nvpmodel 모드에서 GPU 연산 성능 측정 (GEMM)
2. 실시간 전력 소비 모니터링 (tegrastats / INA3221 센서)
3. 성능/와트 (TFLOPS/W) 효율 계산
4. 여러 모드에서 반복 실행하면 모드별 비교 가능

사용법:
  # 현재 모드에서 벤치마크
  python3 power_perf.py

  # 모든 모드 자동 순회 (sudo 필요)
  sudo python3 power_perf.py --all-modes
"""

import torch
import subprocess
import time
import threading
import sys
import os

# ─── 설정 ───
DIM = 2048          # 행렬 크기 (Xavier 메모리 고려)
BATCH_SIZE = 512
ITERATIONS = 300
WARMUP = 50

# ─── 전력 모니터링 (INA3221 센서) ───
POWER_PATHS = [
    # Xavier의 전력 센서 경로 (JetPack 4.x / 5.x)
    "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",  # GPU
    "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input",  # CPU
    "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power2_input",  # SoC
    # 대체 경로 (JetPack 버전에 따라)
    "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/power1_input",     # VDD_GPU_SOC
    "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/power2_input",     # VDD_CPU_CV
    "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon1/power1_input",     # VDD_IN (총 입력)
]

def find_power_sensor():
    """사용 가능한 전력 센서 경로를 자동으로 찾음"""
    # 먼저 하드코딩된 경로 확인
    for p in POWER_PATHS:
        if os.path.exists(p):
            return p

    # hwmon에서 자동 탐색
    hwmon_base = "/sys/class/hwmon/"
    if os.path.exists(hwmon_base):
        for hw in sorted(os.listdir(hwmon_base)):
            hw_path = os.path.join(hwmon_base, hw)
            name_file = os.path.join(hw_path, "name")
            if os.path.exists(name_file):
                with open(name_file) as f:
                    name = f.read().strip()
                if "ina3221" in name:
                    # power1_input = 첫 번째 채널 전력 (mW)
                    for i in range(1, 4):
                        pp = os.path.join(hw_path, f"power{i}_input")
                        if os.path.exists(pp):
                            return pp
    return None


def read_power_mw(sensor_path):
    """전력 센서에서 현재 전력(mW) 읽기"""
    try:
        with open(sensor_path) as f:
            return float(f.read().strip())
    except:
        return 0.0


def get_total_power_mw():
    """tegrastats 없이 직접 센서에서 총 전력 읽기 (mW)"""
    total = 0.0
    hwmon_base = "/sys/class/hwmon/"
    if not os.path.exists(hwmon_base):
        return 0.0
    for hw in sorted(os.listdir(hwmon_base)):
        hw_path = os.path.join(hwmon_base, hw)
        name_file = os.path.join(hw_path, "name")
        if os.path.exists(name_file):
            with open(name_file) as f:
                name = f.read().strip()
            if "ina3221" in name:
                for i in range(1, 4):
                    pp = os.path.join(hw_path, f"power{i}_input")
                    if os.path.exists(pp):
                        try:
                            with open(pp) as f:
                                total += float(f.read().strip())
                        except:
                            pass
    return total


class PowerMonitor:
    """백그라운드에서 전력 소비를 주기적으로 샘플링"""
    def __init__(self, interval_ms=50):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None
        self.sensor_path = find_power_sensor()

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _sample_loop(self):
        while self._running:
            power = get_total_power_mw()
            if power > 0:
                self.samples.append(power)
            time.sleep(self.interval)

    def avg_power_w(self):
        if not self.samples:
            return 0.0
        return (sum(self.samples) / len(self.samples)) / 1000.0  # mW → W


def get_nvpmodel_mode():
    """현재 nvpmodel 모드 번호와 이름"""
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "NV Power Mode" in line:
                return line.strip()
        return result.stdout.strip()
    except:
        return "unknown (nvpmodel not available)"


def get_gpu_freq():
    """현재 GPU 클럭 (MHz)"""
    freq_paths = [
        "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq",
        "/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq",
    ]
    for p in freq_paths:
        if os.path.exists(p):
            with open(p) as f:
                return int(f.read().strip()) / 1_000_000  # Hz → MHz
    return 0


def run_bench():
    print("=" * 55)
    print("  Jetson AGX Xavier — 전력/성능 효율 분석")
    print("=" * 55)

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    mode = get_nvpmodel_mode()
    gpu_freq = get_gpu_freq()

    print(f"Device      : {device}")
    print(f"NVPModel    : {mode}")
    print(f"GPU Freq    : {gpu_freq:.0f} MHz")
    print(f"Matrix      : ({BATCH_SIZE}, {DIM}) x ({DIM}, {DIM})")
    print(f"Iterations  : {ITERATIONS}")
    print()

    # 데이터 준비
    weight = torch.randn(DIM, DIM, device='cuda', dtype=torch.float16)
    input_data = torch.randn(BATCH_SIZE, DIM, device='cuda', dtype=torch.float16)

    # Warm-up
    print("Warming up...", end=" ", flush=True)
    for _ in range(WARMUP):
        torch.mm(input_data, weight)
    torch.cuda.synchronize()
    print("done")

    # 전력 모니터링 시작
    power_mon = PowerMonitor(interval_ms=50)
    power_mon.start()

    # 벤치마크
    print("Benchmarking...", end=" ", flush=True)
    times = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.mm(input_data, weight)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    power_mon.stop()
    print("done\n")

    # 결과 계산
    avg_ms = sum(times) / len(times)
    flops = 2 * BATCH_SIZE * DIM * DIM
    tflops = flops / (avg_ms * 1e-3) / 1e12
    avg_power = power_mon.avg_power_w()

    print("─" * 55)
    print(f"평균 지연 시간     : {avg_ms:.4f} ms")
    print(f"연산 성능          : {tflops:.3f} TFLOPS")

    if avg_power > 0:
        efficiency = tflops / avg_power * 1000  # GFLOPS/W
        print(f"평균 전력 소비     : {avg_power:.2f} W")
        print(f"전력 효율          : {efficiency:.2f} GFLOPS/W")
        print(f"전력 샘플 수       : {len(power_mon.samples)}")
    else:
        print(f"전력 센서          : 감지 불가 (권한 확인 필요)")
        print(f"  → sudo로 실행하거나 센서 경로를 확인하세요")

    print("─" * 55)
    return avg_ms, tflops, avg_power


def run_all_modes():
    """모든 nvpmodel 모드를 순회하며 벤치마크"""
    print("모든 nvpmodel 모드 순회 중...\n")

    # Xavier 모드 목록 (JetPack 4.x 기준)
    modes = [0, 1, 2, 3, 4, 5, 6, 7]
    results = []

    for mode_id in modes:
        print(f"\n{'='*55}")
        print(f"  nvpmodel 모드 {mode_id} 로 전환 중...")
        ret = subprocess.run(["nvpmodel", "-m", str(mode_id)], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  모드 {mode_id} 전환 실패 (지원하지 않는 모드일 수 있음), 건너뜀")
            continue

        # jetson_clocks로 최대 클럭 고정 (일관된 측정)
        subprocess.run(["jetson_clocks"], capture_output=True)
        time.sleep(3)  # 안정화 대기

        avg_ms, tflops, power_w = run_bench()
        results.append((mode_id, avg_ms, tflops, power_w))

    # 요약
    if results:
        print(f"\n{'='*55}")
        print("  전체 모드 비교 요약")
        print(f"{'='*55}")
        print(f"{'모드':>4} | {'지연(ms)':>10} | {'성능(TFLOPS)':>12} | {'전력(W)':>8} | {'효율(GFLOPS/W)':>14}")
        print("-" * 60)
        for mode_id, ms, tf, pw in results:
            eff = (tf / pw * 1000) if pw > 0 else 0
            print(f"{mode_id:>4} | {ms:>10.4f} | {tf:>12.3f} | {pw:>8.2f} | {eff:>14.2f}")


if __name__ == "__main__":
    if "--all-modes" in sys.argv:
        if os.geteuid() != 0:
            print("모든 모드 순회는 sudo가 필요합니다.")
            print("사용법: sudo python3 power_perf.py --all-modes")
            sys.exit(1)
        run_all_modes()
    else:
        run_bench()
