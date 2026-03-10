"""
Jetson AGX Xavier — nvpmodel 전력 모드별 성능/전력 효율 특성 분석

사용법:
  sudo python3 power_perf.py               # 현재 모드에서 측정
  sudo python3 power_perf.py --all-modes   # 모든 모드 순회 비교
"""

import torch
import subprocess
import time
import threading
import sys
import os
import re

# ─── 설정 ───
DIM        = 2048
BATCH_SIZE = 512
ITERATIONS = 300
WARMUP     = 50


# ─── tegrastats 파싱으로 전력 읽기 ───
# 출력 예: GPU 1234mW/1234mW CPU 389mW/389mW SOC 973mW/972mW SYS5V 2384mW/2384mW
POWER_PATTERN = re.compile(
    r'GPU (\d+)mW/\d+mW\s+CPU (\d+)mW/\d+mW\s+SOC (\d+)mW/\d+mW'
    r'.*?SYS5V (\d+)mW/\d+mW'
)

def read_tegrastats_once():
    """tegrastats를 1회 실행해서 전력값(mW) 반환. sudo 필요."""
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True
        )
        for line in proc.stdout:
            m = POWER_PATTERN.search(line)
            if m:
                proc.terminate()
                gpu_mw  = int(m.group(1))
                cpu_mw  = int(m.group(2))
                soc_mw  = int(m.group(3))
                sys5v_mw = int(m.group(4))
                return {"GPU": gpu_mw, "CPU": cpu_mw, "SOC": soc_mw, "SYS5V": sys5v_mw}
        proc.terminate()
    except Exception as e:
        pass
    return None


class PowerMonitor:
    """백그라운드에서 tegrastats를 파싱해 전력을 샘플링"""
    def __init__(self, interval_ms=200):
        self.interval = interval_ms
        self.samples = []   # SYS5V (총 입력 전력) mW 목록
        self.detail  = []   # {"GPU","CPU","SOC","SYS5V"} 딕셔너리 목록
        self._proc   = None
        self._thread = None
        self._running = False

    def start(self):
        self.samples = []
        self.detail  = []
        self._running = True
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            m = POWER_PATTERN.search(line)
            if m:
                d = {
                    "GPU":   int(m.group(1)),
                    "CPU":   int(m.group(2)),
                    "SOC":   int(m.group(3)),
                    "SYS5V": int(m.group(4)),
                }
                self.detail.append(d)
                self.samples.append(d["SYS5V"])

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        if self._thread:
            self._thread.join(timeout=2)

    def avg(self):
        if not self.detail:
            return None
        n = len(self.detail)
        return {k: sum(d[k] for d in self.detail) / n for k in self.detail[0]}


def get_nvpmodel_mode():
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "NV Power Mode" in line:
                return line.strip()
        return r.stdout.strip()
    except:
        return "unknown"


def get_gpu_freq():
    paths = [
        "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq",
        "/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return int(f.read().strip()) / 1_000_000
    return 0


def run_bench():
    print("=" * 55)
    print("  Jetson AGX Xavier — 전력/성능 효율 분석")
    print("=" * 55)

    mode     = get_nvpmodel_mode()
    gpu_freq = get_gpu_freq()
    device   = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"

    print(f"Device   : {device}")
    print(f"NVPModel : {mode}")
    print(f"GPU Freq : {gpu_freq:.0f} MHz")
    print(f"Matrix   : ({BATCH_SIZE}, {DIM}) x ({DIM}, {DIM})")
    print()

    weight     = torch.randn(DIM, DIM, device='cuda', dtype=torch.float16)
    input_data = torch.randn(BATCH_SIZE, DIM, device='cuda', dtype=torch.float16)

    print("Warming up...", end=" ", flush=True)
    for _ in range(WARMUP):
        torch.mm(input_data, weight)
    torch.cuda.synchronize()
    print("done")

    mon = PowerMonitor(interval_ms=200)
    mon.start()

    print("Benchmarking...", end=" ", flush=True)
    times = []
    for _ in range(ITERATIONS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.mm(input_data, weight)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    mon.stop()
    print("done\n")

    avg_ms = sum(times) / len(times)
    tflops = (2 * BATCH_SIZE * DIM * DIM) / (avg_ms * 1e-3) / 1e12
    avg_pwr = mon.avg()

    print("─" * 55)
    print(f"평균 지연 시간  : {avg_ms:.4f} ms")
    print(f"연산 성능       : {tflops:.3f} TFLOPS")

    if avg_pwr:
        total_w = avg_pwr["SYS5V"] / 1000
        gpu_w   = avg_pwr["GPU"]   / 1000
        cpu_w   = avg_pwr["CPU"]   / 1000
        soc_w   = avg_pwr["SOC"]   / 1000
        eff     = tflops / total_w * 1000  # GFLOPS/W
        print(f"전력 (SYS5V)   : {total_w:.2f} W  (GPU {gpu_w:.2f}W / CPU {cpu_w:.2f}W / SOC {soc_w:.2f}W)")
        print(f"전력 효율       : {eff:.2f} GFLOPS/W")
        print(f"샘플 수         : {len(mon.samples)}")
    else:
        print("전력 센서       : 읽기 실패 (sudo로 실행하세요)")
    print("─" * 55)

    return avg_ms, tflops, avg_pwr


def run_all_modes():
    print("모든 nvpmodel 모드 순회...\n")
    results = []
    for mode_id in range(8):
        print(f"\n{'='*55}")
        print(f"  nvpmodel 모드 {mode_id} 전환 중...")
        ret = subprocess.run(["nvpmodel", "-m", str(mode_id)], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  모드 {mode_id} 미지원, 건너뜀")
            continue
        subprocess.run(["jetson_clocks"], capture_output=True)
        time.sleep(3)
        avg_ms, tflops, pwr = run_bench()
        total_w = pwr["SYS5V"] / 1000 if pwr else 0
        results.append((mode_id, avg_ms, tflops, total_w))

    if results:
        print(f"\n{'='*55}")
        print("  전체 모드 비교 요약")
        print(f"{'='*55}")
        print(f"{'모드':>4} | {'지연(ms)':>10} | {'성능(TFLOPS)':>12} | {'전력(W)':>8} | {'효율(GFLOPS/W)':>14}")
        print("-" * 58)
        for mid, ms, tf, pw in results:
            eff = tf / pw * 1000 if pw > 0 else 0
            print(f"{mid:>4} | {ms:>10.4f} | {tf:>12.3f} | {pw:>8.2f} | {eff:>14.2f}")


MODES = {
    0: "MAXN",
    1: "MODE_10W",
    2: "MODE_15W",
    3: "MODE_30W_ALL",
    4: "MODE_30W_6CORE",
    5: "MODE_30W_4CORE",
    6: "MODE_30W_2CORE",
    7: "MODE_15W_DESKTOP",
}

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("[경고] tegrastats 전력 읽기는 sudo가 필요합니다.")
        print("  → sudo python3 power_perf.py")
        sys.exit(1)

    # --mode <id> 로 특정 모드 지정
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode_id = int(sys.argv[idx + 1])
        if mode_id not in MODES:
            print(f"유효한 모드: {list(MODES.keys())}")
            sys.exit(1)
        print(f"nvpmodel 모드 {mode_id} ({MODES[mode_id]}) 로 전환 중...")
        ret = subprocess.run(["nvpmodel", "-m", str(mode_id)], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"모드 전환 실패: {ret.stderr}")
            sys.exit(1)
        time.sleep(2)
        run_bench()

    elif "--all-modes" in sys.argv:
        run_all_modes()

    else:
        # 모드 목록 출력 후 현재 모드로 벤치
        print("사용 가능한 모드:")
        for mid, name in MODES.items():
            print(f"  {mid}: {name}")
        print()
        print("특정 모드 지정: sudo python3 power_perf.py --mode <id>")
        print("모든 모드 순회: sudo python3 power_perf.py --all-modes")
        print()
        run_bench()
