1. TF32 vs FP32 vs FP16/BF16 비교 분석
A100은 **TF32(Tensor Float 32)**를 처음 도입한 모델입니다. 이 기능이 실무에서 얼마나 유용한지 수치로 증명해 보는 실험입니다.

실험 방법: 동일한 PyTorch 모델(예: ResNet-50 또는 BERT) 학습 시, 옵션을 바꿔가며 측정합니다.

torch.backends.cuda.matmul.allow_tf32 = True/False

측정 지표: * Throughput (Samples/sec): 연산 속도 차이.

Convergence Accuracy: 낮은 정밀도를 썼을 때 최종 정확도가 얼마나 변하는지.

Power Consumption: 정밀도에 따른 GPU 소모 전력량 변화.

2. Multi-Instance GPU (MIG) 성능 격리(Isolation) 테스트
A100의 핵심 기능 중 하나는 GPU 한 장을 물리적으로 최대 7개로 쪼개는 MIG입니다.

실험 방법: 1.  GPU 전체(Full)를 사용할 때의 성능 측정.
2.  MIG로 7개 조각(1g.5gb)을 낸 뒤, 각 인스턴스에서 서로 다른 워크로드를 동시에 돌리며 성능 하락(Interference)이 있는지 측정.

분석 포인트: 가상화 환경(Docker 등) 대비 하드웨어 레벨의 자원 격리가 얼마나 완벽한지 확인해 볼 수 있습니다.

3. Roofline Model 기반의 Bottleneck 분석
A100의 하드웨어 스펙 한계치와 내 코드가 어디에 머물러 있는지 시각화하는 아주 전문적인 분석입니다.

도구: NVIDIA Nsight Compute

실험 방법: 특정 커널(Matrix Multiply, Convolution 등)을 실행하고 Roofline 차트를 생성합니다.

분석 포인트:

내 커널이 Compute-bound(연산량이 너무 많음)인지, Memory-bound(데이터 전송이 너무 느림)인지 파악합니다.

HBM2 메모리 대역폭(최대 약 2TB/s)을 실제로 얼마나 활용하고 있는지 확인합니다.

4. 구조적 희소성(Structured Sparsity) 2배 가속 테스트
A100부터 지원하는 기능으로, 가중치(Weights) 중 0인 값들을 압축해 성능을 2배 높이는 기술입니다.

실험 방법: NVIDIA의 apex.sparsity 라이브러리를 사용해 모델에 2:4 Sparsity를 적용합니다.

분석 포인트: 이론상 2배의 성능 향상이 전력 소모량과 추론 지연 시간(Latency)에 어떤 영향을 주는지 측정합니다.

5. 발열 및 쓰로틀링(Throttling) 프로파일링
H100과 마찬가지로 A100도 발열 관리가 중요합니다. 특히 서버용 SXM 모델은 400W를 소모하므로 냉각 성능에 따른 클럭 변화를 봅니다.

실험 방법: nvidia-smi를 초단위로 로깅하며 GPU 온도를 80도 이상으로 강제 유도(Stress Test)합니다.

분석 포인트: * 온도가 몇 도에서 클럭이 깎이기 시작하는지(Thermal Throttling 지점 찾기).

전력 공급이 부족할 때 발생하는 Power Throttling 현상 관찰.