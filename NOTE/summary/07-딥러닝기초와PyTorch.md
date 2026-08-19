# 딥러닝 기초와 PyTorch — 요약

> 출처 TIL: 260819

- **ML vs DL**: 머신러닝은 사람이 특징을 설계, 딥러닝은 모델이 데이터에서 특징 표현을 스스로 학습
- **딥러닝 파이프라인 순서**: import → config → data/dataloader → model → loss → optimizer → train loop → validation loop → logging → checkpoint
- **train loop 5단계**: `zero_grad()` → forward(`model(x)`) → loss 계산 → `backward()` → `step()`
- **validation loop**: `model.eval()` + `torch.no_grad()`로 gradient 없이 성능만 확인
- **Tensor**: PyTorch에서 숫자 데이터를 담는 기본 그릇. 이미지·텍스트·라벨 모두 결국 숫자로 변환되어 담김
- **shape**: 텐서 구조를 왼쪽→오른쪽 차원 순서로 표현 (스칼라 `[]` ~ 이미지 배치 `[32,3,224,224]`)
- **ndim/dtype/device**: 차원 수 / 값의 자료형 / 저장된 연산 장치(cpu, cuda)
- **Batch dimension**: 여러 샘플을 한 번에 처리하기 위해 텐서 맨 앞에 두는 차원. GPU 병렬 연산 효율을 위해 필수
- **데이터 종류별 shape 관례**: 표(batch,features) / 이미지(batch,channels,height,width) / RNN(seq_len,batch,features) / Transformer(batch,seq_len,features)
- **unsqueeze**: 지정한 위치에 크기 1인 차원을 추가 (예: 샘플 1개를 배치로 취급)
- **squeeze**: 크기 1인 차원을 제거 (인자 없으면 전부, 지정하면 그 위치만)
- **squeeze/unsqueeze 쓰임**: 모델 출력 `(batch,1)` 정리, 배치 차원 추가, shape 불일치로 인한 의도치 않은 브로드캐스팅 방지
- **Broadcasting 규칙**: shape을 오른쪽(마지막 차원)부터 비교해, 크기가 같거나 하나가 1이거나 없으면 통과
- **Broadcasting 실수 사례**: `(N,) - (N,1)`을 그대로 연산하면 `(N,N)`으로 잘못 확장됨 → `unsqueeze`로 shape을 맞춰야 함
- **실무 습관**: loss 계산 전 관련 텐서들의 `.shape`을 항상 확인
