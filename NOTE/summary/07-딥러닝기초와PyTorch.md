# 딥러닝 기초와 PyTorch — 요약

> 출처 TIL: 260819, 260820

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
- **MSELoss**: 회귀용. 예측-정답 차이의 제곱평균, 이상치에 민감
- **BCEWithLogitsLoss**: 이진분류용. Sigmoid+BCE를 결합, 모델은 raw logit 출력(Sigmoid 직접 붙이면 안 됨)
- **CrossEntropyLoss**: 다중분류용. Softmax+NLL을 결합, target은 정수 클래스 인덱스
- **device**: 텐서/모델이 올라간 연산 장치(cpu/cuda). 연산에 같이 쓰이는 텐서는 device가 같아야 함
- **표준 device 패턴**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 후 `.to(device)`
- **퍼셉트론**: 가중합+편향 후 임계값 넘으면 발화. `nn.Linear`의 원형. 계단함수는 미분 불가해 학습에 부적합
- **XOR 문제**: 단일 퍼셉트론은 선형 분리 가능한 문제만 풀 수 있음(1969년 증명, 1차 AI 겨울의 원인)
- **MLP**: Linear+활성화함수를 여러 층 쌓은 구조. 활성화함수 없으면 여러 층도 하나의 Linear와 동일해짐
- **보편 근사 정리**: 은닉층 하나(충분히 넓으면)로도 이론상 임의의 연속함수 근사 가능
- **nn.Linear**: 가중합+편향 레이어. `weight` shape `(out,in)`, `bias` shape `(out,)`, 마지막 차원만 `in_features`와 맞으면 됨
- **flatten**: 다차원 텐서를 벡터로 펼침. `start_dim=1`로 batch 차원 보존 필수(안 하면 배치가 뭉개짐)
- **nn.Flatten()**: `start_dim=1`이 기본값이라 배치 보존이 자동, CNN→Linear 연결에 사용
- **view**: 항상 메모리 공유(뷰), 원본이 연속(contiguous)이어야 함 — 아니면 에러
- **reshape**: 가능하면 공유, 안 되면 자동 복사 — 항상 성공하는 안전한 선택
- **flatten vs view/reshape**: flatten은 차원을 합치기만 가능(임의 재배열 불가), view/reshape은 임의 shape 변경 가능
- **-1의 의미**: view/reshape/flatten 모두 `-1`을 넣으면 나머지 차원 크기를 자동 계산
