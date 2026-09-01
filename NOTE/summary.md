# 전체 요약

주제별로 묶은 키워드 요약입니다. 더 깊은 설명이 필요하면 각 섹션 제목에 대응하는 `NOTE/detail/NN-슬러그.md` 파일을 참고하세요.

## 01. 선형대수와 행렬분해

> 출처 TIL: 260805, 260806, 260807

- **스칼라·벡터·행렬**: 축이 0개/1개/2개인 숫자 표현
- **행렬식**: 행렬이 공간을 얼마나 부풀리거나 누르는지 나타내는 값. 0이면 되돌릴 수 없는 특이행렬
- **역행렬**: 어떤 변환을 정확히 원상복구하는 행렬. 특이행렬에는 존재하지 않음
- **단위행렬**: 아무것도 바꾸지 않는 변환. 역행렬 검산 도구
- **전치**: 행과 열을 맞바꾸는 것. 데이터 값 자체는 불변
- **원소별 곱 vs 행렬곱**: 원소별 곱은 shape이 같아야 함, 행렬곱은 앞의 열=뒤의 행 개수만 맞으면 됨
- **비가환성**: 행렬곱은 순서를 바꾸면 결과가 달라짐(AB ≠ BA)
- **선형변환**: 가산성·동차성·원점보존을 만족하는 변환. 대표적으로 스케일링·반사·투영·회전
- **코사인 유사도**: 두 벡터의 방향이 얼마나 비슷한지 측정 (같은 방향/수직/반대 방향)
- **완전연결층**: 입력에 가중치를 행렬곱하고 편향을 더하는 층. 여러 층을 쌓으면 신경망
- **Span**: 벡터들로 만들 수 있는 모든 점의 집합(직선/평면/공간)
- **Rank**: Span의 차원, 즉 서로 독립적인 방향의 개수
- **조건수**: 입력의 작은 변화에 결과가 얼마나 민감한지. 클수록 신뢰도 낮음
- **유일해 vs 무한해**: rank와 변수 개수가 같으면 유일해, 작으면 무한해
- **대각화**: 정사각행렬을 고유벡터·고유값·역행렬로 분해. PCA의 근간
- **회귀계수**: 예측 오차가 입력의 모든 방향과 직교하는 지점에서 구해지는 최적 가중치
- **잔차**: 실제값-예측값. 오차제곱합의 기초
- **직교성**: 내적이 0이면 직교. 직교행렬은 전치만으로 역행렬을 구할 수 있음
- **LU/QR/고유분해/SVD**: 각각 연립방정식 풀이/최소제곱·고유값/동역학/차원축소·PCA에 사용
- **고유값**: 방향은 안 바뀌고 크기만 바뀌는 축(고유벡터)에서의 배율
- **배치 행렬곱**: 마지막 두 축만 행렬곱, 앞의 축은 배치로 취급
- **중심화**: 각 열의 평균을 0으로 만드는 전처리. 공분산행렬 연결에 필수
- **SVD와 PCA**: PCA는 목표, SVD는 그 목표를 구현하는 표준 계산 도구
- **저랭크**: 소수의 핵심 패턴으로 대부분이 설명되는 상태. 압축·노이즈제거·추천시스템에 활용

## 02. 트랜스포머 수학

> 출처 TIL: 260810

- **q @ K.T (벡터×행렬)**: 결과는 키 개수만큼의 어텐션 스코어 벡터
- **Q @ Kᵀ (행렬×행렬)**: 결과는 (쿼리 개수 × 키 개수)의 정방 스코어 행렬
- **swapaxes vs .T**: 배치 텐서에선 .T가 모든 축을 뒤집어 위험, swapaxes(-1,-2)는 마지막 두 축만 안전하게 교환
- **Softmax 적용 축**: 항상 key_len(마지막) 축. 쿼리 축에 적용하면 의미가 뒤바뀜
- **Causal Mask**: 미래 위치에 매우 큰 음수를 채워 Softmax 후 확률을 0으로 만듦
- **Attention weight @ V**: key_len 차원이 소거되고 (query_len × d_v) 출력. 실제 정보는 V만 전달
- **CrossEntropyLoss엔 logits**: 확률을 넣으면 이중 Softmax·수치불안정·gradient 왜곡 발생
- **스케일링 vs Temperature**: 스케일링은 학습 안정화용(내부, 고정), Temperature는 출력 조절용(추론, 가변)
- **미분 부호**: 양수=입력 늘리면 증가, 음수=입력 늘리면 감소
- **편미분**: 미분 대상 외 변수는 전부 상수로 취급
- **Gradient 반대 방향 이동**: Gradient는 증가 방향을 가리키므로 최소화하려면 반대로 이동
- **Gradient≈0의 함정**: 전역/지역 최솟값·안장점·평지를 구분 못함. 고차원에선 안장점이 더 흔함
- **연쇄법칙(Chain Rule)**: 합성함수 미분 = 각 단계 변화율의 곱. 역전파의 수학적 기반
- **Local/Upstream/Downstream Gradient**: Downstream = Local × Upstream
- **경로 분기 후 합류**: 각 경로의 gradient를 계산해 더함(다변수 연쇄법칙)
- **NLL과 정답확률**: 정답확률이 낮아질수록 NLL 손실은 급격히 커짐

## 03. 지도학습·비지도학습 기초 모델

> 출처 TIL: 260811

- **지도학습**: 입력-정답 쌍으로 학습, 회귀(연속값)/분류(범주) 두 유형
- **회귀 평가지표**: MSE/RMSE, MAE, R²
- **분류 평가지표**: 혼동행렬(TP/FP/FN/TN), 정확도, 정밀도, 재현율, F1, ROC-AUC
- **클래스 불균형 대응**: SMOTE 등 오버샘플링, 언더샘플링, 클래스 가중치, F1·AUC 활용
- **비지도학습**: 레이블 없이 구조를 스스로 찾음
- **군집화**: 비슷한 데이터끼리 묶기 (K-means, 계층적, DBSCAN, GMM). 엘보우/실루엣으로 K 결정
- **차원 축소**: 정보를 보존하며 특징 수 줄이기 (PCA-선형, t-SNE/UMAP/오토인코더-비선형)
- **이상치 탐지**: 정상 패턴에서 벗어난 데이터 탐지. 극심한 불균형·레이블 부족이 특징
- **밀도 추정**: 데이터가 따르는 확률분포 자체를 추정 (모수적 GMM vs 비모수적 KDE)
- **train/valid/test 분리**: 학습에 안 쓴 데이터로 일반화 성능을 정직하게 측정하기 위함
- **선형/로지스틱 회귀**: 같은 선형결합 뼈대, 출력변환과 손실함수만 다름(MSE vs 교차엔트로피)
- **KNN**: 가까운 k개 이웃의 다수결/평균으로 예측. 학습 없음(게으른 학습), 스케일링 필수
- **결정트리**: if-then 규칙으로 반복 분할. 해석 쉽지만 분산 크고 과적합 쉬움 → 가지치기 필요
- **K-means**: 데이터를 K개 군집으로 분할, 거리 기반이라 스케일링 필수
- **PCA**: 분산이 큰 방향으로 데이터를 투영해 차원 축소, 공분산행렬의 고유값 분해로 계산
- **PCA+K-means 파이프라인**: 먼저 PCA로 차원 축소 후 K-means로 군집화(차원의 저주 완화)
- **모델 선택 기준**: 데이터 규모, 해석가능성 vs 성능, 선형성, 스케일 민감도, 계산비용, 불균형, 차원수
- **실전 워크플로우**: 단순 베이스라인 → 비선형 모델 개선 → 교차검증으로 편향/분산 점검 → 앙상블/튜닝

## 04. 앙상블 학습

> 출처 TIL: 260812

- **부트스트랩**: 복원추출로 여러 가상 데이터셋 생성. 약 63.2%만 포함, 나머지는 OOB
- **배깅**: 부트스트랩 데이터셋마다 모델 학습 후 평균/투표. 분산만 선택적으로 감소
- **랜덤포레스트의 두 무작위성**: 데이터 샘플링(배깅) + 특징 샘플링 → 트리 간 상관관계 최소화
- **랜덤포레스트가 강해지는 조건**: 개별 트리는 정확해야 하고(강도), 트리끼리는 달라야 함(다양성)
- **OOB Score**: 학습에 안 쓰인 약 36.8% 데이터로 별도 검증셋 없이 성능 추정
- **MDI**: 불순도 감소 기반, 빠르지만 카디널리티 높은 변수 과대평가
- **Permutation Importance**: 값을 섞어 성능 하락폭 측정. 신뢰도 높지만 계산비용 크고 상관특징 과소평가
- **부스팅**: 순차적으로 약한 학습기를 오차 보완하며 추가. 편향을 줄이는 전략 (배깅은 분산 감소)
- **AdaBoost**: 틀린 샘플에 가중치 부여 / **Gradient Boosting**: 잔차(경사)를 학습, 더 유연
- **XGBoost**: 레벨단위 성장, 2차근사+정규화로 정확도↑, 안정적이나 느림
- **LightGBM**: 리프단위 성장, 히스토그램/GOSS/EFB로 가장 빠름, 소규모 데이터엔 과적합 위험
- **CatBoost**: Ordered Target Statistics·Ordered Boosting으로 범주형 변수·타겟누수에 강함
- **학습률·반복수·Early Stopping**: 삼각관계. 학습률 낮게+반복수 크게+Early Stopping으로 자동 최적화
- **공정 비교 4원칙**: 같은 fold, 같은 전처리 경계, 같은 평가지표, 같은 계산조건
- **SHAP**: 섀플리 값 기반, 개별 예측 단위(지역)로 기여도를 공리적으로 유일하게 공정 배분
- **TreeSHAP**: 트리 구조를 활용해 정확한 섀플리 값을 빠르게 계산. 지역해석을 집계해 전역해석 획득
- **SHAP 시각화 3종**: Bar(전역 요약)→Beeswarm(전역 패턴+방향성)→Waterfall(개별 샘플 완전분해)

## 05. 편향-분산과 규제

> 출처 TIL: 260813

- **편향(Bias)**: 모델이 너무 단순해 패턴을 못 잡는 정도. 높으면 과소적합
- **분산(Variance)**: 훈련 데이터 변화에 예측이 얼마나 흔들리는지. 높으면 과적합
- **편향-분산 트레이드오프**: 복잡도↑ → 편향↓ 분산↑ / 복잡도↓ → 편향↑ 분산↓
- **기대오차 3요소**: 편향 요소 + 분산 요소 + 데이터의 본질적 잡음(줄일 수 없음)
- **학습곡선**: x=데이터 크기. 높이=편향, 간격=분산, 끝기울기=데이터 추가 효과 여부
- **검증곡선**: x=모델 복잡도. 검증오차가 U자형, 바닥이 최적 복잡도
- **Train-Valid Gap**: 분산의 실측 근사치. 작고 둘다 나쁘면 과소적합, 크면 과적합
- **Fold/k-fold CV**: 분할 우연성을 줄이기 위해 k번 나눠 평균. Stratified k-fold는 클래스 비율 유지
- **Train AP vs Val AP Gap**: 작고 높으면 이상적, 크면 과적합(데이터부족·증강부족·모델용량과다 등이 원인)
- **고편향 대응**: 모델 복잡도↑, 특성 추가, 규제 완화, 학습 더 오래
- **고분산 대응**: 데이터 추가, 규제 강화, 조기종료, 모델 단순화, 앙상블
- **조기 종료**: validation 성능 정체 시 학습 중단. epoch 수를 규제 대상으로 삼는 기법
- **L1(Lasso)**: 절댓값 페널티, 일부 계수를 0으로(자동 특성선택), 상관특성엔 불안정
- **L2(Ridge)**: 제곱합 페널티, 계수를 고르게 축소(0은 안됨), 다중공선성에 강건, 닫힌 형태 해
- **ElasticNet**: L1+L2 절충, 상관특성 그룹단위 선택, 고차원+상관특성에 적합
- **규제 세기 하이퍼파라미터**: 0=규제없음(과적합), 크면 과소적합. 검증곡선으로 튜닝
- **L1비율 하이퍼파라미터**: ElasticNet 전용, 1=순수Lasso, 0=순수Ridge

## 06. 평가지표와 데이터 누수

> 출처 TIL: 250814, 250818

- **클래스 불균형**: 클래스별 데이터 수가 크게 차이 남. 정확도만으로 평가 왜곡
- **혼동행렬 4요소**: TP(맞은양성)/FP(오탐)/FN(놓침)/TN(맞은음성)
- **양성률**: 실제 양성 비율(prevalence) 또는 예측 양성 비율
- **F1-score**: 정밀도·재현율의 조화평균. 한쪽이 극단적으로 나쁘면 점수도 낮게 반영
- **ROC-AUC**: 위양성률-재현율 곡선의 면적. 불균형 시 과대평가되기 쉬움
- **AP(PR-AUC)**: 재현율-정밀도 곡선의 면적. 불균형 데이터에 더 현실적
- **임계값(Threshold)**: 확률을 클래스로 변환하는 기준선. 낮추면 재현율↑정밀도↓, 높이면 반대
- **SMOTE**: 소수 클래스 샘플 사이를 보간해 합성 샘플 생성. 반드시 train에만 적용
- **봉인 테스트 원칙**: test는 최종 평가 직전까지 절대 열어보지 않음. 전처리/튜닝/모델선택에 관여 금지
- **분할 단위 선택**: "배포 후 새로 만날 대상"에 맞춰 분할(그룹 단위/시간 단위/무작위)
- **KFold**: 그룹·클래스 비율 미고려 / **StratifiedKFold**: 클래스 비율 유지
- **GroupKFold**: 그룹 겹침 방지 / **StratifiedGroupKFold**: 그룹+클래스 비율 동시 고려
- **TimeSeriesSplit**: 항상 과거로 학습, 미래로 검증. gap으로 라벨 확정 지연 반영
- **5대 누수 유형**: 분할 누수, 전처리 누수, Target·미래 누수, 리샘플링 누수, 선택 누수
- **올바른 파이프라인 순서**: 분할(봉인)→학습 데이터 내 전처리·리샘플링·튜닝→봉인된 test에 1회 적용
- **sklearn Pipeline**: 전처리·리샘플링·선택 누수는 구조적으로 방지
- **Pipeline이 못 막는 것**: 분할 단위 오류, Target·미래 누수, 외부 특성 테이블 누수, 라벨 정의 오류
- **GridSearchCV**: 하이퍼파라미터 후보 전 조합을 CV로 평가 (전수 탐색, 느림)
- **RandomizedSearchCV**: 후보를 무작위 n회만 샘플링해 CV로 평가 (빠름, 확률적)
- **탐색 실전 팁**: RandomizedSearchCV로 대략 탐색 → 근처 범위 GridSearchCV로 정밀 탐색
- **ColumnTransformer**: 컬럼 그룹마다 다른 전처리(스케일링/인코딩)를 적용 후 결과를 합침
- **Over-sampling 종류**: RandomOverSampler(단순복제), SMOTE(보간), ADASYN(어려운 영역 집중), BorderlineSMOTE(경계 위주)
- **Under-sampling 종류**: RandomUnderSampler(무작위 제거), TomekLinks/ENN(경계·노이즈 정리), NearMiss(경계 근접 다수클래스만 유지)
- **결합 sampler**: SMOTEENN, SMOTETomek — 오버샘플링 후 언더샘플링으로 정리
- **파이프라인 저장/재현성**: joblib으로 확정 파이프라인 저장 → reload 후 예측값이 저장 전과 동일한지(np.allclose) 검증
- **TPE(Optuna 기본 sampler)**: 과거 시도를 좋은/나쁜 그룹으로 나눠 분포 추정 후, l(x)/g(x) 비율이 큰 지점을 다음 후보로 제안 (베이지안 최적화)
- **Tree-structured**: 조건부 하이퍼파라미터(트리 구조 탐색공간)를 자연스럽게 다룸 — Gaussian Process 기반보다 유리
- **TPE vs Grid/Random**: 과거 결과를 활용해 유망한 영역으로 수렴, pruning(조기 종료)도 지원

## 07. 딥러닝 기초와 PyTorch

> 출처 TIL: 260819, 260820, 260821, 260824, 260825, 260826, 260827, 260828

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
- **Weight(가중치)**: 입력에 곱해지는 값. 그 입력이 출력에 얼마나(어느 방향으로) 영향을 주는지를 학습으로 결정
- **Bias(편향)**: 입력과 무관하게 더해지는 상수. 직선/결정경계가 원점을 지나지 않아도 되게 평행이동시켜줌
- **XOR 문제**: 단일 퍼셉트론은 선형 분리 가능한 문제만 풀 수 있음(1969년 증명, 1차 AI 겨울의 원인)
- **MLP**: Linear+활성화함수를 여러 층 쌓은 구조. 활성화함수 없으면 여러 층도 하나의 Linear와 동일해짐
- **보편 근사 정리**: 은닉층 하나(충분히 넓으면)로도 이론상 임의의 연속함수 근사 가능
- **nn.Linear**: 가중합+편향 레이어. `weight` shape `(out,in)`, `bias` shape `(out,)`, 마지막 차원만 `in_features`와 맞으면 됨
- **비선형성 필요 이유(수식)**: Linear 여러 개를 이어붙여도 결국 새 weight·bias를 가진 하나의 선형 변환으로 합쳐짐 → 비선형 활성화함수가 있어야 층을 쌓는 의미가 생김
- **flatten**: 다차원 텐서를 벡터로 펼침. `start_dim=1`로 batch 차원 보존 필수(안 하면 배치가 뭉개짐)
- **nn.Flatten()**: `start_dim=1`이 기본값이라 배치 보존이 자동, CNN→Linear 연결에 사용
- **view**: 항상 메모리 공유(뷰), 원본이 연속(contiguous)이어야 함 — 아니면 에러
- **reshape**: 가능하면 공유, 안 되면 자동 복사 — 항상 성공하는 안전한 선택
- **flatten vs view/reshape**: flatten은 차원을 합치기만 가능(임의 재배열 불가), view/reshape은 임의 shape 변경 가능
- **-1의 의미**: view/reshape/flatten 모두 `-1`을 넣으면 나머지 차원 크기를 자동 계산
- **ReLU**: `max(0, x)`. 음수는 0, 양수는 그대로. 비선형성 부여, 계산 단순, gradient vanishing 완화
- **Dying ReLU**: 음수 입력만 계속 받는 뉴런은 gradient가 항상 0이라 영영 업데이트 안 됨 → Leaky ReLU 등으로 보완
- **Sigmoid**: 실수를 0~1로 압축(`1/(1+e^-x)`). 이진분류 확률 출력용. 큰/작은 입력에서 gradient vanishing 있어 은닉층엔 잘 안 씀
- **BCEWithLogitsLoss 수치 안정성**: Sigmoid+log를 따로 계산하면 극단 logit에서 `log(0)=-inf` 위험 → log-sum-exp 트릭으로 결합 계산해 안정적
- **Sigmoid 추론 패턴**: 학습 시 loss 안에 숨김(모델은 raw logit 출력) → 추론 시에만 `torch.sigmoid(logits)`로 명시적 확률 변환
- **logits>0 vs sigmoid>0.5**: threshold 0.5 고정이면 수학적으로 동일 — label만 필요하면 Sigmoid 생략 가능
- **Softmax**: logit들을 지수함수로 정규화해 합이 1인 확률 분포로 변환. 다중분류에서 Sigmoid 역할
- **Softmax와 dim**: Sigmoid는 원소별 독립 연산이라 dim 불필요, Softmax는 그룹(축) 전체 합으로 정규화하므로 `dim` 필수(보통 `dim=-1`)
- **CrossEntropyLoss targets**: BCEWithLogitsLoss는 float(0/1) targets, CrossEntropyLoss는 정수 클래스 인덱스(long) targets — 원-핫 아님
- **argmax**: 텐서에서 가장 큰 값의 인덱스를 반환
- **argmax vs softmax 순서**: Softmax는 단조증가 함수라 `argmax(logits)`와 `argmax(softmax(logits))` 결과가 항상 동일
- **확률 수치가 필요한 실무 상황**: 신뢰도 기반 threshold 라우팅, 추천/CTR 등 확률 자체가 산출물, calibration 검증, top-k 후보 제시, beam search
- **Sigmoid vs Softmax**: Sigmoid는 원소 단위 독립 계산(이진분류), Softmax는 축 전체 상대적 정규화(다중분류, dim 필요)
- **손실 함수의 4가지 역할**: 오차 정량화 / 최적화 방향 제시 / 역전파(backward) 시작점 / 문제 유형별 "좋은 모델"의 정의
- **목표 함수(objective function)**: 최적화 대상 함수 전반을 가리키는 상위 개념(⊃ 손실 함수). 지도학습에서는 목표 함수=손실 함수(최소화), RL에서는 보상 최대화가 목표 함수인 경우가 많음. 규제 항이 붙으면 목표 함수=손실+λ·규제항
- **MAE (`nn.L1Loss`)**: 오차 절댓값 평균. 이상치에 덜 민감하나 0 근처에서 미분 불연속
- **Huber Loss (`nn.SmoothL1Loss`)**: 오차 작으면 MSE처럼, 크면 MAE처럼 — 이상치 강건 + 미분 가능 절충
- **손실 함수 선택 체크리스트**: ①예측 타입(회귀/이진/다중/멀티라벨) ②(회귀) 이상치 민감도 ③(분류) 클래스 불균형→`pos_weight`/`weight`/Focal Loss ④활성화함수 중복 방지(모델 vs loss 중 한쪽만) ⑤레이블 dtype/shape 일치(CrossEntropyLoss는 원-핫 아닌 정수 인덱스)
- **Learning Rate**: gradient 방향으로 이동하는 보폭. 너무 크면 발산/진동, 너무 작으면 느림·지역최솟값에 갇힘. 보통 0.1~0.0001에서 시작, 스케줄러로 점감
- **SGD**: 미니배치 단위 gradient로 업데이트. 기본형은 단순 gradient×lr 이동
- **Momentum**: 이전 이동 방향을 일정 비율(보통 0.9) 유지하며 누적 → 진동 감소, 일관된 방향 가속, 지역최솟값 탈출 도움. `SGD(momentum=0.9)`
- **Adam**: 파라미터별 1차 모멘트(방향, Momentum과 동일 개념)·2차 모멘트(변동 크기)를 추적해 learning rate를 자동 적응. 튜닝 부담 적고 수렴 빠름, `Adam(lr=0.001, betas=(0.9,0.999))`. 일반화 성능은 SGD+Momentum이 더 나을 수 있음
- **backward()=역전파**: loss부터 그래프를 거슬러 올라가며 각 텐서의 gradient를 계산하는 알고리즘. 결과는 `.grad`에 누적 저장
- **계산 그래프**: 텐서 연산의 흐름을 노드·엣지로 표현한 DAG. Define-by-Run 방식이라 순전파 시점에 매번 새로 생성됨
- **grad_fn**: 텐서가 "어떤 연산으로 만들어졌는지" 기억하는 포인터. `backward()`가 이걸 따라 거꾸로 이동
- **Chain Rule과 역전파**: 신경망은 층을 쌓은 합성함수라 연쇄법칙으로 각 층의 로컬 미분만 계산해 곱하면 전체 미분을 구할 수 있음
- **분기-합류 gradient**: 한 변수가 여러 경로로 출력에 영향을 주면, 각 경로에서 온 gradient를 모두 더함(다변수 연쇄법칙)
- **requires_grad**: 텐서의 gradient 추적 여부 스위치. True인 텐서가 연산에 섞이면 결과도 자동으로 True(전염성)
- **Leaf/Non-leaf tensor**: leaf=사용자가 직접 만든 텐서(w,b 등), non-leaf=연산 결과 텐서. `.grad`는 기본적으로 leaf에만 저장(non-leaf는 `retain_grad()` 필요)
- **no_grad/detach/clone**: no_grad=블록 전체를 그래프 밖으로, detach=그래프만 끊고 메모리 공유, clone=메모리만 복사하고 그래프 유지, detach().clone()=완전 독립 사본
- **gradient가 None일 때 체크리스트**: forward 경로 포함 여부 → requires_grad → detach/no_grad 개입 여부 → leaf 여부
- **zero_grad/backward/step**: grad 초기화 → grad 계산(값은 아직 안 바뀜) → grad로 파라미터 실제 갱신. 순서 고정
- **loss.item()**: 스칼라 텐서를 순수 float로 추출. 그래프 연결이 끊겨 메모리 누수 방지(리스트에 텐서째로 쌓으면 그래프가 계속 쌓임)
- **train/eval/no_grad 구분**: train↔eval은 Dropout·BatchNorm 등 레이어 동작 모드 제어, no_grad는 그래프 생성 여부 제어 — 평가 코드엔 둘 다 필요
- **Dataset vs DataLoader**: Dataset=`__len__`+`__getitem__`으로 샘플 1개 접근법 정의(데이터 창고), DataLoader=배치화·셔플·병렬 로딩 담당(배달원)
- **TensorDataset vs Custom Dataset**: 이미 텐서로 준비된 데이터는 TensorDataset으로 충분, 파일 읽기·전처리·augmentation이 필요하면 Dataset을 상속해 직접 구현
- **Dataset 디버깅 순서**: `dataset[0]` 타입/shape 확인 → `len()` 확인 → 여러 인덱스 shape 일관성 확인 → `next(iter(loader))` 배치 확인 → 모델 입력 shape 확인 → y dtype이 loss 함수와 맞는지 확인
- **TorchVision transform**: 이미지를 모델 입력에 맞게 전처리하는 함수 묶음(PIL→Tensor, Resize, Normalize, augmentation). `__getitem__` 호출 시점에 지연 적용됨. `Compose`로 순서대로 묶어 사용
- **ToTensor**: `uint8`[0,255] → `float32`[0,1] 스케일링 + `(H,W,C)` → `(C,H,W)` 축 순서 변환(Conv2d가 채널 우선 순서를 기대하기 때문)
- **Normalize(mean, std)**: 채널별로 `(x-mean)/std` 적용해 평균 0 근처로 이동. mean/std는 데이터셋 전체·채널별 픽셀 통계값. ImageNet 사전학습 모델은 ImageNet 통계값을 그대로 써야 함
- **Data Augmentation**: 원본 이미지에 무작위 변형(flip, crop, color jitter 등)을 가해 매 epoch 다르게 보여주는 기법. 과적합 방지용 정규화 기법. train에만 적용, val/test는 결정적 전처리만
- **Batch Size**: 한 학습 스텝에서 한꺼번에 처리하는 샘플 수. 작으면 메모리 적고 노이즈 많음(느림), 크면 메모리 많이 쓰고 안정적(빠름)
- **Shuffle**: 매 epoch마다 데이터 순서를 무작위로 섞음. train은 순서 편향 방지를 위해 항상 True, val/test는 보통 False
- **Train/Valid/Test Split**: train=파라미터 갱신, val=갱신 없이 튜닝·조기종료 판단, test=최종 성능을 딱 한 번 확인. 3분할해야 튜닝 과정에서 test가 오염되지 않음
- **random_split**: `Dataset`을 인덱스 기반으로 무작위 분할해 `Subset`을 반환(메모리 복사 없음). 클래스 비율 미보장, transform 공유 문제 있음 — stratify나 별도 transform Dataset+동일 인덱스로 우회
- **SubsetWithTransform**: 원본 Dataset은 transform 없이 두고, `random_split`은 인덱스만 뽑는 데 쓴 뒤, 그 인덱스+원하는 transform을 직접 소유하는 wrapper 클래스로 감싸는 패턴. "데이터"와 "전처리"를 완전히 분리해 transform 공유 문제를 근본적으로 해결
- **데이터 파이프라인 디버깅 매핑**: NaN loss→Normalize/lr/라벨 이상, val acc 정체→라벨 매칭·데이터 누수·val augmentation 오류, 특정 클래스만 오답→클래스 불균형, shape 에러→Resize 누락·채널 혼재, 재현 안 됨→split 시드 미고정, 배포 후 성능 저하→전처리 불일치
- **randomness 발생 지점**: random_split·weight 초기화·shuffle·augmentation·Dropout·워커 난수·GPU 비결정 연산 — 매 실행 결과가 달라짐
- **재현성(reproducibility)**: 같은 코드 재실행 시 같은 결과가 나오는 성질. 여러 난수 생성기 시드 고정이 출발점
- **seed 고정 3목적**: 재현성, 디버깅(결정적 실행), 공정 비교("다른 조건 동일"). PRNG는 시작 상태 같으면 시퀀스 완전 재생
- **PyTorch 시드 고정**: `random`/`numpy`/`torch.manual_seed`/`cuda.manual_seed_all` + DataLoader는 `generator`·`worker_init_fn`, `random_split`도 generator
- **결정론적 실행(determinism)**: 시드만으로는 재현성(비슷한 결과)까지, GPU 커널 비결정 연산까지 없애 비트 단위 동일하게 = `use_deterministic_algorithms(True)` + `cudnn.deterministic=True`·`benchmark=False` (느려서 최종 실험/디버깅용)
- **seed 주의**: 시드≠완벽 재현(환경 의존), 단일 시드 과신 금지(3~5개 평균±std로 보고), `set_seed()`는 맨 앞 1회, 시드 로그에 기록
- **logging 대상**: config(seed/lr/구조), train·val 지표(항상 쌍), LR·grad norm, 자원, checkpoint 경로. train↔val 간격이 과적합 신호
- **logging 단위**: step(N스텝마다/구간평균) vs epoch(요약) vs best 갱신 시점. 콘솔=요약, 파일/트래커=세부
- **logging sink**: `logging` 모듈(콘솔+파일), metrics.csv/jsonl, TensorBoard/W&B, config 스냅샷, ckpt. run 폴더 하나로 격리(타임스탬프+하이퍼파라미터)
- **logging 실수**: train만 로깅, raw loss만 남김, config·git hash 미기록, `print` 사용, 매 스텝 히스토그램, DDP 전원 로깅, 매 epoch checkpoint
- **state_dict**: "이름→텐서" 매핑. 모델은 파라미터+버퍼(BN running stats), optimizer는 모멘트(exp_avg 등)·param_groups
- **버퍼 저장 중요성**: BN running_mean/var는 gradient 없지만 state_dict에 포함 — 빠뜨리면 추론 시 BN 통계 초기화되어 성능 붕괴
- **모델 저장 방식**: `torch.save(model)`(pickle, 클래스 경로 의존, 깨지기 쉬움) vs `state_dict`만(이식성↑, 권장)
- **checkpoint 구성**: 추론용=model만 / 재개용=model+optimizer+scheduler+scaler+epoch+RNG / 실험관리=+best metric+config+git hash
- **best.pt vs last.pt**: last=매 epoch 덮어씀, full checkpoint, 재개용 / best=val 개선 시만, model만 가볍게, 배포용(마지막 epoch≠최고)
- **checkpoint 문제**: Missing/Unexpected key(구조 불일치→`strict=False`), `module.` 접두사(DDP), BN 저하(버퍼·`eval()`), `map_location`, `weights_only`
- **모델 불러오기 vs 학습 재개**: 전자는 "값"만 복원(best.pt, optimizer 새로 생성, epoch 0) / 후자는 "진행 상태" 전체 복원(last.pt, 모멘텀·스케줄·RNG 유지)
- **optimizer 복원 필수 이유**: Adam 모멘트가 0부터 다시 쌓이면 첫 스텝 업데이트 왜곡 → loss 튐. scheduler `last_epoch`, start_epoch, global_step도 복원
- **추론 시 eval()**: `training=False` 재귀 설정 — Dropout 끔(전체 뉴런), BatchNorm은 running stats 고정 사용. 안 하면 비결정적·배치 의존 출력. autograd와는 무관
- **inference_mode() vs no_grad()**: 둘 다 그래프 생성 차단(메모리·속도) / inference_mode는 version counter·view 추적까지 차단(더 공격적, 결과 텐서 재사용 불가)
- **eval()과 inference_mode()는 직교**: 모듈 동작 모드 vs autograd 동작 — 추론엔 둘 다 필요
- **train/val loss 곡선(x=epoch)**: 겹쳐 그려 일반화 상태·중단 시점·튜닝 방향·버그를 진단. 5절의 learning curve(x=데이터크기)와 다름
- **곡선 3축**: gap(일반화 갭, 크면 과적합), 추세·모양(감소/plateau/U자), val 최저점 위치(마지막이면 더 학습, 중간이면 early stopping)
- **곡선 패턴**: 과적합=train↓ val U자, 과소적합=둘 다 높게 정체, LR 과다=요동/NaN, 계단식 하락=스케줄러 LR 감소(정상)
- **val loss < train loss**: Dropout/측정시점 차이면 정상, val set 편향이면 split 재확인, 갭 크고 지속되면 데이터 누수(버그)
- **곡선 팁**: 같은 축·같은 스케일, 필요시 log scale, accuracy도 같이, 스무딩, val 최저점 마커, 시드 3~5개 밴드
- **Overfitting vs Underfitting**: over=train 좋고 val 나쁨·갭 큼(잡음까지 암기) / under=둘 다 나쁨·갭 작음(패턴 못 배움)
- **Bias–Variance**: bias 높음=underfitting(표현력 부족), variance 높음=overfitting(데이터 변화에 민감). 복잡도↑ → bias↓ variance↑
- **Underfitting 대응**: 모델 용량↑, 더 오래 학습, LR 조정, 규제 완화, 입력 정규화. 먼저 작은 배치에 과적합되는지 확인(안 되면 코드 버그)
- **Overfitting 대응(효과순)**: 데이터·증강 → early stopping → 규제(weight decay·dropout) → 모델 축소 → BN·label smoothing → 앙상블 → 누수 점검
- **3가지 레버**: 모델 복잡도, 학습 시간(epoch, early stopping이 규제처럼 작동), 데이터 양·품질. 목표는 val 성능 최고 지점
- **Dropout**: 학습 중 뉴런을 확률 p로 꺼서 공적응 방지·앙상블 효과. 학습 시 살아남은 값 1/(1-p) 스케일(inverted), 추론 시 끔(eval 필수). 은닉층 0.5, conv엔 잘 안 씀
- **Batch Normalization**: activation을 미니배치 채널별로 평균0·분산1 정규화 후 학습 파라미터로 재조정. 손실 표면을 매끄럽게 해 큰 LR 허용·초기화 둔감·약한 정규화. Linear/Conv→BN→ReLU, 앞 층 bias 생략
- **BN 학습 vs 추론**: 학습=미니배치 통계+running 통계 누적, 추론=running 통계 사용(eval 필수). 작은 배치에서 부정확→LayerNorm/GroupNorm, 시퀀스엔 LayerNorm 표준
- **Dropout vs BN**: Dropout은 정규화 목적·배치 무관, BN은 최적화 안정화 목적·배치 의존 큼. 함께 쓰면 variance shift 충돌(ResNet은 BN만), 순서는 BN→ReLU→Dropout
- **Early Stopping**: val 지표가 개선 안 되면 patience만큼 참다 중단하고 best 가중치 복원. epoch를 규제 대상으로 삼는 사실상 공짜 정규화. L2 weight decay와 유사 효과
- **Early Stopping 하이퍼파라미터**: patience(5~20), min_delta, monitor, mode(min/max), restore_best_weights(True). 반드시 val 기준, LR 스케줄 마지막 구간·double descent 주의
- **Convolution**: 작은 가중치 창을 훑으며 국소 곱-합으로 feature map 생성. 실제론 cross-correlation. 국소 연결+가중치 공유로 파라미터 급감·평행이동 등변성·계층적 특징
- **Kernel vs Filter**: kernel=입력 채널 1개 담당하는 2D 가중치 격자, filter=입력 채널 수만큼 kernel 묶음+bias로 출력 채널 1장 생산. conv 층 weight shape (out_ch, in_ch, kH, kW)
- **3×3 여러 층**: 큰 커널 한 층보다 3×3을 쌓는 게 표준(같은 시야, 파라미터 적음)
- **Padding**: 가장자리에 값(보통 0) 덧대 출력 크기 유지·모서리 정보 손실 방지. 파라미터 무관, valid(없음)/same(크기 보존)
- **Stride**: 창을 몇 칸씩 건너뛸지. 2면 출력 1/2·픽셀 1/4로 다운샘플링, 연산량↓·receptive field↑, 위치 정보 손실
- **CNN 전형 패턴**: conv(K3,P1,S1)로 크기 유지하며 특징 추출 → 단계 전환 지점만 stride2/pooling으로 절반 축소+채널 2배. 해상도↓+채널↑ 반복
- **Pooling**: 창 안 값을 max/average로 요약(학습 가중치 없음). 다운샘플링+평행이동 불변성 강화(등변성과 구별). max pooling은 최댓값 위치로만 gradient
- **Feature map**: filter 하나의 2D 반응 지도. 채널 축="어떤 특징", 공간 축="그 특징이 어디에". 중간 층 채널은 학습된 패턴 검출기, 마지막 conv 층은 Grad-CAM 재료
- **CNN classifier**: backbone(conv/pool 반복, 공간 구조 유지, 픽셀→의미) + 분류 헤드(Flatten/GAP 후 Linear, 사실상 MLP) → 클래스 logit
- **MLP의 이미지 처리 약점**: flatten이 공간 구조 소실·파라미터 폭발(첫 층 약 1.5억)·평행이동 취약(위치마다 따로 학습)·입력 크기 고정을 부름
- **CNN이 해소하는 방식**: 국소 연결(특징은 국소적)+가중치 공유(파라미터 입력크기 무관, 평행이동 등변)+2D 출력(공간 구조 유지)+GAP(입력 크기 유연)
- **귀납적 편향(inductive bias)**: CNN은 "이미지는 국소적, 위치 바뀌어도 같은 물체" 가정을 구조에 내장 → 자연 이미지에서 MLP보다 적은 데이터로 학습

## 08. RNN과 LSTM

> 출처 TIL: 260831

- **RNN이 필요한 이유**: 순서가 의미를 갖는 가변 길이 데이터(문장·음성·시계열). 이전 시점 정보를 hidden state로 기억해 다음 시점에 넘김
- **은닉 상태 갱신**: 매 시점 (새 입력 + 직전 기억) 두 개를 받아 각각 변환 후 더하고 tanh → 갱신된 기억. 필요 시 여기서 출력을 뽑고, 이 기억이 다음 시점 입력으로 되먹임
- **가중치 공유**: 입력변환·기억변환·출력변환 세 규칙을 모든 시점이 재사용 → 길이 무관 처리, 파라미터 수 일정, 위치 무관 패턴 학습(CNN 필터와 같은 발상)
- **BPTT**: 시퀀스를 시간축으로 펼친 뒤 일반 역전파. 길면 메모리 부담 커서 일정 길이로 잘라 학습(truncated BPTT), 경계 넘는 의존성은 학습 불가
- **W_hh (은닉-은닉 가중치)**: 직전 은닉 상태를 이번 기억에 반영하는 규칙. 은닉 차원이 d면 d×d 정사각 행렬. 세 규칙 중 유일하게 시간 방향으로 연쇄해서 곱해짐
- **기울기 소실/폭발**: 먼 과거 gradient는 W_hh를 시점 수만큼 거듭제곱하는 꼴 → 영향력 1 미만이면 지수적 소실(장기 의존성 실패), 1 초과면 폭발(clipping으로 완화). 바닐라 RNN은 10~20 스텝부터 기억 상실
- **RNN 개선 구조**: LSTM(cell state+게이트 3개), GRU(게이트 2개, cell state 없음), 양방향(전체 문장 주어질 때만), 다층(은닉 상태를 다음 층 입력으로)
- **LSTM 해법**: 정보 고속도로(cell state) + 무엇을 지우고 더하고 내보낼지 조절하는 학습된 스위치(게이트)
- **두 개의 상태**: cell state=장기 기억, 곱셈·덧셈만 거쳐 흐르는 컨베이어 벨트(gradient 경로). hidden state=단기 기억이자 출력, cell state를 필터링한 것
- **forget gate**: 기존 cell state 중 버릴 것 선택(0=삭제, 1=유지)
- **input gate**: 새 후보 정보 중 저장할 것 선택. cell state 갱신 = (기존 기억 × forget) + (새 후보 × input), 삭제와 추가가 분리됨
- **output gate**: 갱신된 cell state 중 이번 시점 hidden state로 내보낼 부분 선택
- **LSTM에서 gradient가 사는 이유**: cell state 갱신이 덧셈 중심. forget≈1이면 이전 cell state가 손실 없이 전달되고 역전파도 감쇠 없이 흐름. forget bias를 1로 초기화하는 요령
- **RNN vs LSTM**: 전달 상태 1개 vs 2개, 곱셈 연쇄 vs 덧셈 경로, 장기 의존성 수십 스텝 vs 수백 스텝, 게이트 0개 vs 3개, 파라미터 약 4배
- **GRU**: 게이트 2개·cell state 없음, 파라미터 적고 빠름, 성능 비슷. 데이터 적으면 GRU, 큰 데이터·긴 의존성이면 LSTM
- **한계: 순차 계산**: hidden state가 직전 것에 의존 → 시간축 병렬화 불가, 학습 속도가 길이에 비례. 대규모 사전학습 시대에 밀려난 결정적 이유
- **한계: 고정 크기 은닉 상태 병목**: 읽은 내용 전부를 벡터 하나에 압축 → 긴 문장에서 오래된 정보 소실. seq2seq의 이 병목을 뚫으려 어텐션 등장 → RNN 없이 어텐션만 = Transformer
- **한계: 먼 거리 경로**: 토큰 1↔100은 99개 중간 스텝 통과하며 희석. Transformer는 어텐션으로 경로 길이 1
- **한계: 단방향성**: 기본 RNN은 좌→우만. BiLSTM은 전체 시퀀스 필요(실시간 불가), 두 단방향의 concat에 가까움
- **한계: 학습**: 기울기 문제 완화됐을 뿐 잔존(수백 스텝), 폭발은 여전해 clipping 필수, BPTT 메모리, 노출 편향(teacher forcing↔자기 출력), 계층·중첩 구조 표현 취약
- **여전히 쓰이는 곳**: 저지연 스트리밍(음성 인식·실시간 번역), 임베디드·모바일, 짧은 시계열, 온라인 학습. S4·Mamba 등 state space model이 발상 계승

## 09. LLM 추론과 서빙

> 출처 TIL: 260901

- **FreeToken**: 개인 PC(8GB 노트북 GPU~워크스테이션 1장)를 GPU·CPU·호스트메모리·인터커넥트 묶은 탄력적 추론 플랫폼으로 보고, 프론티어급 대형 MoE를 실사용 속도로 서빙하는 엔진
- **MoE의 양면성**: 토큰당 일부 expert만 활성이라 연산량은 소비자 GPU로 감당 가능하지만, 전체 expert 풀은 VRAM 초과라 호스트 메모리/디스크에 두고 on-demand 로드
- **기존 엣지 엔진 3대 한계**: prefill이 희소성 파괴(전체 풀 스트리밍, 재-prefill 잦음) / decode는 희소하나 캐시 미스(라우팅이 토큰마다 바뀜, CPU 대역폭 부족) / 엣지 자원 변동성(VRAM 공유, 잦은 재시작)
- **대역폭 적응 실행**: 대역폭을 병목이 아닌 스케줄링 신호로. prefill은 레이어 전체 이중 버퍼링으로 전송을 연산 뒤에 숨김
- **q\* 정책**: decode 미스를 "PCIe로 캐시 채우기"와 "CPU 즉석 실행"으로 분할. 채울 비율 = 미스 수 × (PCIe 대역폭 / 호스트 대역폭), 두 대역폭은 기기에서 실측. 부분합은 정확히 병합(근사 없음)
- **의미 기반 캐싱**: special-token 경계(thinking·tool call·턴)에 순환 상태 체크포인트 앵커링 → 편집 후 새 suffix만 재-prefill. full-attention KV는 radix prefix tree, 순환 레이어는 상태 체크포인트로 재사용
- **LRU expert 캐시**: 인접 토큰 간 expert 겹침(시간적 지역성)을 전 레이어 공유 캐시로 흡수, 잔여 미스만 q\*로
- **탄력적 메모리 관리**: CPU 상주 expert 풀이 항상 정답 원본 → GPU 메모리는 성능에만 영향. 실행 중 캐시 재빌드·KV↔expert 분배 조정, 재시작 없음
- **빠른 부팅**: expert를 최종 레이아웃 그대로 디스크에서 읽어 pin, GPU 워밍업 생략(첫 요청은 콜드 캐시)
- **주요 결과**: decode 1.3~2.3× / 에이전트형일수록 안정적(단일 턴 대비 12% 이내 하락) / tail TTFT 44초 미만 vs 베이스라인 150초+ / 8GB 노트북에서 350억 모델 39.3 tok/s
- **차별점**: 라우팅 연산은 정확히 유지한 채 "잔여 미스를 전송할지 CPU 실행할지"를 바꿈. 실측 대역폭에서 닫힌 형태 비율로 도출해 CUDA Graph 내부 device-resident로 유지

## 10. 하네스 엔지니어링

> 출처 TIL: 260901

- **하네스**: 모델 호출을 감싸는 모든 것 — 시스템 프롬프트·에이전트 루프·도구 정의·컨텍스트 관리·권한 게이트·에러 처리·서브에이전트. 그 한가운데 LLM 가중치
- **하네스 엔지니어링**: 모델은 그대로 두고 주변부를 바꿔 에이전트 성능을 올리는 작업. 같은 모델을 쓰는 Claude Code·Cursor·Devin의 결과 차이는 대부분 여기서 옴
- **왜 하는가**: 모델 재학습은 비싸고 느림. 하네스 개선은 학습 없이 분 단위 반복·즉시 배포 → 투자 대비 효과가 큼
- **컨텍스트 엔지니어링**: 제한된 윈도우에 무엇을·어떤 순서로·얼마나 넣을지 결정. 길어지면 요약·압축, 필요한 파일만 검색 주입
- **도구 설계**: 개수·이름·설명문·스키마 다듬기. 설명 한 줄로 호출 정확도가 크게 달라짐. 너무 많으면 묶거나 지연 로딩(deferred tools)
- **에이전트 루프**: 출력 파싱·도구 실행·결과 포맷팅·종료 조건 판단. 언제 멈출지, 실패를 어떻게 알릴지가 핵심
- **검증·피드백 루프**: 테스트 실패 로그를 다시 모델에 넣어 스스로 고칠 재료 제공
- **서브에이전트 구조**: 큰 작업을 쪼개 별도 컨텍스트에 위임하고 결과만 회수
- **안전·권한 게이트**: 되돌리기 어려운 행동(파일 삭제·외부 전송) 앞에 확인 단계
- **범위 관계**: 프롬프트(보내는 텍스트) ⊂ 컨텍스트(그 텍스트에 무엇을 담을지) ⊂ 하네스(모델을 굴리는 시스템 전체)
