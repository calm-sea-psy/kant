# 딥러닝 기초와 PyTorch

> 출처 TIL: 260819, 260820

## 1. 머신러닝과 딥러닝의 접근 차이

머신러닝은 사람이 특징(feature)을 직접 설계해서 모델에 넣어주는 방식입니다. 반면 딥러닝은 원본 데이터를 그대로 모델에 넣으면, 모델이 데이터 안에서 유용한 특징 표현을 스스로 학습합니다. 특징 설계를 사람이 하느냐 모델이 하느냐가 두 접근의 핵심적인 차이입니다.

## 2. 딥러닝 학습 파이프라인의 기본 구조

딥러닝 학습은 대체로 "데이터 → 모델 → 손실 → 최적화 → 평가"라는 흐름을 따르며, 실제 코드는 보통 다음 순서로 구성됩니다.

1. **import**: 필요한 라이브러리를 불러옵니다.
2. **config / hyperparameter**: batch size, learning rate, epoch 수 같은 값을 미리 정의합니다.
3. **data / dataloader**: 입력(X)과 정답(y)을 만들고, 이를 배치 단위로 꺼내 쓸 수 있도록 DataLoader를 구성합니다.
4. **model**: 입력을 예측값으로 바꾸는 신경망을 정의합니다.
5. **loss function**: 예측값과 정답의 차이를 계산하는 함수를 정의합니다.
6. **optimizer**: 손실을 줄이는 방향으로 파라미터를 업데이트하는 방법을 정합니다.
7. **train loop**: 학습 데이터로 파라미터를 실제로 업데이트합니다. 매 스텝은 다음과 같은 정해진 순서를 따릅니다.
   ```python
   model.train()                        # 학습 모드로 전환 (Dropout·BatchNorm이 학습 모드로 동작)
   optimizer.zero_grad()                # 이전 batch에서 계산된 gradient 초기화
   preds = model(batch_x)               # 현재 batch에 대한 예측값 계산 (forward)
   loss = loss_fn(preds, batch_y)       # 예측값과 정답의 차이 계산
   loss.backward()                      # loss 기준으로 gradient 계산 (backward)
   optimizer.step()                     # gradient를 사용해 parameter 업데이트
   ```
   `zero_grad()`가 필요한 이유는 PyTorch가 기본적으로 gradient를 누적(accumulate)하기 때문입니다 — 지워주지 않으면 이전 batch의 gradient가 계속 더해집니다. 이후 forward로 예측값을 구하고, loss를 계산한 뒤, `backward()`로 각 파라미터의 `.grad`에 gradient를 채우고, `step()`으로 그 gradient를 이용해 파라미터를 갱신하는 순서가 고정되어 있습니다.
8. **validation loop**: 검증 데이터로 현재 모델의 성능을 확인합니다. gradient 계산이 필요 없으므로 `model.eval()`과 `torch.no_grad()`를 함께 씁니다.
9. **logging**: epoch마다의 loss·metric을 기록해 학습 추이를 추적합니다.
10. **checkpoint**: 학습된 모델을 파일로 저장해 나중에 재사용할 수 있게 합니다.

## 3. Tensor 생성과 dtype / shape 확인

딥러닝 모델은 결국 숫자만 입력받고 숫자만 출력합니다. 이미지도, 텍스트도, 정답 라벨도 전부 숫자로 바꿔서 모델에 넣어야 하는데, PyTorch에서 이 숫자 데이터를 담는 기본 그릇이 **Tensor**입니다.

**shape**: Tensor의 구조를 왼쪽에서 오른쪽 차원 순서로 설명합니다.

| shape | 읽는 방법 | 예시 |
|---|---|---|
| `torch.Size([])` | 값 하나 | scalar |
| `torch.Size([3])` | 값 3개 | vector |
| `torch.Size([2, 3])` | 2행 3열 | matrix |
| `torch.Size([10, 4])` | 샘플 10개, 특성 4개 | tabular batch |
| `torch.Size([32, 3, 224, 224])` | 이미지 32장, 채널 3개, 높이 224, 너비 224 | image batch |

이 외에 Tensor를 다룰 때 확인해야 할 기본 속성이 세 가지 더 있습니다.

- **ndim**: 텐서의 차원 수입니다. shape에 적힌 숫자의 개수와 같습니다.
- **dtype**: 텐서에 담긴 값의 자료형입니다(`torch.float32`, `torch.int64` 등). 정수 값이라도 모델 입력·손실 계산 등 gradient가 필요한 연산에 쓰이려면 float 계열이어야 하며, `mean()`처럼 float만 지원하는 연산도 있으므로 목적에 맞는 dtype을 선택해야 합니다.
- **device**: 텐서가 실제로 저장되어 있는 연산 장치입니다(`cpu`, `cuda:0` 등). 모델과 입력 텐서는 같은 device에 있어야 연산이 가능합니다.

## 4. Batch dimension

Batch dimension은 여러 개의 샘플(데이터)을 한 번에 묶어서 처리하기 위해 텐서의 맨 앞(보통 0번째)에 두는 차원입니다. "샘플이 몇 개인가"를 나타내는 축입니다.

딥러닝에서 샘플을 한 개씩 처리하면 GPU 연산 효율이 크게 떨어집니다. GPU는 행렬 연산을 병렬로 처리하는 데 특화되어 있어서, 샘플을 여러 개 쌓아 한 번의 행렬곱으로 처리하는 쪽이 훨씬 빠릅니다. 그래서 거의 모든 딥러닝 프레임워크는 "항상 batch 차원이 있다"고 가정하고 설계되어 있으며, `nn.Linear` 같은 레이어를 정의할 때도 batch 차원은 신경 쓸 필요 없이 feature 차원만 지정하면 됩니다(배치 크기가 얼마든 동일한 가중치로 각 샘플에 독립적으로 적용됨).

데이터 종류에 따라 batch 차원이 놓이는 위치와 나머지 차원의 의미가 관례적으로 정해져 있습니다.

| 데이터 종류 | 일반적인 shape | 의미 |
|---|---|---|
| 표(정형) 데이터 | `(batch, features)` | MLP/선형모델 입력 |
| 이미지 | `(batch, channels, height, width)` | CNN 입력 |
| 시퀀스(RNN 기본) | `(seq_len, batch, features)` | `batch_first=False` 기본값 |
| 시퀀스(Transformer 등) | `(batch, seq_len, features)` | `batch_first=True` |

RNN/Transformer 계열은 `batch_first` 옵션으로 batch 차원의 위치가 달라질 수 있으므로, 사용하는 라이브러리의 기본값을 확인해야 합니다.

샘플이 하나뿐인 입력을 모델에 넣어야 할 때도 batch 차원이 있어야 하므로, 5절의 `unsqueeze`로 크기 1짜리 batch 차원을 인위적으로 만들어 넣습니다.

## 5. squeeze / unsqueeze

`squeeze`와 `unsqueeze`는 텐서 shape에서 크기가 1인 차원을 없애거나(squeeze) 새로 끼워 넣는(unsqueeze) 함수입니다. 값 자체는 전혀 바뀌지 않고 차원(축) 구조만 바뀝니다.

**unsqueeze: 크기 1인 차원을 추가**

```python
x = torch.tensor([1, 2, 3])       # shape: (3,)
x.unsqueeze(0)                     # shape: (1, 3)  -> tensor([[1, 2, 3]])
x.unsqueeze(1)                     # shape: (3, 1)  -> tensor([[1], [2], [3]])
```
인자는 "몇 번째 위치에 새 차원을 끼워넣을지"를 뜻합니다. `0`이면 맨 앞, `1`이면 두 번째 위치에 크기 1인 차원이 생깁니다.

**squeeze: 크기 1인 차원을 제거**

```python
y = torch.zeros(1, 3, 1)          # shape: (1, 3, 1)
y.squeeze()                        # shape: (3,)     -> 크기 1인 차원 전부 제거
y.squeeze(0)                       # shape: (3, 1)    -> 0번 차원만 제거
y.squeeze(2)                       # shape: (1, 3)    -> 2번 차원만 제거
```
인자 없이 쓰면 크기가 1인 차원을 전부 없애고, 인자를 주면 그 위치의 차원만 없앱니다(그 위치가 1이 아니면 아무 일도 일어나지 않습니다).

**왜 필요한가**

`nn.Linear(16, 1)` 같은 레이어의 출력은 `(batch, 1)`처럼 항상 마지막에 크기 1인 차원이 남습니다. 이를 스칼라처럼 다루고 싶을 때는 `preds.squeeze(-1)`로 `(batch, 1)`을 `(batch,)`로 정리합니다. 반대로 1차원 벡터를 모델에 넣거나 다른 텐서와 연산하려면 batch 차원을 새로 끼워 넣어야 하는 경우가 많습니다.

```python
x_1d = torch.tensor([1.0, 2.0, 3.0])   # (3,)
x_1d.unsqueeze(0)                       # (1, 3)  배치 차원 추가 — "샘플 1개짜리 배치"로 취급
```

**shape 불일치로 인한 잘못된 브로드캐스팅**

```python
pred = torch.tensor([1.0, 2.0, 3.0])         # shape: (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])  # shape: (3, 1)
pred - target   # shape: (3, 3)  !! 의도치 않은 브로드캐스팅
```
`(3,)`와 `(3,1)`을 그냥 빼면 6절의 브로드캐스팅 규칙 때문에 `(3,3)` 행렬이 나와버리는 흔한 버그가 발생합니다. `pred.unsqueeze(1)`로 shape을 `(3,1)`로 맞춰줘야 의도한 대로 원소별 결과 `(3,1)`이 나옵니다. `nn.MSELoss()`에 입력·타깃 shape이 안 맞을 때 경고가 뜨는 것도 같은 이유입니다.

정리하면, **squeeze는 불필요한 1차원을 제거**(예: 모델 출력 후 정리), **unsqueeze는 필요한 1차원을 추가**(예: 배치 차원 맞추기, 브로드캐스팅 shape 맞추기)하는 함수입니다.

## 6. Broadcasting

브로드캐스팅은 shape이 다른 두 텐서를 실제로 복사해서 크기를 맞추지 않고도 연산할 수 있게, 규칙에 따라 자동으로 shape을 맞춰주는 기능입니다. NumPy와 PyTorch가 동일한 규칙을 씁니다.

**규칙(뒤에서부터 차원을 비교)**: 두 텐서의 shape을 오른쪽(마지막 차원)부터 왼쪽으로 하나씩 비교하면서, 각 차원마다 다음 조건 중 하나를 만족해야 합니다.

- 두 차원의 크기가 같다
- 둘 중 하나가 1이다(그러면 1인 쪽이 다른 쪽 크기만큼 복제된 것처럼 취급됨)
- 둘 중 하나가 아예 존재하지 않는다(차원 개수가 다른 경우, 짧은 쪽 앞에 1을 채운 것으로 간주)

이 세 조건 중 어느 것도 만족하지 못하는 차원이 있으면 에러가 납니다.

**예시 1: 스칼라 연산**
```python
x = torch.tensor([1, 2, 3])   # (3,)
x + 10                         # (3,) + () -> (3,)  각 원소에 10을 더함
```

**예시 2: 벡터 + 행렬**
```python
a = torch.ones(3, 4)                 # (3, 4)
b = torch.tensor([1., 2., 3., 4.])   # (4,)
a + b   # (3, 4) + (4,)
        # b 앞에 1을 채워서 (1, 4)로 간주 -> 3번 복제되어 (3, 4)와 연산
```
결과는 `a`의 각 행마다 `b`를 더한 `(3, 4)` 텐서입니다.

**예시 3: 흔히 발생하는 실수 케이스**
```python
pred = torch.tensor([1.0, 2.0, 3.0])          # (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])   # (3, 1)
pred - target
```
뒤에서부터 비교하면, 마지막 차원은 `3` vs `1`이라 1인 쪽이 3으로 확장되고, 그다음 차원은 없음(pred는 1차원) vs `3`이라 없는 쪽이 1로 채워진 뒤 다시 3으로 확장됩니다. 결과 shape은 `(3, 3)`. `pred`가 `(1, 3)`처럼, `target`이 `(3, 1)`처럼 확장되어 사실상 모든 pred와 모든 target의 조합을 계산해버리는, 원소별 뺄셈이 아닌 외적에 가까운 결과가 나옵니다. 5절에서 다룬 것처럼 `pred.unsqueeze(1)`로 shape을 `(3, 1)`로 맞춰야 의도한 연산이 됩니다.

**예시 4: 양쪽이 동시에 확장되는 경우**
```python
a = torch.ones(3, 1)   # (3, 1)
b = torch.ones(1, 4)   # (1, 4)
(a + b).shape           # (3, 4)
```
양쪽 모두 크기 1인 차원이 있으면 서로 다른 방향으로 동시에 확장되어 `(3, 4)`가 됩니다.

**요약**: 비교는 오른쪽(마지막 차원)부터 시작하며, 크기가 같거나 둘 중 하나가 1이거나 아예 없으면 통과, 그 외에는 `RuntimeError: size mismatch`가 납니다. 딥러닝에서 흔한 실수는 `(N,)`과 `(N, 1)`을 헷갈려 의도치 않게 `(N, N)`으로 확장되는 것이므로, loss 계산 전에 관련 텐서들의 `.shape`을 항상 확인하는 습관이 중요합니다.

## 7. 손실함수: MSELoss / BCEWithLogitsLoss / CrossEntropyLoss

세 손실 함수는 문제 유형(회귀 vs 이진분류 vs 다중분류)에 따라 골라 쓰는 것입니다.

**`nn.MSELoss()` — 회귀(Regression)**: 예측값과 정답의 차이를 제곱해서 평균낸 값입니다(Mean Squared Error). 정답이 연속적인 숫자(집값, 온도, 점수 예측 등)일 때 씁니다. 제곱을 쓰기 때문에 큰 오차에 더 강하게 벌점을 주어 이상치에 민감하며, 출력값에 별도 제약이 없어 모델 마지막 레이어가 그냥 `nn.Linear`로 끝나도 됩니다.

**`nn.BCEWithLogitsLoss()` — 이진분류(Binary Classification)**: "둘 중 하나"(예: 스팸/정상, 사기/정상)를 분류할 때 씁니다. Binary Cross Entropy(BCE)에 Sigmoid를 내부적으로 합쳐놓은 버전입니다. 모델의 마지막 레이어는 확률로 변환하기 전의 raw 점수(logit)를 그대로 출력해야 하며, 마지막에 `nn.Sigmoid()`를 직접 붙이면 안 됩니다. "Sigmoid + BCE"를 따로 하지 않고 하나로 합친 이유는 수치적으로 더 안정적이기 때문입니다 — 두 연산을 결합해서 계산하면 극단적인 logit 값에서 발생하는 오버플로/언더플로 문제를 피할 수 있습니다. `nn.BCELoss()`도 있지만, 그건 모델이 이미 확률(0~1, sigmoid를 직접 적용한 값)을 출력했을 때만 쓰는 것이고, 요즘은 `BCEWithLogitsLoss`를 쓰는 게 표준입니다.

**`nn.CrossEntropyLoss()` — 다중분류(Multi-class Classification)**: "셋 이상 중 하나"(예: 고양이/개/새, 숫자 0~9 분류)를 분류할 때 씁니다. Softmax + Negative Log Likelihood를 내부적으로 합쳐놓은 버전입니다. 모델의 마지막 레이어는 클래스 개수만큼의 raw 점수(logit)를 출력해야 하며, 여기도 마지막에 `nn.Softmax()`를 직접 붙이면 안 됩니다(내부에서 이미 처리). `targets`가 BCE처럼 0/1 float이 아니라 정수 클래스 인덱스(0, 1, 2, ...)라는 점이 shape/dtype 실수가 가장 잦은 부분입니다. BCEWithLogitsLoss와 마찬가지로 Softmax를 따로 분리하지 않는 이유는 수치 안정성(`log_softmax`를 내부적으로 결합 계산) 때문입니다.

BCEWithLogitsLoss와 CrossEntropyLoss는 둘 다 "활성화함수(sigmoid/softmax)를 loss 안에 숨겨놓았다"는 점이 같습니다. 그래서 모델은 항상 raw logit을 출력하게 설계하고, 확률이 실제로 필요한 추론 시점에만 별도로 sigmoid/softmax를 적용합니다.

## 8. device: 텐서와 모델의 연산 장치

`device`는 텐서(또는 모델)가 실제로 어느 하드웨어에 올라가서 연산되는지를 나타내는 속성입니다. PyTorch에서 가장 흔한 값은 `cpu`와 `cuda`(NVIDIA GPU) 두 가지입니다. 텐서는 만들어질 때 기본적으로 CPU 메모리에 올라가고, GPU에서 연산하려면 명시적으로 옮겨줘야 합니다.

GPU는 4절에서 다룬 batch 연산(대량의 행렬곱)을 CPU보다 훨씬 빠르게 병렬 처리할 수 있어서, 딥러닝에서는 가능하면 모델과 데이터를 GPU로 옮겨서 학습·추론합니다. 다만 CPU 메모리와 GPU 메모리는 물리적으로 분리되어 있어서, 같은 연산에 참여하는 텐서들은 반드시 같은 device에 있어야 합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)          # 모델 파라미터를 device로 이동
batch_x = batch_x.to(device)      # 입력 텐서도 같은 device로 이동
batch_y = batch_y.to(device)
```
`torch.cuda.is_available()`로 GPU 유무를 확인해서 있으면 `cuda`, 없으면 `cpu`로 자동 분기하는 게 표준 패턴이며, 같은 코드가 GPU 있는 환경/없는 환경 어디서든 그대로 동작합니다.

자주 쓰는 관련 메서드는 다음과 같습니다.

- `tensor.to(device)`: 지정한 device로 복사본을 반환(원본 유지, 새 텐서 할당)
- `tensor.cuda()` / `tensor.cpu()`: `.to("cuda")` / `.to("cpu")`의 축약형(구식 스타일, 요즘은 `.to(device)`가 더 권장됨)
- `model.to(device)`: 모델 안의 모든 파라미터·버퍼를 한 번에 이동

## 9. 퍼셉트론과 선형 결정 경계

퍼셉트론(Perceptron)은 신경망의 가장 기본 단위를 모델링한, 인공 뉴런의 원형입니다. 1958년 프랑크 로젠블랫이 제안한 개념으로, 지금 쓰는 `nn.Linear` + 활성화함수 구조의 뿌리입니다.

퍼셉트론은 여러 개의 입력값을 받아서, 각 입력에 가중치(weight)를 곱해 다 더하고 편향(bias)을 더한 뒤, 그 결과가 특정 기준(보통 0)을 넘으면 1, 안 넘으면 0(또는 -1)을 출력하는 단순한 구조입니다. "입력들을 가중합해서, 임계값을 넘으면 활성화(발화)하고 안 넘으면 안 한다"는 것이 핵심이며, 실제 뉴런이 여러 신호를 받아 일정 자극 이상이면 전기 신호를 발화하는 방식을 단순화해서 흉내 낸 것입니다.

`nn.Linear(4, 1)`이 사실 퍼셉트론의 "가중합 + 편향" 부분과 정확히 같은 연산입니다. 원조 퍼셉트론은 여기에 계단함수(threshold 넘으면 1, 아니면 0)를 씌워 최종 출력을 만들었는데, 계단함수는 미분이 불가능(또는 미분값이 거의 항상 0)해서 gradient 기반 학습(`loss.backward()`)을 적용할 수 없습니다. 그래서 이후 Sigmoid, ReLU처럼 부드럽고 미분 가능한 함수로 대체된 것이 오늘날 딥러닝 학습을 가능하게 만든 핵심 전환점입니다.

**한계 — XOR 문제**: 단일 퍼셉트론(단층)은 선형으로 분리 가능한 문제만 풀 수 있습니다. 즉 하나의 직선(또는 초평면)으로 두 클래스를 나눌 수 있는 문제만 학습 가능합니다. XOR처럼 하나의 직선으로 절대 나눌 수 없는 문제는 단일 퍼셉트론으로는 원리적으로 풀 수 없다는 것이 1969년에 증명되면서, 한동안 신경망 연구가 침체되는 계기가 됐습니다(1차 AI 겨울의 원인 중 하나). 이 한계는 퍼셉트론을 여러 층으로 쌓는 방식(10절의 MLP)으로 극복됩니다.

## 10. MLP (Multi-Layer Perceptron)

MLP는 여러 개의 완전연결층(Linear layer)을 활성화함수로 이어붙인, 가장 기본적인 신경망 구조입니다. 지금까지 다뤄온 모든 모델 예제가 사실상 MLP였습니다.

```python
nn.Sequential(
    nn.Linear(4, 16),   # 입력층 -> 은닉층 (16개 뉴런)
    nn.ReLU(),          # 비선형 활성화
    nn.Linear(16, 1),   # 은닉층 -> 출력층
)
```

- **입력층**: 데이터가 들어오는 자리(층이라기보다 입력 벡터 자체)
- **은닉층(hidden layer)**: 입력층과 출력층 사이의 층. 개수와 각 층의 뉴런 수 모두 자유롭게 정할 수 있는 하이퍼파라미터
- **출력층**: 최종 예측값을 내는 층

각 층은 "이전 층의 출력 전체 → 다음 층의 모든 뉴런"으로 완전히 연결되어 있어 완전연결층(Fully Connected Layer)이라고도 부르며, `nn.Linear`가 정확히 이 연산을 수행합니다.

**왜 층을 쌓아야 하는가**: 9절에서 설명한 대로 단일 퍼셉트론(층 하나)은 선형 분리 가능한 문제만 풉니다. MLP는 층과 층 사이에 비선형 활성화함수(ReLU, Sigmoid, Tanh 등)를 끼워 넣는데, 이게 핵심입니다. 활성화함수 없이 Linear만 계속 쌓으면 행렬곱을 아무리 이어붙여도 결국 하나의 선형 변환으로 합쳐지기 때문에 층을 쌓는 의미가 없어집니다. 층 사이에 비선형 함수를 넣어야만 "선형 변환 → 비선형 변환 → 선형 변환 → ..."이 반복되면서 곡선·곡면 같은 복잡한 형태의 결정경계나 함수를 표현할 수 있습니다.

**보편 근사 정리(Universal Approximation Theorem)**: 은닉층이 하나만 있어도(폭이 충분히 넓다면) MLP는 이론적으로 임의의 연속함수를 원하는 정밀도로 근사할 수 있다는 것이 수학적으로 증명되어 있습니다. 다만 "이론적으로 가능"과 "실제로 학습이 잘 된다"는 다른 문제라서, 실무에서는 넓고 얕은 층 하나보다 좁고 깊은 여러 층을 쌓는 쪽이 더 효율적으로 학습되는 경우가 많습니다.

**한계와 이후 발전**: MLP는 입력의 순서나 공간적 구조(이미지의 픽셀 배치, 문장의 단어 순서)를 전혀 고려하지 않고 모든 입력을 평평한 벡터로 취급합니다. 그래서 이미지에는 CNN(합성곱), 시퀀스에는 RNN/Transformer처럼 구조에 맞는 특화된 아키텍처가 따로 발전했지만, 이들 내부에도 결국 MLP(완전연결층)가 부분적으로 들어가는 경우가 대부분입니다.

## 11. nn.Linear

`nn.Linear`는 가중합 + 편향 연산(완전연결층)을 수행하는 기본 레이어로, 9절의 퍼셉트론 연산을 병렬로 여러 개 쌓아놓은 것과 같습니다.

```python
layer = nn.Linear(in_features=4, out_features=16)
```
입력 벡터의 각 원소에 가중치를 곱해서 다 더하고 편향을 더하는 연산을, 출력 차원(16개) 수만큼 각각 독립적으로 반복합니다.

**내부 파라미터**
```python
layer.weight   # shape: (out_features, in_features) = (16, 4)
layer.bias     # shape: (out_features,) = (16,)
```
`weight`는 출력마다 입력 각각에 곱해질 계수들(학습되는 가중치 행렬), `bias`는 출력마다 하나씩 있는 편향 벡터입니다. 이 둘이 `model.parameters()`로 옵티마이저에 전달되는 실제 학습 대상이고, `loss.backward()` → `optimizer.step()`으로 값이 갱신되는 대상입니다.

**batch dimension과의 관계**: `nn.Linear`는 입력의 마지막 차원만 `in_features`와 맞으면 되고, 그 앞의 모든 차원(batch 등)은 그대로 유지됩니다.
```python
x = torch.randn(32, 4)     # (batch=32, in_features=4)
layer = nn.Linear(4, 16)
layer(x).shape               # (32, 16)  -- batch는 그대로, feature만 4->16
```
그래서 배치 크기가 몇이든 같은 레이어 정의 하나로 처리할 수 있습니다.

**"Linear"라는 이름의 의미**: 이 연산 자체는 선형 변환(곱셈과 덧셈만 있고 꺾임이나 곡률이 없음)입니다. 그래서 `nn.Linear`만 여러 개 이어붙이면 결국 하나의 `nn.Linear`와 수학적으로 동일해져 버리므로, 표현력을 가지려면 반드시 사이사이에 `nn.ReLU()` 같은 비선형 활성화함수가 있어야 합니다. `nn.Linear` 자체는 "각 뉴런이 입력을 어떻게 가중합하는가"만 담당하는 조각입니다.

## 12. flatten: 다차원 텐서를 벡터로 펼치기

`flatten`은 다차원 텐서를 더 낮은 차원(보통 1차원)으로 펼치는 연산입니다. 이미지처럼 여러 차원을 가진 데이터를 `nn.Linear`(입력이 1차원 벡터여야 함)에 넣기 전에 형태를 맞춰주는 용도로 가장 많이 씁니다.

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape: (2, 3)
torch.flatten(x)                 # shape: (6,)  -> tensor([1, 2, 3, 4, 5, 6])
```

**핵심: batch dimension은 보통 남겨야 함**. 이미지 배치처럼 batch 차원이 있는 텐서에 `torch.flatten`을 그냥 쓰면 batch 차원까지 뭉개져서 서로 다른 샘플이 하나로 섞여버립니다.
```python
imgs = torch.randn(32, 3, 28, 28)   # (batch=32, channel=3, h=28, w=28)
torch.flatten(imgs)                  # shape: (75264,)  -- 배치 32개가 전부 한 줄로 뭉개짐! 잘못됨
```
그래서 실제로는 batch 차원(0번째)은 그대로 두고 그 뒤의 차원들만 펼치도록 `start_dim`을 지정합니다.
```python
torch.flatten(imgs, start_dim=1)     # shape: (32, 2352)  -- batch는 유지, 나머지만 3*28*28=2352로 펼침
```
즉 각 샘플이 "batch, 나머지 전부 펼친 벡터" 형태가 되어 `nn.Linear(2352, ...)`에 그대로 넣을 수 있습니다.

`nn.Flatten()`은 모델 안에서 레이어처럼 쓰는 버전으로, `start_dim` 기본값이 이미 `1`이라서 `Sequential` 안에 넣기만 하면 batch 차원을 안전하게 보존하면서 나머지를 펼쳐줍니다. CNN(합성곱층) 뒤에 완전연결층을 붙일 때 거의 항상 등장하는 패턴입니다.
```python
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),   # (batch, 3, H, W) -> (batch, 16, H', W')
    nn.ReLU(),
    nn.Flatten(),           # (batch, 16, H', W') -> (batch, 16*H'*W')
    nn.Linear(16*H_*W_, 10) # 완전연결층은 벡터 입력만 받으므로 반드시 필요
)
```

5절에서 다룬 squeeze/unsqueeze는 크기가 1인 차원만 없애거나 추가하는 반면, flatten은 크기와 상관없이 여러 차원을 하나로 합쳐버립니다. squeeze/unsqueeze는 "불필요한 자리 차원 정리", flatten은 "다차원 구조를 벡터로 변환"이 목적이라는 점에서 서로 다릅니다.

## 13. view, reshape, flatten 비교

세 함수 모두 텐서의 shape을 바꾸는 데 쓰이지만, "메모리를 공유하느냐, 언제 쓸 수 있느냐, 얼마나 유연하냐"에서 차이가 있습니다.

**`view`**: 새 텐서를 만드는 게 아니라 같은 메모리를 다른 shape으로 읽는 것이라, 항상 메모리를 공유합니다.
```python
x = torch.arange(6)          # tensor([0,1,2,3,4,5])
y = x.view(2, 3)              # shape 변경, 메모리는 x와 100% 공유
y[0, 0] = 99
x                             # tensor([99,1,2,3,4,5])  -- x도 바뀜!
```
단, 원본이 메모리상 연속(contiguous)이어야만 가능합니다. `transpose`, `permute` 등을 거친 텐서는 메모리 배치와 논리적 shape이 어긋나 있어(non-contiguous), `view`를 바로 쓰면 에러가 납니다.
```python
a = torch.randn(3, 4)
b = a.t()
b.view(4, 3)                  # RuntimeError!
```

**`reshape`**: 가능하면 `view`처럼 메모리를 공유하지만, 불가능한 경우(non-contiguous)에는 자동으로 데이터를 복사해서라도 원하는 shape을 만들어줍니다. 그래서 항상 성공하지만, 언제 복사가 일어나는지는 겉으로 잘 드러나지 않습니다.
```python
b = a.t()
b.reshape(4, 3)               # 성공! 내부적으로 필요하면 복사해서 처리
```
메모리 공유 여부를 명확히 통제하고 싶을 때는 `view` + `.contiguous()`를 명시적으로 쓰고, 편하게 shape만 바꾸고 싶을 때는 `reshape`을 쓰는 것이 실무 기준입니다.

**`flatten`**: 사실상 `reshape`의 특수 케이스로, "여러 차원을 하나로 합치기"만 할 수 있고 `view`/`reshape`처럼 임의의 shape 재배열은 못 합니다. 대신 12절에서 다룬 대로 `start_dim`으로 batch 차원을 안전하게 보존하는 것이 기본 동작이라, CNN → Linear 연결처럼 목적이 명확한 상황에서 가장 읽기 쉽습니다.
```python
x.flatten(start_dim=1)         # (32, 2352)
x.reshape(32, -1)              # 동일한 결과, -1은 "나머지를 자동 계산"
```

**언제 뭘 쓰나**: shape을 자유롭게 재배열해야 하면 `reshape`(안전), 성능이 중요하고 연속성이 확실하면 `view`. 다차원을 벡터로 펼치기만 하면 `flatten` 또는 `nn.Flatten()`. `view`가 연속성 에러로 실패하면 `.contiguous().view(...)` 또는 그냥 `reshape(...)`으로 바꾸면 해결됩니다.

세 함수 모두 shape 인자에 `-1`을 하나 넣으면 나머지 차원 크기를 원소 수로부터 자동 계산해줍니다. 전체 크기를 일일이 계산할 필요 없이 batch만 고정하고 나머지를 펼칠 때 자주 씁니다.
