# 딥러닝 기초와 PyTorch

> 출처 TIL: 260819

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
