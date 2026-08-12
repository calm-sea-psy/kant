## 실습 배경

여러분은 병원 데이터팀에서 새 예측 모델의 평가 보고서를 검토하는 머신러닝 엔지니어입니다. 한 팀은 환자의 1년 뒤 질병 진행 정도를 예측하고 있고, 다른 팀은 세포핵 측정값으로 악성 종양을 탐지하고 있습니다. 두 팀 모두 높은 숫자 하나만 보고 모델을 배포하려 하지만, 회귀와 분류는 지표가 다르고 분할이나 임계값을 잘못 다루면 test 성능까지 의사결정에 섞일 수 있습니다.

이번 실습에서는 다음 질문에 답합니다.

1. 회귀 모델은 train 평균 기준 모델보다 실제로 나은가요?
2. 악성을 놓치지 않으려면 어떤 분류 지표를 함께 봐야 하나요?
3. Recall 정책을 만족하는 임계값을 어디에서 정하고, 어떤 모델과 함께 고정해야 하나요?

최종 결과는 모델 카드의 평가 항목으로 사용할 수 있는 짧은 보고서로 정리합니다.

## 실습 목표

- 회귀와 분류 문제를 구분하고 문제별 핵심 지표를 계산할 수 있습니다.
- train·validation·test의 역할을 설명하고 원본 인덱스가 겹치지 않는지 검증할 수 있습니다.
- 회귀 모델을 train 평균 기준 모델과 비교할 수 있습니다.
- 악성을 양성 클래스로 두고 Accuracy·Precision·Recall·F1·ROC-AUC·AP를 해석할 수 있습니다.
- validation Recall 정책으로 임계값을 선택하고 같은 fitted 모델과 함께 test에 적용할 수 있습니다.
- test를 다시 선택에 사용하지 않는 이유를 평가 보고서에 설명할 수 있습니다.

## 진행 방식

- **사용 환경:** Google Colab 또는 Jupyter Notebook
- **사용 라이브러리:** Python 3.x, NumPy, pandas, scikit-learn
- **구성:** 필수 2개 + 심화 1개
- 공통 준비 코드를 먼저 실행한 뒤 문제 1부터 문제 3까지 순서대로 진행하세요.
- 회귀와 분류 모델은 train에서 학습하고, 선택과 임계값 결정은 validation에서 수행하세요.
- test는 모델과 임계값을 고정한 뒤 마지막에 한 번만 확인하세요.
- 외부 파일이나 네트워크 연결은 필요하지 않습니다.

오늘의 업무 흐름

실제 데이터와 target 의미 확인
→ train / validation / test 분리
→ 기준 모델과 후보 모델 비교
→ validation 지표 해석
→ Recall 정책 임계값 선택
→ fitted 모델 + 임계값 고정
→ 최종 test 보고

## 입력 데이터 카드

| 구분 | Diabetes 회귀 | Wisconsin Breast Cancer 분류 |
| --- | --- | --- |
| 출처 | scikit-learn 내장 실제 데이터 | scikit-learn 내장 실제 데이터 |
| 전체 Shape | `X=(442, 10)`, `y=(442,)` | `X=(569, 30)`, `y=(569,)` |
| target | 1년 뒤 질병 진행 정도, 25~346 | 악성=1, 양성=0 |
| 분할 | 265 / 88 / 89 | 341 / 114 / 114 |
| 분할 방법 | `random_state=42` | `stratify=y`, `random_state=42` |
| 결측·범주 | 결측 없음, 모두 수치형 | 결측 없음, 모두 수치형 |
| 핵심 계약 | 기준값은 train target에서만 계산 | validation에서 임계값 선택, test는 봉인 |

Breast Cancer 원본 target은 `0=malignant`, `1=benign`입니다. 이번 실습에서는 탐지해야 할 악성을 positive class로 사용하기 위해 값을 뒤집습니다.


필수 1. 회귀 모델을 기준 모델과 비교하기

문제 1-1: Diabetes 회귀 평가 보고서 작성

### 업무 요청

질병 진행도 예측팀은 Linear Regression의 validation RMSE만 보고 모델이 충분히 좋다고 주장합니다. 하지만 RMSE의 크기만으로는 개선 여부를 판단하기 어렵습니다. train target 평균만 반복해서 예측하는 기준 모델을 만들고 같은 validation에서 비교하세요.

### 수행해야 할 작업

1. `StandardScaler → LinearRegression` Pipeline을 train에 학습하세요.
2. validation MAE·RMSE·R²를 계산하세요.
3. `y_reg_train.mean()`만 예측하는 기준 모델의 같은 지표를 계산하세요.
4. 세 분할의 원본 인덱스가 서로 겹치지 않는지 다시 확인하세요.
5. 모델 RMSE가 기준 모델보다 작은지 assertion으로 검증하세요.
6. 어떤 지표에서 얼마나 개선되었는지 2~3문장으로 해석하세요.

시작 코드
def build_regression_report(X_train, y_train, X_valid, y_valid):
    """회귀 모델과 train 평균 기준 모델의 validation 지표를 반환합니다."""
    # X_train과 X_valid는 같은 특성 열을 가지며, y는 연속형 target이어야 합니다.
    # 기준 평균은 validation을 보지 않고 y_train에서만 계산해야 누수가 생기지 않습니다.
    # TODO 1: Pipeline을 학습하세요.
    # TODO 2: 모델과 기준 모델의 MAE·RMSE·R²를 계산하세요.
    raise NotImplementedError("TODO: 회귀 평가 보고서를 완성하세요.")


필수 2. 악성 종양 분류 지표를 함께 읽기

## 문제 2-1: 고정 임계값 지표와 순위 지표 비교

### 업무 요청

악성 종양 탐지팀은 Accuracy 하나만 보고 모델을 승인하려고 합니다. 악성을 놓치는 비용이 크므로 Recall을 포함한 임계값 지표와 ROC-AUC·AP 같은 순위 지표를 같은 validation에서 계산하세요.

### 수행해야 할 작업

1. `StandardScaler → LogisticRegression` Pipeline을 train에 학습하세요.
2. `predict_proba()[:, 1]`로 악성 확률을 구하세요.
3. 임계값 0.5에서 Accuracy·Precision·Recall·F1을 계산하세요.
4. 확률로 ROC-AUC와 AP를 계산하세요.
5. 모든 지표가 0과 1 사이인지 assertion으로 확인하세요.
6. 임계값 지표와 순위 지표의 차이를 평가 보고에 작성하세요.

시작 코드
def build_classification_report(X_train, y_train, X_valid, y_valid):
    """분류 Pipeline과 validation 확률·지표를 반환합니다."""
    # y=1은 악성이므로 predict_proba의 양성 클래스 열을 사용해야 합니다.
    # 전처리 통계는 train에서만 학습하고 validation은 평가에만 사용하세요.
    # TODO 1: Logistic Regression Pipeline을 학습하세요.
    # TODO 2: 확률과 여섯 지표를 계산하세요.
    raise NotImplementedError("TODO: 분류 평가 보고서를 완성하세요.")


심화 1. Recall 정책으로 임계값 선택하기

문제 3-1: fitted 모델과 임계값을 한 쌍으로 고정

### 업무 요청

운영 정책은 **validation Recall 0.90 이상인 후보 중 F1이 가장 높은 임계값**을 요구합니다. 정책을 만족하는 임계값이 없으면 기준을 몰래 낮추지 말고 실패를 보고해야 합니다. 임계값을 선택한 뒤에는 그 확률을 만든 fitted `clf`와 임계값을 함께 고정하여 test를 한 번 평가하세요.

### 수행해야 할 작업

1. 0.05부터 0.95까지 0.01 간격의 임계값 표를 만드세요.
2. validation Recall 0.90 이상인 행만 남기세요.
3. F1, Precision, 임계값 순으로 동률을 처리하세요.
4. 후보가 없으면 `RuntimeError`를 발생시키세요.
5. fitted `clf`를 재학습하지 않고 test 악성 확률을 계산하세요.
6. test AP·Precision·Recall·F1을 출력하고 결과를 다시 선택에 사용하지 않는 이유를 설명하세요.

시작 코드
def choose_threshold(y_true, probability, minimum_recall=0.90):
    """Recall 정책을 만족하는 validation 임계값과 비교표를 반환합니다."""
    # probability는 validation에서 얻은 값이며 test 확률을 임계값 선택에 사용하면 안 됩니다.
    # minimum_recall은 완화 가능한 힌트가 아니라 반드시 만족해야 하는 운영 정책입니다.
    # TODO 1: 임계값별 Precision·Recall·F1을 계산하세요.
    # TODO 2: 정책 후보가 없으면 명시적으로 실패하세요.
    raise NotImplementedError("TODO: Recall 정책 임계값을 선택하세요.")

제출해야 할 보고 형식

[임계값 정책 보고]
- 정책: validation Recall >= 0.90
- 선택 임계값:
- validation Precision / Recall / F1:
- test AP / Precision / Recall / F1:
- 같은 fitted 모델을 유지한 이유:
- test를 보고 다시 선택하지 않는 이유:

최종 제출 보고

[모델 평가 승인 보고]
1. 회귀 후보와 train 평균 기준 모델의 validation 지표
2. 분류 validation 지표 여섯 개와 positive class 정의
3. Recall 정책, 선택 임계값, validation 정책 충족 여부
4. 봉인된 test 결과
5. 데이터 누수와 재선택을 막기 위해 지킨 규칙 두 가지
6. 배포 승인 / 보완 실험 필요 중 하나와 근거

## 실습 마무리 체크리스트

- [ ]  회귀와 분류에서 사용하는 지표가 다른 이유를 설명할 수 있습니다.
- [ ]  기준 모델의 평균을 train target에서만 계산할 수 있습니다.
- [ ]  세 분할의 원본 인덱스가 겹치지 않는지 검증할 수 있습니다.
- [ ]  악성을 positive class 1로 변환한 이유를 설명할 수 있습니다.
- [ ]  임계값 지표와 순위 지표의 차이를 설명할 수 있습니다.
- [ ]  fitted 모델과 validation에서 선택한 임계값을 한 쌍으로 고정할 수 있습니다.
- [ ]  test 결과를 모델 선택에 다시 사용하지 않는 이유를 설명할 수 있습니다.