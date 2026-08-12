## 실습 배경

데이터팀은 새 분류 과제에 사용할 첫 후보군을 정해야 합니다. 같은 데이터에서도 Logistic Regression, KNN, Decision Tree의 결과가 다르고, label이 없는 데이터에서는 분류 지표 대신 군집 구조를 확인해야 합니다. 이번 실습에서는 하나의 모델을 정답처럼 외우지 않고, 데이터와 요구사항에 따라 후보를 비교하고 선택 근거를 남깁니다.

분류 모델 비교, label 없는 Wine 데이터의 구조 탐색, 데이터 조건별 후보 추천을 차례대로 수행한 뒤 최종 test 결과를 한 번 보고합니다.

## 실습 목표

- Logistic Regression·KNN·Decision Tree를 같은 분할과 지표로 비교할 수 있습니다.
- 거리 기반 모델에 scaling이 필요한 이유를 설명할 수 있습니다.
- validation AP 동률을 F1·Recall·고정 우선순위로 처리할 수 있습니다.
- K-means의 K 후보를 Silhouette로 비교하고 PCA 좌표를 시각화할 수 있습니다.
- target 유무·문제 유형·설명 필요성·희소 고차원 여부로 후보군을 바꿀 수 있습니다.
- validation에서 선택한 모델 family를 유지하고 test를 마지막에 한 번 평가할 수 있습니다.

## 진행 방식

- **사용 환경:** Google Colab 또는 Jupyter Notebook
- **사용 라이브러리:** Python 3.x, pandas, NumPy, scikit-learn, matplotlib
- **구성:** 필수 2개 + 심화 1개
- 공통 준비 코드를 실행한 뒤 문제를 순서대로 진행하세요.
- 문제 1의 분류 모델은 같은 train·validation 분할에서 비교하세요.
- 문제 2의 K-means에는 Wine 정답 label을 학습 입력으로 사용하지 마세요.
- 문제 3에서 최종 모델을 선택한 뒤 test를 한 번만 확인하세요.
- 외부 파일과 네트워크 연결은 필요하지 않습니다.

## 오늘의 업무 흐름

```
문제 유형과 데이터 계약 확인
→ 동일 분할에서 분류 모델 비교
→ AP 동률 규칙으로 후보 선택
→ label 없는 군집 구조 탐색
→ 데이터 조건별 후보군 추천
→ 선택 모델 family 고정
→ 최종 test 보고
```

## 입력 데이터 카드

| 구분 | Wisconsin Breast Cancer | Wine recognition |
| --- | --- | --- |
| 출처 | scikit-learn 내장 실제 데이터 | scikit-learn 내장 실제 데이터 |
| 전체 Shape | `X=(569, 30)` | `X=(178, 13)` |
| target 사용 | 악성=1, 양성=0 | K-means 학습에는 사용하지 않음 |
| 분할 | train 341 / validation 114 / test 114 | 전체 178행으로 구조 탐색 |
| 전처리 | Logistic·KNN만 StandardScaler | 모든 특성 StandardScaler |
| 핵심 지표 | AP → F1 → Recall → 고정 우선순위 | Silhouette, PCA 설명분산비 |
| 결측·범주 | 결측 없음, 수치형 30개 | 결측 없음, 수치형 13개 |

분류 문제의 고정 우선순위는 `logistic → knn → tree`입니다. AP·F1·Recall이 모두 같을 때만 이 순서를 사용합니다.

# 필수 1. 세 가지 분류 모델을 같은 조건에서 비교하기

## ▶ 문제 1-1: validation 모델 비교와 동률 처리

### 업무 요청

세 후보 모델 중 하나를 후속 실험에 사용할 모델 family로 정해야 합니다. 모델마다 다른 split을 사용하면 표본 차이와 모델 차이를 구분할 수 없으므로 같은 train과 validation을 사용하세요. 악성 탐지가 목적이므로 AP를 첫 기준으로 사용하고 동률 규칙까지 명시하세요.

### 수행해야 할 작업

1. Logistic Regression과 KNN에 `StandardScaler`를 포함하세요.
2. Decision Tree는 원래 특성 단위로 학습하세요.
3. 세 모델의 validation AP·F1·Recall을 계산하세요.
4. AP, F1, Recall 내림차순과 고정 우선순위 오름차순으로 정렬하세요.
5. 코드로 계산한 선택 결과와 같은 규칙의 `max()` 결과가 일치하는지 확인하세요.
6. 선택 모델의 장점과 이 데이터에서만 유효한 결과라는 한계를 작성하세요.

시작 코드

def compare_classifiers(models, X_train, y_train, X_valid, y_valid):
    """세 분류 후보의 validation 표와 선택 모델 이름을 반환합니다."""
    # 모든 후보는 동일한 train·validation 행과 악성=1 정의를 공유해야 합니다.
    # AP 동률을 임의로 깨지 말고 F1·Recall·고정 우선순위까지 명시적으로 적용하세요.
    # TODO 1: 같은 validation에서 AP·F1·Recall을 계산하세요.
    # TODO 2: 동률 처리 규칙을 적용하세요.
    raise NotImplementedError("TODO: 분류 모델 비교를 완성하세요.")

제출해야 할 결과    

model | AP | F1 | Recall | priority

[모델 선택 보고]
- 1차 지표:
- 동률 처리 순서:
- 선택 모델:
- 선택 이유:
- 이 결과를 다른 데이터에 그대로 일반화할 수 없는 이유:

필수 2. label 없이 Wine 군집 구조 탐색하기

## 문제 2-1: K 후보 비교와 PCA 시각화

### 업무 요청

Wine 데이터의 품종 label을 보지 않은 상태에서 화학 성분만으로 자연스러운 군집 구조가 있는지 확인해야 합니다. K를 임의로 하나 고정하지 말고 2부터 6까지 Silhouette를 비교하고, PCA 2차원 그림은 보조 자료로만 사용하세요.

### 수행해야 할 작업

1. Wine의 13개 수치 특성을 표준화하세요.
2. K=2부터 6까지 같은 seed로 K-means를 학습하세요.
3. 각 K의 Silhouette를 표로 만들고 가장 높은 K를 선택하세요.
4. PCA 2차원 좌표와 설명분산비를 계산하세요.
5. 선택 K의 군집 label로 산점도를 그리세요.
6. PCA 그림만으로 군집 품질을 확정하면 안 되는 이유를 작성하세요.

시작 코드

def explore_wine_clusters(wine_X, k_values=range(2, 7)):
    """K별 Silhouette 표와 PCA 좌표·설명분산비를 반환합니다."""
    # wine_X는 label을 제외한 178×13 수치 특성이며 거리 계산 전에 척도를 맞춰야 합니다.
    # PCA 그림은 보조 설명용이고 K 선택은 전체 특성의 Silhouette로 수행하세요.
    # TODO 1: scaling 후 K별 Silhouette를 계산하세요.
    # TODO 2: 가장 높은 K와 PCA 2차원 좌표를 구하세요.
    raise NotImplementedError("TODO: Wine 군집 탐색을 완성하세요.")


제출해야 할 결과

k | silhouette
PCA explained ratio: [...]
best_k: ...

[군집 탐색 보고]
- 선택 K:
- 선택 근거:
- PC1+PC2 설명분산비:
- 그림만으로 품질을 확정할 수 없는 이유:

심화 1. 데이터 조건에 따라 후보 모델 추천하기

문제 3-1: 추천 체크리스트와 최종 선택 모델 연결

### 업무 요청

동료가 데이터 조건과 무관하게 항상 같은 세 모델을 추천하는 함수를 작성했습니다. target 유무, 문제 유형, 설명 필요성, 희소 고차원 여부가 실제 후보 목록을 바꾸도록 수정하세요. 마지막에는 문제 1에서 선택한 바로 그 모델 family를 개발 데이터 전체에 학습하고 test를 한 번 평가하세요.

### 수행해야 할 작업

1. target이 없으면 KMeans·PCA와 비지도 검증 방법을 반환하세요.
2. 분류와 회귀에 서로 다른 기본 후보를 반환하세요.
3. 설명이 필요하면 선형 모델과 얕은 트리를 앞쪽에 배치하세요.
4. 희소 고차원 입력이면 선형 계열 후보로 목록을 바꾸세요.
5. 두 조건이 추천 결과를 실제로 바꾸는지 assertion으로 확인하세요.
6. 문제 1의 `selected_template`을 train+validation에 다시 학습하고 test AP·F1을 한 번 출력하세요.

시작 코드

def recommend_candidates(
    has_target,
    task=None,
    needs_explanation=False,
    sparse_high_dimensional=False,
):
    """데이터 조건에 맞는 후보 모델과 검증 방법을 반환합니다."""
    # target 유무와 task가 먼저 후보군을 결정하며 두 boolean 조건도 결과에 실제 영향을 줘야 합니다.
    # 이 함수는 탐색 시작점을 제안할 뿐 validation 우승 모델을 대신 선택하지 않습니다.
    # TODO 1: target과 task에 따라 기본 후보를 만드세요.
    # TODO 2: 설명 필요성과 희소 고차원 조건을 반영하세요.
    raise NotImplementedError("TODO: 후보 모델 추천 규칙을 완성하세요.")

제출해야 할 보고 형식    

조건 | 추천 후보 3개 | 검증 방법

[후보 추천 및 최종 평가 보고]
- 기본 분류 후보:
- 설명 필요 후보:
- 희소 고차원 후보:
- validation 선택 모델:
- 최종 test AP / F1:
- 추천 목록은 시작점일 뿐인 이유:

최종 제출 보고

[모델 후보 선정 보고]
1. 세 분류 모델의 validation 비교표와 동률 처리 규칙
2. 선택 모델과 선택 근거
3. Wine K별 Silhouette, 선택 K, PCA 설명분산비
4. 기본·설명 필요·희소 고차원 조건별 후보 목록
5. 선택 모델 family의 최종 test AP·F1
6. 다음 실험에서 사용할 CV와 업무 지표

## 실습 마무리 체크리스트

- [ ]  세 분류 모델을 같은 train·validation에서 비교할 수 있습니다.
- [ ]  KNN에 scaling이 필요한 이유를 설명할 수 있습니다.
- [ ]  AP 동률을 F1·Recall·고정 우선순위로 처리할 수 있습니다.
- [ ]  K-means 학습에 정답 label을 사용하지 않을 수 있습니다.
- [ ]  Silhouette와 PCA 그림의 역할을 구분할 수 있습니다.
- [ ]  데이터 조건이 추천 후보 목록을 실제로 바꾸게 할 수 있습니다.
- [ ]  validation 선택 모델 family를 유지하고 test를 한 번 평가할 수 있습니다.