# 실습 : 벡터 노름 정규화..


# 최초 1회만 실행 (새 환경일 때)
# Colab에는 ucimlrepo가 기본 설치되어 있지 않으므로 이 셀을 건너뛰지 마세요.
#$!pip install -q ucimlrepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def _fallback(reason):
    """UCI 데이터를 쓸 수 없을 때 대체 데이터셋을 반환합니다."""
    print(f'[경고] {reason}')
    print('[경고] UCI Wine Quality를 불러오지 못해 다른 데이터셋(sklearn wine)으로 대체됩니다.')
    print('[경고] 따라서 아래 출력값은 교안의 예시 값과 다르게 나옵니다. 계산·해석 방법은 동일합니다.')
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.data, data.target


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패 원인(설치/네트워크)을 구분해 안내합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return _fallback('ucimlrepo 패키지가 설치되어 있지 않습니다. 위의 pip 설치 셀을 먼저 실행하세요.')

    try:
        ds = fetch_ucirepo(id=dataset_id)
    except Exception as e:
        return _fallback(f'UCI 서버 접속에 실패했습니다(네트워크·방화벽 확인 필요): {e}')

    X, y = ds.data.features.copy(), ds.data.targets.copy()
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]  # 컬럼 1개짜리 DataFrame -> Series
    return X, y


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(186)   # Wine Quality
X = numeric_frame(X_raw)
print('데이터 shape:', X.shape)


# ===================================================
# 문제 1-1 : 스칼라·벡터·행렬 구분하기
# ===================================================
v1 = X.iloc[0].to_numpy()

print('\n--- 문제 1-1 ---')
print('X (행렬) shape:', X.shape)
print(X.head())
print('\nv1 (벡터):', v1)
print('v1 shape:', v1.shape)
print('\nv1[0] (스칼라):', v1[0])
print('v1[0] type:', type(v1[0]))

print("""
설명:
- X는 (샘플 수 x 특성 수) 크기의 2차원 배열이므로 행렬이며, 여러 샘플의 측정값 전체를 담는다.
- v1은 1차원 배열로, 한 샘플의 여러 측정값을 하나로 묶은 벡터이며 분석의 기본 단위가 된다.
- v1[0]은 단일 숫자인 스칼라로, 벡터를 구성하는 개별 측정값 하나를 나타낸다.
""")


# ===================================================
# 필수 2-1 : 노름 계산
# ===================================================
l1 = np.linalg.norm(v1, 1)
l2 = np.linalg.norm(v1, 2)
linf = np.linalg.norm(v1, np.inf)
l2_manual = np.sqrt(np.sum(v1 ** 2))

print('\n--- 필수 2-1 ---')
print(f'L1 노름: {l1:.4f}')
print(f'L2 노름: {l2:.4f}')
print(f'L∞ 노름: {linf:.4f}')
print(f'L2 노름(직접 계산): {l2_manual:.4f}')
print('L2 노름 일치 여부:', np.isclose(l2, l2_manual))

print("""
설명:
- L1 노름은 모든 측정값의 절댓값을 단순히 합산하므로 각 항목의 변동을 고르게 반영한다.
- L2 노름은 제곱합의 제곱근(유클리드 거리)으로, 값이 큰 항목의 영향을 더 크게 반영한다.
- L∞ 노름은 가장 큰 절댓값 하나만 반영하므로 가장 두드러진 측정값을 대표한다.
""")


# ===================================================
# 필수 2-2 : 정규화 (단위벡터)
# ===================================================
u1 = v1 / np.linalg.norm(v1, 2)

print('\n--- 필수 2-2 ---')
print('u1 (정규화된 벡터):', u1)
print('u1의 L2 노름:', np.linalg.norm(u1, 2))
print('L2 노름이 1인지 확인:', np.isclose(np.linalg.norm(u1, 2), 1.0))

first5 = X.iloc[:5].to_numpy()
rows = []
for i, row in enumerate(first5):
    norm_before = np.linalg.norm(row, 2)
    normalized = row / norm_before
    norm_after = np.linalg.norm(normalized, 2)
    rows.append({'샘플': i, '정규화 전 L2 노름': norm_before, '정규화 후 L2 노름': norm_after})

compare_df = pd.DataFrame(rows)
print('\n앞 5개 샘플 정규화 전후 노름 비교:')
print(compare_df)

print("""
설명:
- 정규화는 벡터의 방향(각 항목 간 상대적 비율)은 그대로 유지하면서, 크기(L2 노름)만 1로 통일한다.
- 따라서 서로 다른 크기의 샘플이라도 방향만 비교할 수 있게 된다.
""")
