# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패하면 구조가 비슷한 대체 데이터를 사용합니다."""
    import time
    n_retry, last_err = 4, None
    for attempt in range(1, n_retry + 1):
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=dataset_id)
            X, y = ds.data.features.copy(), ds.data.targets.copy()
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            return X, y
        except Exception as e:
            last_err = e
            print(f'[안내] UCI 로드 실패 ({attempt}/{n_retry}회):', e)
            if attempt < n_retry:
                time.sleep(2 * attempt)   # 일시적 네트워크 오류를 대비한 지수 대기

    if last_err is not None:
        print('=' * 72)
        print('[경고] UCI 로드가 최종 실패해 대체 데이터(make_regression)로 진행합니다.')
        print('[경고] 대체 데이터는 컬럼이 서로 독립으로 생성되어 상관계수가 거의 0입니다.')
        print('[경고] 따라서 문제 2-1의 "상관 0.876인데도 rank가 5"라는 실데이터 해석은')
        print('[경고] 이 대체 데이터로는 재현되지 않습니다. 상관계수 숫자 자체가 아니라')
        print('[경고] "상관이 높은 것과 rank가 깎이는 것은 별개"라는 논리에만 집중하세요.')
        print('=' * 72)
        print('[최종 오류]', last_err)
        from sklearn.datasets import make_regression
        Xa, ya = make_regression(n_samples=2000, n_features=5, noise=10,
                                 random_state=RANDOM_STATE)
        cols = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear']
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(601)   # AI4I 2020 Predictive Maintenance
X_df = numeric_frame(X_raw)
A = StandardScaler().fit_transform(X_df)
print('센서 데이터 행렬 A:', A.shape)
print('컬럼:', list(X_df.columns))


# ===================================================
# 문제 1-1 : 목표 벡터를 선형결합으로 표현하기
# ===================================================
v1 = np.array([1, 1])
v2 = np.array([1, -1])
target = np.array([3, 5])

V = np.column_stack([v1, v2])
coeffs, *_ = np.linalg.lstsq(V, target, rcond=None)
c1, c2 = coeffs
reconstructed = c1 * v1 + c2 * v2

print('\n--- 문제 1-1 ---')
print(f'c1={c1:.4f}, c2={c2:.4f}')
print('복원된 벡터:', reconstructed, '/ 목표:', target)
print('일치 여부 (np.allclose):', np.allclose(reconstructed, target))

print("""
설명:
- v1, v2는 서로 평행하지 않는(선형독립) 두 벡터이므로, 이 둘의 선형결합으로 2차원 평면의 어떤
  벡터든 표현할 수 있다. 즉 v1, v2가 만드는 span은 2차원 평면 전체(R^2)이다.
""")


# ===================================================
# 문제 1-2 : 평행한 벡터들의 span 확인하기
# ===================================================
w1 = np.array([1, 2])
w2 = np.array([2, 4])   # = 2 * w1

W = np.column_stack([w1, w2])
rank_W = np.linalg.matrix_rank(W)

print('--- 문제 1-2 ---')
print('W = [w1, w2] rank:', rank_W)

target1 = np.array([3, 5])
coeffs1, *_ = np.linalg.lstsq(W, target1, rcond=None)
recon1 = W @ coeffs1
print('\n[3,5]를 표현 시도 -> 복원:', recon1, '/ 목표와 일치:', np.allclose(recon1, target1))

target2 = np.array([2, 4])   # w1의 배수
coeffs2, *_ = np.linalg.lstsq(W, target2, rcond=None)
recon2 = W @ coeffs2
print('[2,4]를 표현 시도 -> 복원:', recon2, '/ 목표와 일치:', np.allclose(recon2, target2))

print("""
설명:
- w1, w2는 서로 평행(w2=2*w1)이라 선형독립이 아니며 rank는 1이다. 두 벡터의 선형결합으로는
  w1이 가리키는 방향의 직선 위 점들만 만들 수 있어, 그 직선 위에 없는 [3,5]는 표현되지 않지만
  직선 위에 있는 [2,4](w1의 배수)는 표현된다. 즉 평행한 두 벡터의 span은 평면이 아니라 원점을
  지나는 직선(1차원)이다.
""")


# ===================================================
# 문제 2-1 : 데이터 행렬의 rank 계산하기
# ===================================================
rank_A = np.linalg.matrix_rank(A)
max_rank = min(A.shape)

print('--- 문제 2-1 ---')
print('A shape:', A.shape)
print('rank(A):', rank_A, '/ 가능한 최대 rank(min(shape)):', max_rank)

corr = pd.DataFrame(A, columns=X_df.columns).corr()
mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
pairs = corr.where(mask).stack()
top_pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(2)

print('\n상관계수 절댓값 최상위 2개 컬럼 쌍:')
print(top_pairs)

print("""
설명:
- 상관계수가 0.87대로 강하게 연관된 컬럼 쌍이 있어도, 그 관계가 "완전히 정확한 배수 관계
  (상관계수 = 1.0 또는 -1.0)"가 아닌 한 두 컬럼은 여전히 선형독립이라서 rank는 컬럼 수(5) 그대로
  유지된다. rank가 실제로 깎이려면 상관계수의 절댓값이 정확히 1.0이어야 한다.
""")


# ===================================================
# 문제 2-2 : 선형종속 컬럼을 추가해도 rank가 늘지 않음 확인하기
# ===================================================
dep_col1 = (A[:, 0] * 2).reshape(-1, 1)          # 첫 컬럼의 배수
dep_col2 = (A[:, 1] + A[:, 2]).reshape(-1, 1)     # 두 컬럼의 합
A_dep = np.hstack([A, dep_col1, dep_col2])

print('--- 문제 2-2 ---')
print('A     shape:', A.shape, '/ rank:', np.linalg.matrix_rank(A))
print('A_dep shape:', A_dep.shape, '/ rank:', np.linalg.matrix_rank(A_dep))

rng = np.random.default_rng(RANDOM_STATE)
rand_col = rng.standard_normal((A.shape[0], 1))
A_rand = np.hstack([A, rand_col])
print('A_rand shape:', A_rand.shape, '/ rank:', np.linalg.matrix_rank(A_rand))

print("""
설명:
- A_dep처럼 기존 컬럼들의 선형결합(배수, 합 등)으로 만들어진 컬럼을 추가하면, 그 컬럼은 이미
  기존 컬럼들의 span 안에 있는 벡터라서 rank가 늘지 않는다. 반면 A_rand처럼 기존 컬럼들의
  선형결합으로 설명되지 않는 새로운(독립적인) 컬럼을 추가하면 rank가 실제로 1 늘어난다.
""")


# ===================================================
# 심화 1 문제 3-1 : 영공간의 비자명한 해로 중복 관계 확인하기
# ===================================================
n_cols = A_dep.shape[1]   # A(5개) + dep_col1(1개) + dep_col2(1개) = 7

print('--- 문제 3-1 ---')
print('A_dep 컬럼 수:', n_cols)

# 관계1: A_dep[:,0]*2 = A_dep[:,5]  ->  2*col0 - col5 = 0
c1 = np.zeros(n_cols)
c1[0] = 2
c1[5] = -1
null_check1 = A_dep @ c1
print('\nc1 (2*컬럼0 - 컬럼5 = 0):', c1)
print('A_dep @ c1 ≈ 0 (np.allclose):', np.allclose(null_check1, 0))

# 관계2: A_dep[:,1] + A_dep[:,2] = A_dep[:,6]  ->  col1 + col2 - col6 = 0
c2 = np.zeros(n_cols)
c2[1] = 1
c2[2] = 1
c2[6] = -1
null_check2 = A_dep @ c2
print('\nc2 (컬럼1 + 컬럼2 - 컬럼6 = 0):', c2)
print('A_dep @ c2 ≈ 0 (np.allclose):', np.allclose(null_check2, 0))

rank_dep = np.linalg.matrix_rank(A_dep)
nullity_dep = n_cols - rank_dep
print(f'\nA_dep: 컬럼 수={n_cols}, rank={rank_dep}, 영공간 차원(컬럼수-rank)={nullity_dep}')

rank_A_only = np.linalg.matrix_rank(A)
nullity_A = A.shape[1] - rank_A_only
print(f'A (중복 없음): 컬럼 수={A.shape[1]}, rank={rank_A_only}, 영공간 차원={nullity_A}')

print("""
설명:
- A_dep의 (컬럼 수 7 - rank 5) = 영공간 차원 2로, 실제로 찾은 두 개의 독립적인 중복 관계
  (c1, c2)와 정확히 일치한다.
- 중복이 없는 A는 컬럼 수와 rank가 같아 영공간이 {0}뿐이며, 이는 Ax=0을 만족하는 x가 영벡터
  밖에 없다는 뜻이다.
- 영공간의 차원이 0보다 크다는 것은 데이터에 실질적으로 새로운 정보를 주지 않는 중복 특성이
  존재한다는 뜻이다. 이런 컬럼을 그대로 모델에 넣으면 다중공선성 문제(계수가 불안정해지는,
  class4에서 다룬 조건수 폭증과 동일한 문제)를 일으키므로, 영공간이 알려주는 정확한 계수 관계를
  참고해 중복 컬럼을 제거하거나 PCA 등으로 차원을 축소하는 것이 좋다.
""")