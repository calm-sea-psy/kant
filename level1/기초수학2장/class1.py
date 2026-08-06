# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_uci(dataset_id):
    """UCI에서 데이터를 불러오고, 실패하면 구조가 비슷한 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=dataset_id)
        X, y = ds.data.features.copy(), ds.data.targets.copy()
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        return X, y
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 대체 데이터(sklearn breast cancer)로 진행합니다.')
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


def plot_pair(before, after, label_after, title):
    """변환 전후를 같은 축에서 비교합니다."""
    plt.figure(figsize=(6, 6))
    plt.scatter(before[:, 0], before[:, 1], s=10, alpha=.5, label='original')
    plt.scatter(after[:, 0], after[:, 1], s=10, alpha=.5, label=label_after)
    plt.axhline(0, lw=.8, color='gray'); plt.axvline(0, lw=.8, color='gray')
    plt.gca().set_aspect('equal')       # 축 비율을 맞춰야 회전이 왜곡되지 않음
    plt.legend(); plt.title(title); plt.show()


def apply_T(M, X):
    """변환을 '열벡터 관점'으로 적용합니다.

    이론에서는 벡터를 열(column)으로 놓고 M @ v 로 쓰지만,
    데이터 X는 한 샘플이 한 '행'으로 들어 있습니다.
    그래서 (M @ v)를 모든 행에 한 번에 계산하려면 X @ M.T 가 됩니다.

    이 실습의 모든 변환은 이 함수를 사용해 적용합니다.
    X @ M 처럼 직접 곱하면 M.T를 적용하는 결과가 되어
    회전 방향과 합성 순서가 이론과 반대로 나오니 주의하세요.
    """
    return np.asarray(X) @ np.asarray(M).T


X_raw, y_raw = load_uci(17)   # Breast Cancer Wisconsin
X2 = StandardScaler().fit_transform(numeric_frame(X_raw).iloc[:, :2])
print('좌표평면 데이터 X2:', X2.shape)

# 헬퍼 함수 검산: (1,0)을 45도 회전하면 (0.71, 0.71)이 나와야 합니다.
_theta = np.deg2rad(45)
_R = np.array([[np.cos(_theta), -np.sin(_theta)],
               [np.sin(_theta),  np.cos(_theta)]])
print('apply_T 검산:', np.round(apply_T(_R, np.array([1.0, 0.0])), 4))

I2 = np.eye(2)


# ===================================================
# 문제 1-1 : 스케일링 행렬 적용하기
# ===================================================
S = np.array([[2, 0], [0, 0.8]])
X2_scaled = apply_T(S, X2)

print('\n--- 문제 1-1 ---')
print('변환 전 표준편차:', X2.std(axis=0))
print('변환 후 표준편차:', X2_scaled.std(axis=0))
print('배율(after/before):', X2_scaled.std(axis=0) / X2.std(axis=0))

plot_pair(X2, X2_scaled, 'scaled', '스케일링 (x2, y0.8)')

print("""
설명:
- S의 (0,0) 원소는 x축(1번째 열)의 배율을, (1,1) 원소는 y축(2번째 열)의 배율을 담당한다.
  대각 원소가 정확히 각 축의 표준편차 배율로 그대로 반영된다.
""")


# ===================================================
# 문제 1-2 : 반사와 투영 행렬 적용하기
# ===================================================
F = np.array([[1, 0], [0, -1]])
X2_reflected = apply_T(F, X2)

norm_before = np.linalg.norm(X2, axis=1)
norm_after_F = np.linalg.norm(X2_reflected, axis=1)

print('--- 문제 1-2 ---')
print('반사 전후 원점까지 거리 유지 (np.allclose):', np.allclose(norm_before, norm_after_F))

P = np.array([[1, 0], [0, 0]])
X2_projected = apply_T(P, X2)
print('투영 결과의 2번째 축이 모두 0인가:', np.allclose(X2_projected[:, 1], 0))

FF = F @ F
PP = P @ P
print('\nF @ F == I:', np.allclose(FF, I2))
print('P @ P == I:', np.allclose(PP, I2))

test_points = np.array([[1, 2], [1, -5]])
print('[1,2], [1,-5]를 투영한 결과:\n', apply_T(P, test_points))

print("""
설명:
- 반사(F)는 F@F=I가 되어 두 번 적용하면 원상복구되므로 정보를 잃지 않는다.
- 투영(P)은 P@P=P(≠I)가 되어, 한 번 누르고 나면 원래의 y값 정보가 사라져 다시 적용해도
  복구되지 않는다. 즉 반사는 가역적(정보 보존), 투영은 비가역적(정보 손실)인 변환이다.
""")


# ===================================================
# 필수 2 : 회전 행렬 적용하기
# ===================================================
theta = np.deg2rad(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

check = apply_T(R, np.array([1.0, 0.0]))

print('--- 필수 2 (회전) ---')
print('(1,0)을 45도 회전:', np.round(check, 4), '/ (0.71,0.71)에 근접:', np.allclose(check, [0.7071, 0.7071], atol=1e-3))

X2_rotated = apply_T(R, X2)
plot_pair(X2, X2_rotated, 'rotated', '45도 회전')

norm_before_r = np.linalg.norm(X2, axis=1)
norm_after_r = np.linalg.norm(X2_rotated, axis=1)
print('회전 전후 원점까지 거리 유지:', np.allclose(norm_before_r, norm_after_r))

RtR = R.T @ R
print('RᵀR == I:', np.allclose(RtR, I2))

print("""
설명:
- RᵀR=I는 R이 직교행렬(orthogonal matrix)임을 뜻하며, 회전은 방향만 바꿀 뿐 벡터의
  길이(노름)는 절대 바꾸지 않는 변환임을 의미한다.
""")


# ===================================================
# 문제 2-2 : 합성 변환의 순서 비교하기
# ===================================================
M1 = R @ S   # 스케일링 후 회전
M2 = S @ R   # 회전 후 스케일링

print('--- 문제 2-2 ---')
print('M1 (R@S, 스케일 후 회전) =\n', M1)
print('M2 (S@R, 회전 후 스케일) =\n', M2)
print('M1 == M2:', np.allclose(M1, M2))

X2_m1 = apply_T(M1, X2)
X2_m2 = apply_T(M2, X2)
print('두 결과가 서로 다른가:', not np.allclose(X2_m1, X2_m2))

plot_pair(X2_m1, X2_m2, 'R then S (M2)', '합성 순서 비교: M1(파랑) vs M2(주황)')

print("""
설명:
- 변환 A를 먼저, B를 나중에 적용하면 v -> B(A(v)) = B@(A@v) = (B@A)@v가 되어 전체 변환
  행렬은 BA(나중에 적용하는 쪽이 왼쪽)가 된다. 행렬곱은 교환법칙이 성립하지 않으므로
  "스케일 후 회전"(RS)과 "회전 후 스케일"(SR)은 서로 다른 결과를 낳는다.
""")


# ===================================================
# 심화 1 문제 3-1 : 가산성·동차성으로 선형 변환 판별하기
# ===================================================
bias = np.array([3, -2])


def T_linear(v):
    return apply_T(M1, v)


def T_affine(v):
    return apply_T(M1, v) + bias


u = np.array([1.0, 2.0])
v = np.array([-3.0, 4.0])
c = 2.5

print('--- 문제 3-1 ---')
rows = []
for name, T in [('T_linear', T_linear), ('T_affine', T_affine)]:
    additive = np.allclose(T(u + v), T(u) + T(v))
    homogeneous = np.allclose(T(c * u), c * T(u))
    origin_preserved = np.allclose(T(np.zeros(2)), 0)
    rows.append({
        '변환': name,
        '가산성 T(u+v)=T(u)+T(v)': additive,
        '동차성 T(cu)=cT(u)': homogeneous,
        '원점보존 T(0)=0': origin_preserved,
    })

result_df = pd.DataFrame(rows)
print(result_df)

X2_linear = T_linear(X2)
X2_affine = T_affine(X2)
plot_pair(X2_linear, X2_affine, 'affine (with bias)', '선형 변환 vs 아핀 변환')

print("""
설명:
- 평행이동(bias)이 더해지면 원점이 더 이상 원점으로 유지되지 않고(T(0)=bias≠0), 가산성과
  동차성도 깨지므로 엄밀한 의미의 선형 변환이 아니다(이런 형태를 아핀 변환이라 부른다).
- 그럼에도 XW+b가 딥러닝에서 널리 쓰이는 이유는, 순수 선형 변환(원점 고정)만으로는 데이터가
  원점에서 벗어난 위치에 있는 경우를 표현할 수 없어 모델의 표현력이 크게 제한되기 때문이다.
  편향 b를 더해 원점 제약을 없애면 훨씬 다양한 함수 형태를 학습할 수 있다.
""")