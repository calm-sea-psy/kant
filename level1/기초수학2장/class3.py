# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
        print('[안내] 대체 데이터(sklearn wine)로 진행합니다.')
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        return data.data, data.target


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


def describe_solution(A, b):
    """rank 비교로 해의 종류를 판별합니다."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    n_vars = A.shape[1]
    if rank_A < rank_Ab:
        kind = '해 없음(불능)'
    elif rank_A == n_vars:
        kind = '유일해'
    else:
        kind = '무한해'
    return {'rank(A)': rank_A, 'rank([A|b])': rank_Ab, '변수 수': n_vars, '판정': kind}


X_raw, y_raw = load_uci(186)   # Wine Quality
print('데이터 shape:', X_raw.shape)


# ===================================================
# 문제 1-1 : 조건을 Ax = b로 정리하고 풀기
# ===================================================
A = np.array([[2, 1], [1, 3]])   # 원액A·B의 알코올/산도 기여
b = np.array([8, 13])            # 목표 알코올, 목표 산도

print('\n--- 문제 1-1 ---')
print('A shape:', A.shape, '/ b shape:', b.shape)

x = np.linalg.solve(A, b)
print('해 x:', x)

check = A @ x
print('A@x:', check, '/ b:', b, '/ 일치 (np.allclose):', np.allclose(check, b))
print(f'\n원액 A {x[0]:.2f}L, 원액 B {x[1]:.2f}L를 섞으면 됩니다.')


# ===================================================
# 문제 1-2 : 역행렬 해법과 solve 비교하기
# ===================================================
x_inv = np.linalg.inv(A) @ b
x_solve = np.linalg.solve(A, b)
diff = np.linalg.norm(x_inv - x_solve)

print('\n--- 문제 1-2 ---')
print('inv 방식 해:', x_inv)
print('solve 방식 해:', x_solve)
print('두 해의 차이(노름):', diff)

A_ill = np.array([[2, 1], [2.000001, 1.0000004]])
b_ill = np.array([8, 8.000002])
cond_ill = np.linalg.cond(A_ill)
print('\nA_ill 조건수:', cond_ill)

x_ill_solve = np.linalg.solve(A_ill, b_ill)
x_ill_inv = np.linalg.inv(A_ill) @ b_ill
print('A_ill solve 해:', x_ill_solve)
print('A_ill inv  해:', x_ill_inv)
print('두 방식 차이(노름):', np.linalg.norm(x_ill_solve - x_ill_inv))

b_ill2 = b_ill.copy()
b_ill2[1] += 0.0000001
x_ill_solve2 = np.linalg.solve(A_ill, b_ill2)
print('\nb를 미세하게(0.0000001) 바꾼 뒤 해:', x_ill_solve2)
print('원래 해와의 차이(노름):', np.linalg.norm(x_ill_solve2 - x_ill_solve))

print("""
설명:
- 조건수가 매우 큰(불안정한) 시스템에서는 입력값의 아주 작은 변화(b의 소수점 자리 변화)에도
  해가 크게 요동친다. 역행렬을 직접 구하는 방식은 이런 불안정성을 증폭시키기 쉬우므로, 실무에서는
  np.linalg.inv(A)@b보다 수치적으로 더 안정적인 np.linalg.solve(A, b)를 권장한다.
""")


# ===================================================
# 문제 2-1 : rank 비교로 해의 종류 판별하기
# ===================================================
systems = {
    '유일해': (np.array([[2, 1], [1, 3]]), np.array([8, 13])),
    '무한해': (np.array([[2, 1], [4, 2]]), np.array([8, 16])),
    '불능':   (np.array([[2, 1], [4, 2]]), np.array([8, 20])),
}

print('--- 문제 2-1 ---')
rows = []
for name, (Ai, bi) in systems.items():
    result = describe_solution(Ai, bi)
    result['시스템'] = name
    rows.append(result)

result_df = pd.DataFrame(rows)[['시스템', 'rank(A)', 'rank([A|b])', '변수 수', '판정']]
print(result_df)

for name, (Ai, bi) in systems.items():
    print(f'\n[{name}] solve 시도:')
    try:
        xi = np.linalg.solve(Ai, bi)
        print('  결과:', xi)
    except np.linalg.LinAlgError as e:
        print('  오류:', e)

print("""
설명:
- 유일해: 두 직선이 기울기가 달라 정확히 한 점에서 만난다.
- 무한해: 두 식이 서로 배수 관계라 사실상 같은 직선이며, 그 직선 위의 모든 점이 해가 된다.
- 불능: 좌변(기울기)은 배수 관계지만 우변(절편)은 그 배수를 따르지 않아, 두 직선이 평행하며
  절대 만나지 않는다.
""")


# ===================================================
# 심화 1 문제 3-1 : 최소제곱해와 정규방정식 비교하기
# ===================================================
X_df = numeric_frame(X_raw)
X_std = StandardScaler().fit_transform(X_df)
n_samples = X_std.shape[0]
Xd = np.hstack([np.ones((n_samples, 1)), X_std])   # 절편(1) 컬럼을 앞에 붙임
y = np.asarray(y_raw, dtype=float)

print('--- 문제 3-1 ---')
print('Xd shape:', Xd.shape, '(방정식 수 = 샘플 수, 미지수 수 = 특성 수 + 절편)')

solution_desc = describe_solution(Xd, y)
print('Xd x = y 해 종류 판정:', solution_desc)

coef_lstsq, residuals, rank_lstsq, sv = np.linalg.lstsq(Xd, y, rcond=None)
y_pred_lstsq = Xd @ coef_lstsq
rmse_lstsq = np.sqrt(mean_squared_error(y, y_pred_lstsq))
print('\n최소제곱해(lstsq) RMSE:', rmse_lstsq)

XtX = Xd.T @ Xd
Xty = Xd.T @ y
coef_normal = np.linalg.solve(XtX, Xty)

diff_coef = np.linalg.norm(coef_lstsq - coef_normal)
cond_XtX = np.linalg.cond(XtX)

print('lstsq 계수 vs 정규방정식 계수 차이(노름):', diff_coef)
print('XᵀX 조건수:', cond_XtX)

print("""
설명:
- 방정식(샘플) 수가 미지수(계수) 수보다 훨씬 많은 과잉결정계에서는 모든 방정식을 정확히
  만족하는 해가 일반적으로 존재하지 않는다. 하지만 오차 제곱합을 최소화하는 최소제곱해는
  항상 구할 수 있으므로(lstsq는 SVD 기반이라 XᵀX가 특이행렬이어도 동작함), 정확한 해가 없는
  문제에서도 항상 근사해를 얻을 수 있다.
- 정규방정식(XᵀXβ=Xᵀy)은 XᵀX의 조건수가 클 경우 수치적으로 불안정해질 수 있는 반면, lstsq는
  SVD 기반으로 계산되어 훨씬 안정적이기 때문에 실무에서는 정규방정식을 직접 푸는 것보다
  lstsq(또는 그 원리를 내장한 회귀 라이브러리)를 사용하는 것이 권장된다.
""")