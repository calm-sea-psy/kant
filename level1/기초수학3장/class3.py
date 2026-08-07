# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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


X_raw, y_raw = load_uci(186)   # Wine Quality
Xn = numeric_frame(X_raw).iloc[:, :5]           # 특성 5개
y = pd.to_numeric(y_raw).values.astype(float)

Xs = StandardScaler().fit_transform(Xn)
Xd = np.column_stack([np.ones(len(Xs)), Xs])    # 절편 컬럼 + 특성 5개
print('설계행렬 Xd:', Xd.shape, '/ 타깃 y:', y.shape)

rng = np.random.default_rng(RANDOM_STATE)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# ===================================================
# 문제 1-1 : 내적으로 직교 여부 판별하기
# ===================================================
pairs = {
    '(a,b)': (np.array([1, 0]), np.array([0, 1])),
    '(c,d)': (np.array([1, 2]), np.array([2, -1])),
    '(e,f)': (np.array([1, 2]), np.array([2, 3])),
}

print('\n--- 문제 1-1 ---')
rows = []
for name, (u, v) in pairs.items():
    dot = np.dot(u, v)
    cos = cosine(u, v)
    rows.append({'쌍': name, '내적': dot, '코사인유사도': round(cos, 4), '직교여부': bool(np.isclose(dot, 0))})

pairs_df = pd.DataFrame(rows)
print(pairs_df)

print("""
설명:
- 직교하는 두 벡터는 내적이 0이며, 이는 한 벡터의 방향 성분이 다른 벡터의 방향에 전혀 겹치지
  않는다는 뜻이다. 즉 한 벡터가 아무리 변해도 다른 벡터가 가리키는 방향에는 영향을 주지 않으므로,
  두 벡터는 서로 "독립적인 정보"를 담고 있다고 해석할 수 있다.
""")


# ===================================================
# 문제 1-2 : 직교행렬 QᵀQ = I 확인하기
# ===================================================
Q, R = np.linalg.qr(Xd)
print('--- 문제 1-2 ---')
print('Q shape:', Q.shape, '/ R shape:', R.shape)

QtQ = Q.T @ Q
print('QᵀQ == I (np.allclose):', np.allclose(QtQ, np.eye(Q.shape[1])))

col_norms = np.linalg.norm(Q, axis=0)
print('\nQ의 각 열의 노름 (전부 1이어야 함):', np.round(col_norms, 6))

off_diag = QtQ - np.diag(np.diag(QtQ))
print('Q 열들 간 직교성 (비대각선 최대 절댓값, 0에 가까워야 함):', np.max(np.abs(off_diag)))

v = rng.standard_normal(Q.shape[1])
left_inverse_check = Q.T @ (Q @ v)
print('\nQᵀ(Qv) == v (np.allclose):', np.allclose(left_inverse_check, v))

QQt = Q @ Q.T
is_identity_QQt = np.allclose(QQt, np.eye(QQt.shape[0]))
print('QQᵀ == I:', is_identity_QQt, '(shape:', QQt.shape, ')')

print("""
설명:
- 직교행렬(열이 정규직교인 Q)은 QᵀQ=I를 만족하므로, 역행렬을 따로 계산할 필요 없이 Qᵀ를
  곱하는 것만으로 "되돌리는" 연산(왼쪽 역행렬 역할)을 할 수 있어 계산이 훨씬 빠르고 수치적으로
  안정적이다.
- Q가 정사각행렬이 아닌 경우(샘플 수 > 특성 수), QQᵀ는 I가 되지 않는다. Qᵀ는 Q의 "왼쪽"
  역행렬 역할만 하고 "오른쪽" 역행렬 역할은 하지 못하기 때문이다.
""")


# ===================================================
# 문제 2-1 : 정규방정식으로 회귀계수 계산하기
# ===================================================
coef_formula = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y
coef_lstsq, *_ = np.linalg.lstsq(Xd, y, rcond=None)

lr = LinearRegression(fit_intercept=False)   # Xd에 이미 절편 컬럼이 있으므로 False로 맞춤
lr.fit(Xd, y)
coef_sklearn = lr.coef_

print('--- 문제 2-1 ---')
print('정규방정식 계수:', coef_formula)
print('lstsq 계수:     ', coef_lstsq)
print('sklearn 계수:   ', coef_sklearn)

print('\n공식 vs lstsq 차이(노름):', np.linalg.norm(coef_formula - coef_lstsq))
print('공식 vs sklearn 차이(노름):', np.linalg.norm(coef_formula - coef_sklearn))

pred_formula = Xd @ coef_formula
pred_lstsq = Xd @ coef_lstsq
pred_sklearn = Xd @ coef_sklearn

rmse_df = pd.DataFrame({
    '방식': ['정규방정식', 'lstsq', 'sklearn'],
    'RMSE': [
        np.sqrt(mean_squared_error(y, pred_formula)),
        np.sqrt(mean_squared_error(y, pred_lstsq)),
        np.sqrt(mean_squared_error(y, pred_sklearn)),
    ],
})
print('\n', rmse_df)

print("""
설명:
- 세 방식 모두 "오차 제곱합을 최소화하는 계수"라는 같은 최적화 문제를 풀고 있고, 설계행렬이
  full rank이면 그 해가 수학적으로 유일하게 결정되므로 세 방식의 결과가 (부동소수점 오차
  수준까지) 일치한다.
""")


# ===================================================
# 문제 2-2 : 잔차의 직교성으로 투영 의미 확인하기
# ===================================================
pred = Xd @ coef_lstsq
resid = y - pred

print('--- 문제 2-2 ---')
resid_dot_norm = np.linalg.norm(Xd.T @ resid)
print('‖Xᵀ·잔차‖:', resid_dot_norm, '(0에 가까움:', np.isclose(resid_dot_norm, 0, atol=1e-6), ')')

print('\n각 컬럼과 잔차의 내적:')
col_names = ['절편'] + [f'특성{i + 1}' for i in range(Xd.shape[1] - 1)]
for name, col in zip(col_names, Xd.T):
    print(f'  {name}: {np.dot(col, resid):.10f}')

coef_bad = coef_lstsq + rng.normal(0, 0.05, size=coef_lstsq.shape)
pred_bad = Xd @ coef_bad
resid_bad = y - pred_bad

print('\n임의로 바꾼 계수(coef_bad)의 경우:')
print('‖Xᵀ·잔차_bad‖:', np.linalg.norm(Xd.T @ resid_bad))
print('오차제곱합(최적해):', np.sum(resid ** 2))
print('오차제곱합(임의계수):', np.sum(resid_bad ** 2))

print("""
설명:
- 잔차가 설계행렬의 모든 열과 직교한다는 것은, 잔차 벡터가 설계행렬이 만드는 열공간(가능한 모든
  예측값들의 공간)과 완전히 수직이라는 뜻이다. 어떤 점(y)에서 한 평면(열공간)까지의 최단 거리는
  그 평면에 수직으로 내린 발(투영)까지의 거리이므로, 잔차가 열공간과 직교한다는 것은 그 예측값이
  열공간 안에서 y에 가장 가까운 점(최소제곱해)이라는 것을 의미한다. 계수를 살짝만 바꿔도 이
  직교성이 깨지고 오차제곱합이 커지는 것이 이를 뒷받침한다.
""")


# ===================================================
# 심화 1 문제 3-1 : Gram-Schmidt로 정규직교기저 만들기
# ===================================================
print('--- 문제 3-1 ---')
cols = Xd[:, 1:4]   # 절편 제외, 특성 1~3 (서로 상관이 있는 컬럼들)
print('원본 컬럼 shape:', cols.shape)


def gram_schmidt(vectors):
    U_list = []
    for v_col in vectors.T:
        w = v_col.copy().astype(float)
        for u in U_list:
            w = w - np.dot(w, u) * u   # u는 이미 정규화되어 있으므로 분모(u·u)는 1
        U_list.append(w / np.linalg.norm(w))
    return np.column_stack(U_list)


U = gram_schmidt(cols)
print('U shape:', U.shape)

UtU = U.T @ U
print('UᵀU == I (np.allclose):', np.allclose(UtU, np.eye(U.shape[1])))

Q_cols, R_cols = np.linalg.qr(cols)
print('\n직접 구현 U vs np.linalg.qr(cols)의 Q (절댓값 비교):', np.allclose(np.abs(U), np.abs(Q_cols)))

orig_corr = np.corrcoef(cols, rowvar=False)
new_corr = np.corrcoef(U, rowvar=False)

print('\n원본 컬럼 간 상관계수:\n', np.round(orig_corr, 4))
print('\n직교화 후 축 간 상관계수:\n', np.round(new_corr, 10))

print("""
설명:
- 정규직교기저를 쓰면 각 축이 서로 완전히 무관해지므로(상관계수 0), 회귀 등에서 각 축의 기여를
  다른 축과 뒤섞이지 않고 독립적으로 해석할 수 있다. 또한 축들 간 중복(다중공선성)이 사라져
  XᵀX의 조건수가 좋아지고 계산이 훨씬 안정적으로 이뤄진다.
""")