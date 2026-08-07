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


X_raw, y_raw = load_uci(17)   # Breast Cancer Wisconsin
Xs = StandardScaler().fit_transform(numeric_frame(X_raw))
cov = np.cov(Xs, rowvar=False)      # 대칭이며 정사각인 행렬
print('표준화 데이터:', Xs.shape, '/ 공분산행렬:', cov.shape)


def normalize(v):
    return v / np.linalg.norm(v)


# ===================================================
# 문제 1-1 : Av = λv 검증하기
# ===================================================
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print('\n--- 문제 1-1 ---')
print('고유값:', eigenvalues)
print('고유벡터:\n', eigenvectors)

lam1, v1 = eigenvalues[0], eigenvectors[:, 0]
lam2, v2 = eigenvalues[1], eigenvectors[:, 1]

print('\nA@v1:', A @ v1, '/ λ1*v1:', lam1 * v1, '/ 일치:', np.allclose(A @ v1, lam1 * v1))
print('A@v2:', A @ v2, '/ λ2*v2:', lam2 * v2, '/ 일치:', np.allclose(A @ v2, lam2 * v2))

v1_scaled = 3 * v1
print('\nA@(3v1):', A @ v1_scaled, '/ λ1*(3v1):', lam1 * v1_scaled,
      '/ 일치:', np.allclose(A @ v1_scaled, lam1 * v1_scaled))

print("""
설명:
- 고유벡터는 스칼라를 곱해도(3배 등) 여전히 같은 고유값에 대한 고유벡터 조건을 만족한다. 즉
  고유벡터가 나타내는 건 "특정 방향"이며, 그 방향 위의 어떤 크기의 벡터든 같은 성질을 갖는다.
  그래서 고유벡터는 크기가 아니라 방향으로만 의미를 갖는다(관례상 노름 1로 정규화해 표현한다).
""")


# ===================================================
# 문제 1-2 : 특성방정식과 일반 벡터 비교하기
# ===================================================
print('--- 문제 1-2 ---')
I2 = np.eye(2)
for lam in eigenvalues:
    det_val = np.linalg.det(A - lam * I2)
    print(f'det(A - {lam:.1f}I) = {det_val:.10f} (0에 가까움: {np.isclose(det_val, 0)})')

non_eigen_val = 3.0   # A의 고유값(5, 2)에 없는 값
det_non_eigen = np.linalg.det(A - non_eigen_val * I2)
print(f'\ndet(A - {non_eigen_val}I) = {det_non_eigen} (0 아님: {not np.isclose(det_non_eigen, 0)})')

u = np.array([1.0, 0.0])   # 고유벡터가 아닌 일반 벡터
u_after = A @ u
print('\n일반 벡터 u=[1,0]:')
print('변환 전 방향:', normalize(u))
print('변환 후 방향:', normalize(u_after))
print('방향 유지 여부:', np.allclose(normalize(u), normalize(u_after)))

v1_after = A @ v1
print('\n고유벡터 v1:')
print('변환 전 방향:', normalize(v1))
print('변환 후 방향:', normalize(v1_after))
print('방향 유지 여부:', np.allclose(normalize(v1), normalize(v1_after)))

print("""
설명:
- 특성방정식 det(A-λI)=0은 "(A-λI)가 특이행렬이 되는 λ"를 찾는 식이다. (A-λI)가 특이행렬이라는
  것은 (A-λI)v=0을 만족하는 0이 아닌 v(영공간의 원소)가 존재한다는 뜻이고, 이는 Av=λv를
  의미하므로, 이 방정식을 만족하는 λ가 바로 고유값이 된다.
""")


# ===================================================
# 문제 2-1 : 공분산행렬 고유분해하기
# ===================================================
print('--- 문제 2-1 ---')
print('cov 대칭 확인 (np.allclose):', np.allclose(cov, cov.T))

eigvals, eigvecs = np.linalg.eigh(cov)   # eigh는 오름차순으로 반환
order = np.argsort(-eigvals)             # 내림차순으로 재정렬
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

print('\n상위 5개 고유값:', eigvals_sorted[:5])
print('전체 고유값 합:', eigvals_sorted.sum())
print('상위 5개가 차지하는 비율:', eigvals_sorted[:5].sum() / eigvals_sorted.sum())

print('\n고유값이 전부 실수인가:', np.all(np.isreal(eigvals_sorted)))
print('고유값이 전부 0 이상인가:', np.all(eigvals_sorted >= -1e-10))

print("""
설명:
- 공분산행렬은 대칭행렬(더 정확히는 양의 준정부호 행렬)이므로, 고유값이 항상 실수이고 0 이상이
  되는 것이 선형대수적으로 보장된다. 분산은 음수가 될 수 없으므로, "그 방향의 분산"을 뜻하는
  고유값도 음수가 나올 수 없는 것이 자연스럽다.
""")


# ===================================================
# 문제 2-2 : 고유벡터의 직교성과 eig/eigh 비교하기
# ===================================================
pc1 = eigvecs_sorted[:, 0]
pc2 = eigvecs_sorted[:, 1]

print('--- 문제 2-2 ---')
print('pc1·pc2 내적:', np.dot(pc1, pc2), '(0에 가까움:', np.isclose(np.dot(pc1, pc2), 0), ')')

V = eigvecs_sorted
VtV = V.T @ V
print('VᵀV == I (np.allclose):', np.allclose(VtV, np.eye(VtV.shape[0])))

eigvals_eig, eigvecs_eig = np.linalg.eig(cov)
VtV_eig = eigvecs_eig.T @ eigvecs_eig

ortho_error_eigh = np.max(np.abs(VtV - np.eye(VtV.shape[0])))
ortho_error_eig = np.max(np.abs(VtV_eig - np.eye(VtV_eig.shape[0])))
print('\neigh 직교성 오차 (max|VᵀV-I|):', ortho_error_eigh)
print('eig  직교성 오차 (max|VᵀV-I|):', ortho_error_eig)

is_sorted_eigh_raw = np.all(np.diff(eigvals) >= 0)
print('\neigh 원본 결과가 오름차순 정렬되어 나오는가:', is_sorted_eigh_raw)

order_eig = np.argsort(-eigvals_eig)
eigvals_eig_sorted = eigvals_eig[order_eig].real
diff_eigvals = np.linalg.norm(eigvals_sorted - eigvals_eig_sorted)
print('정렬을 맞춘 뒤 eigh vs eig 고유값 차이(노름):', diff_eigvals)

print("""
설명:
- 대칭행렬 전용 함수인 eigh는 대칭성을 활용한 수치적으로 더 안정적인 알고리즘을 사용하므로
  직교성 오차가 매우 작고, 고유값을 항상 정렬된 순서로 반환하는 것이 보장된다. 반면 일반 행렬용
  eig는 이런 보장이 없어 직교성 오차가 더 크고 정렬 순서도 임의일 수 있다. 그래서 공분산행렬처럼
  대칭행렬을 다룰 때는 반드시 eigh를 사용해야 한다.
""")


# ===================================================
# 심화 1 문제 3-1 : 투영값의 분산과 고유값 비교하기
# ===================================================
print('--- 문제 3-1 ---')
z1 = Xs @ pc1
var_z1 = np.var(z1, ddof=1)
print('z1(1번째 고유벡터 투영) 분산:', var_z1, '/ 1번째 고유값:', eigvals_sorted[0],
      '/ 일치:', np.isclose(var_z1, eigvals_sorted[0]))

z2 = Xs @ pc2
var_z2 = np.var(z2, ddof=1)
print('z2(2번째 고유벡터 투영) 분산:', var_z2, '/ 2번째 고유값:', eigvals_sorted[1],
      '/ 일치:', np.isclose(var_z2, eigvals_sorted[1]))

rng = np.random.default_rng(RANDOM_STATE)
random_dir = rng.standard_normal(Xs.shape[1])
random_dir = normalize(random_dir)
z_random = Xs @ random_dir
var_random = np.var(z_random, ddof=1)
print('\n임의의 단위벡터 방향 투영 분산:', var_random)
print('1번째 고유값(최대 분산)보다 작은가:', var_random < eigvals_sorted[0])

plt.figure(figsize=(6, 4))
plt.hist(z1, bins=30, alpha=.7)
plt.title('z1 (1번째 고유벡터 방향 투영값) 히스토그램')
plt.xlabel('z1 값')
plt.ylabel('빈도')
plt.show()

print("""
설명:
- 고유값이 큰 방향을 우선 선택하면, 그 방향으로 투영했을 때 데이터의 분산(=정보량)을 가장 많이
  보존할 수 있다. 분산이 크다는 것은 그 방향으로 데이터 포인트들이 서로 잘 구분된다는 뜻이므로,
  적은 수의 축(차원)만 남기고도 원본 데이터가 가진 차이를 최대한 유지할 수 있다.
- 반대로 임의의 방향이나 고유값이 작은 방향을 선택하면 투영값들이 서로 뭉쳐버려(분산이 작아져)
  원본 데이터가 가진 차이를 제대로 구분하지 못하게 되므로 차원 축소에 불리하다.
""")