# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def synthetic_retail(n_customers=120, n_products=150, n_latent=6,
                     n_active=2, mean_q=1.5, seed=RANDOM_STATE):
    """저랭크 잠재요인 구조를 가진 대체 거래 데이터를 만듭니다.

    고객은 소수(n_active)의 취향 세그먼트에만 반응하고, 상품도 주로 한 세그먼트에
    속합니다. 따라서 구매량 행렬이 '소수의 잠재요인 + 약한 노이즈'로 설명되며,
    실제 Online Retail 데이터와 같은 저랭크 구조를 갖습니다.
    """
    rng = np.random.default_rng(seed)

    # 고객 잠재요인: 각 고객은 n_active개 세그먼트에만 관심을 가짐
    cust_f = np.zeros((n_customers, n_latent))
    for i in range(n_customers):
        active = rng.choice(n_latent, size=n_active, replace=False)
        cust_f[i, active] = rng.gamma(2.0, 1.0, size=n_active)

    # 상품 잠재요인: 상품마다 주 세그먼트 1개 + 약한 교차 수요
    prod_f = rng.gamma(0.3, 0.3, size=(n_latent, n_products))
    seg = rng.integers(0, n_latent, n_products)
    prod_f[seg, np.arange(n_products)] += rng.gamma(2.0, 1.0, n_products)

    base = cust_f @ prod_f                       # 랭크 <= n_latent 인 저랭크 신호
    base = base / base.mean() * mean_q
    lam = np.clip(base * (1 + rng.normal(0, 0.15, base.shape)), 0, None)
    counts = rng.poisson(lam)                    # 저랭크 신호 + 작은 노이즈

    cust_ids = np.repeat(np.arange(1000, 1000 + n_customers), n_products)
    prod_ids = np.tile([f'P{i:03d}' for i in range(n_products)], n_customers)
    q = counts.ravel()
    keep = q > 0                                 # 구매하지 않은 칸은 거래로 남지 않음
    m = int(keep.sum())
    return pd.DataFrame({
        'InvoiceNo': rng.integers(10000, 10900, m).astype(str),
        'StockCode': prod_ids[keep],
        'Quantity': q[keep],
        'UnitPrice': rng.gamma(2.0, 10.0, m),
        'CustomerID': cust_ids[keep],
    })


def load_retail():
    """Online Retail 거래 데이터를 불러오고, 실패하면 같은 구조의 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=352)
        return ds.data.original.copy()
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다.')
        return synthetic_retail()


def customer_product_matrix(df, n_customers=60, n_products=40):
    """고객(행) x 상품(열) 구매량 행렬을 만듭니다."""
    df = df.dropna(subset=['CustomerID', 'StockCode']).copy()
    pivot = df.pivot_table(index='CustomerID', columns='StockCode',
                           values='Quantity', aggfunc='sum', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).head(n_products).index]
    return pivot.astype(float)


retail = load_retail()
M_df = customer_product_matrix(retail, 60, 40)
M = M_df.values
print('고객-상품 행렬 M:', M.shape, '(정사각행렬이 아님)')


# ===================================================
# 문제 1-1 : SVD를 적용하고 각 행렬의 shape 해석하기
# ===================================================
U, S, Vt = np.linalg.svd(M, full_matrices=False)

print('\n--- 문제 1-1 ---')
print('M shape:', M.shape)
print('U shape:', U.shape, '(고객 60명 x 잠재축 40개)')
print('S shape:', S.shape, '(특이값 40개, 각 축의 중요도)')
print('Vt shape:', Vt.shape, '(잠재축 40개 x 상품 40개)')

Uf, Sf, Vtf = np.linalg.svd(M, full_matrices=True)
print('\nfull_matrices=True 결과:')
print('Uf shape:', Uf.shape, '/ Sf shape:', Sf.shape, '/ Vtf shape:', Vtf.shape)

is_descending = np.all(np.diff(S) <= 0)
print('\nS가 내림차순인가:', is_descending)
print('상위 5개 특이값:', S[:5])

print("""
설명:
- S는 원래 대각행렬(Σ)이지만 대각선 외 나머지가 전부 0이라 굳이 그 값들을 저장할 필요가 없다.
  그래서 NumPy는 대각 원소만 1차원 배열로 반환해 메모리를 절약한다.
""")


# ===================================================
# 문제 1-2 : 직교성 확인과 원본 복원하기
# ===================================================
print('--- 문제 1-2 ---')
UtU = U.T @ U
VtVt_T = Vt @ Vt.T
print('UᵀU == I (np.allclose):', np.allclose(UtU, np.eye(U.shape[1])))
print('VtVtᵀ == I (np.allclose):', np.allclose(VtVt_T, np.eye(Vt.shape[0])))

Sigma = np.diag(S)
M_reconstructed = U @ Sigma @ Vt
print('\nM shape:', M.shape, '/ 복원 shape:', M_reconstructed.shape)

recon_error = np.linalg.norm(M - M_reconstructed)
relative_error = recon_error / np.linalg.norm(M)
print('복원 오차(프로베니우스 노름):', recon_error)
print('상대 오차:', relative_error)

print("""
설명:
- 모든 특이값(= 전체 rank만큼)을 다 사용해서 복원했으므로, 원본 행렬이 가진 정보를 손실 없이
  그대로 재구성한 것과 같다. 따라서 부동소수점 계산 오차 수준의 극히 작은 값만 남고 사실상 0이다.
""")


# ===================================================
# 문제 2-1 : 특이값과 AᵀA 고유값의 관계 확인하기
# ===================================================
print('--- 문제 2-1 ---')
MtM = M.T @ M
print('MᵀM shape:', MtM.shape, '/ 대칭 확인:', np.allclose(MtM, MtM.T))

eigvals_MtM = np.linalg.eigvalsh(MtM)
eigvals_sorted = np.sort(eigvals_MtM)[::-1]
sqrt_eigvals = np.sqrt(np.clip(eigvals_sorted, 0, None))

compare_df = pd.DataFrame({'특이값(S)': S[:5], 'sqrt(고유값)': sqrt_eigvals[:5]})
print('\n상위 5개 비교:')
print(compare_df)
print('최대 차이:', np.max(np.abs(S - sqrt_eigvals)))

print("""
설명:
- 반올림 오차로 인해 이론적으로 0이어야 할 아주 작은 고유값이 미세하게 음수로 계산될 수 있다.
  음수에 제곱근을 취하면 nan이 나오므로, clip으로 0 미만 값을 0으로 강제해 안전하게 계산해야 한다.
""")

C_mat = np.random.default_rng(RANDOM_STATE).standard_normal((60, 5))
P_mat = np.random.default_rng(RANDOM_STATE + 1).standard_normal((5, 40))
R = C_mat @ P_mat
rank_R = np.linalg.matrix_rank(R)
print('R shape:', R.shape, '/ rank(R):', rank_R, '(<= 5 이어야 함)')

RtR = R.T @ R
eigvals_RtR = np.linalg.eigvalsh(RtR)
n_negative = int(np.sum(eigvals_RtR < 0))
print('RᵀR의 음수 고유값 개수:', n_negative)

with np.errstate(invalid='ignore'):
    sqrt_no_clip = np.sqrt(eigvals_RtR)
n_nan = int(np.sum(np.isnan(sqrt_no_clip)))
print('clip 없이 sqrt 적용 시 nan 개수:', n_nan)

sqrt_with_clip = np.sqrt(np.clip(eigvals_RtR, 0, None))
n_nan_clipped = int(np.sum(np.isnan(sqrt_with_clip)))
print('clip 적용 후 nan 개수:', n_nan_clipped)


# ===================================================
# 문제 2-2 : 고유분해와 SVD의 적용 범위 비교하기
# ===================================================
print('--- 문제 2-2 ---')
try:
    np.linalg.eig(M)
except Exception as e:
    print('M에 eig 시도 -> 오류:', type(e).__name__, ':', e)

U_M, S_M, Vt_M = np.linalg.svd(M, full_matrices=False)
print('M에 SVD 적용 -> 정상 (S shape:', S_M.shape, ')')

B = np.array([[1, 1], [0, 1]])   # 기초수학3장에서 대각화 불가능했던 행렬
U_B, S_B, Vt_B = np.linalg.svd(B)
print('\nB(대각화 불가능했던 행렬)에 SVD 적용 -> 특이값:', S_B)

comparison = pd.DataFrame({
    '항목': ['적용 대상', '결과 벡터 직교성', '수치 안정성'],
    '고유분해': ['정사각행렬만 (그중에서도 대각화 가능한 경우만)', '일반적으로 보장 안 됨(대칭행렬일 때만 직교)', '결함행렬 등에서 불안정/불가능'],
    'SVD': ['어떤 shape의 행렬이든 항상 가능', 'U, V 모두 항상 직교', '항상 안정적으로 계산 가능'],
})
print('\n', comparison)

print("""
설명:
- 행렬이 정사각이 아니거나, 정사각이어도 대각화가 보장되지 않는 경우(예: 결함행렬 B)에는
  고유분해 대신 SVD를 사용해야 한다. SVD는 어떤 행렬에도 적용 가능하고 결과가 항상 직교하며
  수치적으로 안정적이기 때문에, 데이터 압축이나 PCA 등 실무에서는 SVD가 기본으로 선호된다.
""")


# ===================================================
# 심화 1 문제 3-1 : 저랭크 근사의 오차와 정보 보존량 계산하기
# ===================================================
print('--- 문제 3-1 ---')
ks = [1, 2, 5, 10, 20]
total_energy = np.sum(S ** 2)
M_norm = np.linalg.norm(M)
n_rows, n_cols = M.shape

rows_result = []
for k in ks:
    M_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    err = np.linalg.norm(M - M_k)
    rel_err = err / M_norm
    energy = np.sum(S[:k] ** 2) / total_energy
    storage = k * (n_rows + n_cols + 1)
    storage_ratio = storage / (n_rows * n_cols)
    rank_Mk = np.linalg.matrix_rank(M_k)
    rows_result.append({
        'k': k, '복원오차': err, '상대오차': rel_err,
        '보존에너지': energy, '저장원소수': storage,
        '저장비율': storage_ratio, 'rank(M_k)': rank_Mk,
    })

result_df = pd.DataFrame(rows_result)
print(result_df)

energies_all_k = np.cumsum(S ** 2) / total_energy
k_90 = int(np.searchsorted(energies_all_k, 0.90) + 1)
print('\n보존 에너지 90% 넘는 최소 k:', k_90)

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(S) + 1), np.cumsum(S ** 2) / total_energy, marker='o', markersize=3)
plt.axhline(0.9, color='red', linestyle='--', label='90% 기준선')
plt.xlabel('k (상위 특이값 개수)')
plt.ylabel('누적 보존 에너지')
plt.title('k에 따른 보존 에너지')
plt.legend()
plt.show()

print("""
설명:
- 정보 손실 허용 기준을 정할 때는 목적(저장 공간 절감 vs 정보 보존)의 우선순위를 먼저 정해야
  한다. 보존 에너지가 급격히 늘어나다가 완만해지는 지점(elbow)을 찾으면, 그 이후로는 k를 늘려도
  얻는 정보가 적어 효율이 떨어진다는 것을 알 수 있다. 또한 데이터의 실제 잠재요인 구조에 가까운
  k를 선택하면 노이즈를 걸러내면서도 핵심 패턴은 유지하는 근사가 가능하다.
""")