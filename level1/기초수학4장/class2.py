# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def synthetic_retail(n_customers=150, n_products=200, n_latent=6,
                     n_active=2, mean_q=1.5, seed=RANDOM_STATE):
    """저랭크 잠재요인 구조를 가진 대체 거래 데이터를 만듭니다.

    고객은 소수(n_active)의 취향 세그먼트에만 반응하고, 상품도 주로 한 세그먼트에
    속합니다. 따라서 구매량 행렬이 '소수의 잠재요인 + 약한 노이즈'로 설명되며,
    저랭크 근사·추천 실습이 실데이터와 같은 방향의 결론을 주게 됩니다.
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


def customer_product_matrix(df, n_customers=80, n_products=60):
    """고객(행) x 상품(열) 구매량 행렬을 만듭니다."""
    df = df.dropna(subset=['CustomerID', 'StockCode']).copy()
    pivot = df.pivot_table(index='CustomerID', columns='StockCode',
                           values='Quantity', aggfunc='sum', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).head(n_products).index]
    return pivot.astype(float)


retail = load_retail()
M_df = customer_product_matrix(retail, 80, 60)
M = M_df.values
print('고객-상품 행렬 M:', M.shape)


# ===================================================
# 문제 1-1 : 중심화 후 SVD의 Vᵀ와 PCA 주성분 비교하기
# ===================================================
M_mean = M.mean(axis=0)
Mc = M - M_mean

print('\n--- 문제 1-1 ---')
print('중심화 후 열 평균 최대 절댓값 (0에 가까움):', np.max(np.abs(Mc.mean(axis=0))))

U, S, Vt = np.linalg.svd(Mc, full_matrices=False)

pca = PCA(n_components=5)
pca.fit(M)   # sklearn은 내부에서 자동으로 중심화를 수행

print('\nVt[:5] vs pca.components_ (절댓값 비교):')
print('일치 (np.allclose):', np.allclose(np.abs(Vt[:5]), np.abs(pca.components_)))

print("""
설명:
- 상품 수가 수천 개가 되면 공분산행렬(상품수 x 상품수) 자체를 만드는 데 많은 메모리와 계산이
  필요하다. 반면 중심화한 데이터에 SVD를 바로 적용하면 공분산행렬을 만들지 않고도 동일한
  주성분을 얻을 수 있어, sklearn은 내부적으로 SVD 방식을 사용한다.
""")


# ===================================================
# 문제 1-2 : 특이값에서 설명 분산 계산하기
# ===================================================
n_samples = M.shape[0]
explained_variance_manual = S ** 2 / (n_samples - 1)
explained_ratio_manual = explained_variance_manual / explained_variance_manual.sum()

print('--- 문제 1-2 ---')
print('직접 계산 설명분산(상위5):', explained_variance_manual[:5])
print('pca.explained_variance_(상위5):', pca.explained_variance_)
print('일치:', np.allclose(explained_variance_manual[:5], pca.explained_variance_))

print('\n설명분산비(상위5, 직접계산):', explained_ratio_manual[:5])
print('설명분산비(상위5, pca):', pca.explained_variance_ratio_)
print('일치:', np.allclose(explained_ratio_manual[:5], pca.explained_variance_ratio_))

cumulative_5 = np.cumsum(explained_ratio_manual[:5])
print('\n상위 5개 누적 설명분산비:', cumulative_5)

print("""
설명:
- 특이값 S는 그 방향으로 데이터를 투영했을 때의 퍼짐 정도(표준편차)에 비례하는 값이므로,
  특이값이 클수록 그 방향의 분산(정보량)이 크다. 특이값을 제곱해서 (n-1)로 나누면 정확히
  그 방향의 분산(고유값)이 된다.
""")


# ===================================================
# 문제 2-1 : 4단계 파이프라인 구현하기
# ===================================================
print('--- 문제 2-1 ---')
M_scaled = StandardScaler().fit_transform(M)

U2, S2, Vt2 = np.linalg.svd(M_scaled, full_matrices=False)

explained_var2 = S2 ** 2 / (M_scaled.shape[0] - 1)
explained_ratio2 = explained_var2 / explained_var2.sum()
cumulative2 = np.cumsum(explained_ratio2)

k = int(np.searchsorted(cumulative2, 0.80) + 1)
print('누적 설명분산비 80% 넘는 최소 k:', k)

Z = U2[:, :k] * S2[:k]
print('Z shape:', Z.shape)

pca_k = PCA(n_components=k)
Z_pca = pca_k.fit_transform(M_scaled)
print('SVD 투영 vs sklearn PCA 투영 (절댓값 비교):', np.allclose(np.abs(Z), np.abs(Z_pca)))

plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], s=15, alpha=.6)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('SVD 기반 PCA 투영 (PC1 vs PC2)')
plt.show()


# ===================================================
# 문제 2-2 : 표준화를 생략하면 어떻게 되는지 확인하기
# ===================================================
print('--- 문제 2-2 ---')
Mc_nostd = M - M.mean(axis=0)   # 중심화만, 표준화는 안 함
_, S_nostd, Vt_nostd = np.linalg.svd(Mc_nostd, full_matrices=False)

explained_var_nostd = S_nostd ** 2 / (M.shape[0] - 1)
ratio_nostd = explained_var_nostd / explained_var_nostd.sum()

print('PC1 설명분산비 (표준화 X):', ratio_nostd[0])
print('PC1 설명분산비 (표준화 O):', explained_ratio2[0])
print('-> 이 지표만으로는 표준화 효과가 잘 안 드러남')

loading_concentration_nostd = np.max(np.abs(Vt_nostd[0]))
loading_concentration_std = np.max(np.abs(Vt2[0]))
print('\nPC1 로딩 집중도 max|Vt[0]| (표준화 X):', loading_concentration_nostd)
print('PC1 로딩 집중도 max|Vt[0]| (표준화 O):', loading_concentration_std)

cumulative_nostd = np.cumsum(ratio_nostd)
k_80_nostd = int(np.searchsorted(cumulative_nostd, 0.80) + 1)
print('\n80% 도달 k (표준화 X):', k_80_nostd)
print('80% 도달 k (표준화 O):', k)

top3_idx = np.argsort(-np.abs(Vt_nostd[0]))[:3]
top3_products = M_df.columns[top3_idx]
print('\n표준화 안 한 PC1에서 절댓값 상위 3개 상품:', list(top3_products))

col_stds = M_df.std(axis=0)
stds_rank = col_stds.rank(ascending=False)
print('\n그 상품들의 구매량 표준편차 순위(전체 상품 중):')
for p in top3_products:
    print(f'  {p}: 표준편차={col_stds[p]:.2f}, 순위={int(stds_rank[p])}/{len(col_stds)}')

print("""
설명:
- 표준화를 생략하면, 구매량 표준편차가 큰(단위/스케일이 큰) 소수의 상품이 PC1 방향을 거의
  독점해버려 나머지 상품들의 정보가 묻힌다. 이는 PC1 설명분산비만 봐서는 잘 드러나지 않고,
  로딩 집중도나 80% 도달에 필요한 축의 개수를 봐야 확인된다. 그래서 스케일이 서로 다른 여러
  특성을 다룰 때는 PCA 전에 반드시 표준화를 해야 왜곡 없는 주성분을 얻을 수 있다.
""")


# ===================================================
# 심화 1 문제 3-1 : TruncatedSVD 임베딩으로 유사 고객과 추천 후보 찾기
# ===================================================
print('--- 문제 3-1 ---')
svd_model = TruncatedSVD(n_components=5, random_state=RANDOM_STATE)
emb = svd_model.fit_transform(M)
print('emb shape:', emb.shape)
print('설명분산비:', svd_model.explained_variance_ratio_)

sim_with_pc1 = cosine_similarity(emb)[0]
sim_without_pc1 = cosine_similarity(emb[:, 1:])[0]

print('\n0번 고객 기준 유사도 범위 (1번 성분 포함):', sim_with_pc1.min(), '~', sim_with_pc1.max())
print('0번 고객 기준 유사도 범위 (1번 성분 제외):', sim_without_pc1.min(), '~', sim_without_pc1.max())

sim_series = pd.Series(sim_without_pc1, index=M_df.index)
top5 = sim_series.drop(M_df.index[0]).sort_values(ascending=False).head(5)
print('\n0번 고객과 유사도 상위 5명 (1번 성분 제외 기준):')
print(top5)

M_approx = svd_model.inverse_transform(emb)

base_purchases = M_df.iloc[0]
scores = pd.Series(M_approx[0], index=M_df.columns)
scores_candidates = scores[base_purchases.to_numpy() == 0].sort_values(ascending=False)

top10_candidates = scores_candidates.head(10)
compare_df = pd.DataFrame({
    '복원점수': top10_candidates,
    '원본값': [base_purchases[p] for p in top10_candidates.index],
})
print('\n추천 후보 상위 10개 (복원 점수 vs 원본값):')
print(compare_df)

print("""
설명:
- 저랭크 근사는 "적은 수의 잠재 요인(취향 패턴)만으로 고객과 상품의 관계를 설명할 수 있다"는
  가정 위에서, 그 잠재 요인들을 조합해 원래는 0이었던 칸에도 값을 채워 넣는다. 이 복원 점수가
  높다는 것은 그 고객의 잠재 취향 패턴상 그 상품과 어울릴 가능성이 크다는 뜻이므로, "아직 사지
  않았지만 살 만한 상품"의 근거가 된다.
""")