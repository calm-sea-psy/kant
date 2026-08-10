# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy matplotlib

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 그래프에 한글 라벨을 쓰므로 한글 폰트를 지정합니다.
# (지정하지 않으면 라벨이 사각형(□□□)으로 깨져 보입니다)
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']   # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 음수 부호가 깨지지 않도록


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
        print('[안내] 대체 데이터로 진행합니다. 분석 흐름은 동일합니다.')
        from sklearn.datasets import make_classification
        Xa, ya = make_classification(n_samples=4000, n_features=10,
                                     n_informative=5, random_state=RANDOM_STATE)
        cols = [f'feature_{i}' for i in range(Xa.shape[1])]
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(222)   # Bank Marketing
X_df = numeric_frame(X_raw)

y = pd.Series(y_raw)
# 주의: 최신 pandas는 문자열 컬럼의 기본 dtype이 object가 아니라 str이다.
# 그래서 `y.dtype == 'object'`로 분기하면 매핑이 건너뛰어지고,
# 바로 아랫줄 astype(int)에서 ValueError가 난다.
# pandas 버전에 상관없이 동작하도록 '숫자가 아니면 매핑'으로 판단한다.
if not pd.api.types.is_numeric_dtype(y):
    y = y.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    if y.isna().any():
        raise ValueError(f'매핑되지 않은 타깃 값 {int(y.isna().sum())}건이 있습니다.')
y = y.astype(int).values

print('수치형 특성:', X_df.shape, '/ 타깃 분포:', np.bincount(y))


# ===================================================
# 문제 1-1 : 표준화가 필요한지 데이터로 확인하기
# ===================================================
means = X_df.mean()
stds = X_df.std()

print('\n--- 문제 1-1 ---')
print('특성별 평균:\n', means)
print('\n특성별 표준편차:\n', stds)

# 표준편차 스케일이 특성마다 얼마나 벌어져 있는지 하나의 숫자로 요약
std_ratio = stds.max() / stds.min()
print('\n표준편차 최댓값/최솟값 비율:', std_ratio)

# 평균 0, 표준편차 1로 맞춘 표준화 데이터
Xs = StandardScaler().fit_transform(X_df)
print('\n표준화 후 평균 최대 절댓값(0에 가까움):', np.max(np.abs(Xs.mean(axis=0))))
print('표준화 후 표준편차 (1에 가까움):', np.allclose(Xs.std(axis=0), 1))

# 표준화 전/후 각각 PCA를 적용해 PC1 설명분산비가 어떻게 달라지는지 비교
pca_raw = PCA(n_components=2).fit(X_df.to_numpy())
pca_std = PCA(n_components=2).fit(Xs)

print('\nPC1 설명분산비 (표준화 X):', pca_raw.explained_variance_ratio_[0])
print('PC1 설명분산비 (표준화 O):', pca_std.explained_variance_ratio_[0])

print("""
설명:
- 특성마다 단위(스케일)가 크게 다르면, 표준화 없이 PCA를 적용할 경우 값의 절대적 크기가 큰
  특성이 분산을 독점해 다른 특성들의 정보가 묻힌다. 표준화를 먼저 해야 모든 특성이 동등한
  비중으로 반영된 주성분을 얻을 수 있다.
""")


# ===================================================
# 문제 1-2 : PCA로 2차원 지도 그리기
# ===================================================
pca2 = PCA(n_components=2)
Z = pca2.fit_transform(Xs)   # 45,211명을 PC1·PC2 좌표 2개로 압축

print('--- 문제 1-2 ---')
print('Z shape:', Z.shape)
print('설명분산비:', pca2.explained_variance_ratio_)
print('누적 설명분산비:', np.sum(pca2.explained_variance_ratio_))

# 가입 여부(y)를 색으로 구분해 2차원 지도 위에 고객들을 흩뿌림
plt.figure(figsize=(7, 6))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='coolwarm', s=5, alpha=.4)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('고객 2차원 지도 (색: 가입 여부)')
plt.colorbar(scatter, label='가입 여부')
plt.show()

# components_의 각 행이 PC1, PC2를 구성하는 원본 특성별 가중치(로딩)
loadings = pd.DataFrame(pca2.components_.T, index=X_df.columns, columns=['PC1', 'PC2'])
top3_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(3)
top3_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(3)

print('\nPC1에 가장 기여한 특성 top3:\n', top3_pc1)
print('\nPC2에 가장 기여한 특성 top3:\n', top3_pc2)

print("""
설명:
- 2차원 지도에서는 고객들이 대략 어떤 그룹으로 나뉘는지, 가입 고객(색)이 특정 영역에 몰려있는지
  등 전반적인 경향은 읽을 수 있다.
- 하지만 원본 특성 값이 정확히 얼마였는지, 개별 고객이 왜 그 위치에 놓였는지 같은 세부적인
  인과관계는 2차원 지도만으로 알 수 없다(설명분산비가 100%가 아니므로 정보 손실이 존재한다).
""")


# ===================================================
# 문제 2-1 : 원본 공간에서 유사 고객 찾기
# ===================================================
Xs_small = Xs[:2000]   # 전체 4.5만 명 대신 2000명만 잘라서 실습(계산량 조절)

print('--- 문제 2-1 ---')
start = time.time()
sim_full = cosine_similarity(Xs_small)   # (2000, 2000) 유사도 행렬: 고객 쌍마다 코사인 유사도
elapsed_full = time.time() - start
print('원본 공간 유사도 계산 시간:', elapsed_full, '초')
print('유사도 행렬 shape:', sim_full.shape, '/ 원소 수:', sim_full.size)
print('메모리 사용량:', sim_full.nbytes, f'바이트 ({sim_full.nbytes / 1024 / 1024:.2f} MB)')

sim0 = pd.Series(sim_full[0])
top5_full = sim0.drop(0).sort_values(ascending=False).head(5)   # 0번 고객 자기 자신(유사도=1)은 제외
print('\n0번 고객과 유사도 상위 5명 (원본 공간):')
print(top5_full)

print("""
설명:
- 특성 수가 많아지면 벡터 하나당 계산량도 늘지만, 훨씬 큰 부담은 샘플 수가 많을 때 유사도
  행렬(샘플수 x 샘플수)의 크기가 제곱으로 커진다는 점이다.
""")


# ===================================================
# 문제 2-2 : 축소 공간에서 유사 고객 찾고 비교하기
# ===================================================
print('--- 문제 2-2 ---')
pca_full_fit = PCA().fit(Xs)   # 가능한 모든 주성분(7개)에 대해 설명분산비를 구함
cumulative = np.cumsum(pca_full_fit.explained_variance_ratio_)
k_search = int(np.searchsorted(cumulative, 0.80) + 1)   # 누적 80%를 처음 넘기는 주성분 개수
print('검색용 차원 k_search (80% 기준):', k_search)

# 시각화용 2차원과는 별개로, 검색 정밀도를 위한 k_search차원 좌표를 따로 만듦
pca_search = PCA(n_components=k_search)
Z_search_full = pca_search.fit_transform(Xs)
Z_search = Z_search_full[:2000]
Z_small_2d = Z[:2000]   # 문제 1-2에서 만든 시각화용 2차원 좌표 재사용

start = time.time()
sim_2d = cosine_similarity(Z_small_2d)   # 2차원(시각화용) 좌표로 계산한 유사도
elapsed_2d = time.time() - start

start = time.time()
sim_search = cosine_similarity(Z_search)   # k_search차원(검색용) 좌표로 계산한 유사도
elapsed_search = time.time() - start

print('\n2차원 공간 유사도 계산 시간:', elapsed_2d, '초')
print('k_search차원 공간 유사도 계산 시간:', elapsed_search, '초')

sim0_2d = pd.Series(sim_2d[0])
top5_2d = sim0_2d.drop(0).sort_values(ascending=False).head(5)

sim0_search = pd.Series(sim_search[0])
top5_search = sim0_search.drop(0).sort_values(ascending=False).head(5)

print('\n0번 고객과 유사도 상위 5명 (2차원 공간):')
print(top5_2d)
print('\n0번 고객과 유사도 상위 5명 (k_search차원 공간):')
print(top5_search)

# 원본 공간의 top5 명단과 각 축소 공간의 top5 명단이 몇 명이나 겹치는지(교집합 크기)
overlap_2d = len(set(top5_full.index) & set(top5_2d.index))
overlap_search = len(set(top5_full.index) & set(top5_search.index))
print('\n원본 vs 2차원 겹치는 인원 수:', overlap_2d, '/ 5')
print('원본 vs k_search차원 겹치는 인원 수:', overlap_search, '/ 5')

# 유사도 행렬 전체를 1차원으로 펼쳐서, 원본 공간 순위와 축소 공간 순위가 얼마나 비슷한지 상관계수로 확인
corr_2d = np.corrcoef(sim_full.ravel(), sim_2d.ravel())[0, 1]
corr_search = np.corrcoef(sim_full.ravel(), sim_search.ravel())[0, 1]
print('\n원본 vs 2차원 유사도행렬 전체 상관계수:', corr_2d)
print('원본 vs k_search차원 유사도행렬 전체 상관계수:', corr_search)

# 속도 대신 "저장 용량" 관점에서 세 공간을 비교
n_features = Xs.shape[1]
storage_df = pd.DataFrame({
    '공간': ['원본', f'검색용({k_search}차원)', '시각화용(2차원)'],
    '차원 수': [n_features, k_search, 2],
    'nbytes': [Xs.nbytes, Z_search_full.nbytes, Z.nbytes],
})
storage_df['절감률(%)'] = (1 - storage_df['nbytes'] / Xs.nbytes) * 100
print('\n저장 용량 비교:')
print(storage_df)

print("""
설명:
- 이 데이터는 특성 수가 원래 한 자리 수라 차원을 줄여도 유사도 계산 자체의 속도 이득은 거의
  없다. 유사도 행렬 계산 비용은 차원 수가 아니라 샘플 수의 제곱이 지배하기 때문이다.
- 시각화가 목적이면 2차원처럼 극단적으로 압축해도 무방하지만, 유사 고객 검색처럼 정밀도가
  중요한 작업에는 누적 설명분산비가 충분히 높은 차원을 따로 써야 원본과 유사한 결과를 얻을 수
  있다. 저장 공간을 아끼려는 목적이라면 두 경우 모두 원본보다 크게 절감된다.
""")


# ===================================================
# 심화 1 문제 3-1 : NumPy 직접 구현과 sklearn 결과 대조하기
# ===================================================
print('--- 문제 3-1 ---')
# full_matrices=False가 핵심: True로 두면 U가 (45211, 45211)이 되어 메모리 부족으로 죽는다
U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
print('U shape:', U.shape, '/ S shape:', S.shape, '/ Vt shape:', Vt.shape)

# PCA 좌표 = U에 특이값(S)을 곱한 것과 같음 (U*S == 데이터를 주성분 방향으로 투영한 값)
Z_manual = U[:, :2] * S[:2]
print('\nZ_manual vs sklearn Z (절댓값 비교):', np.allclose(np.abs(Z_manual), np.abs(Z)))

# 3장 방식(공분산행렬 고유분해)으로도 같은 투영을 다시 계산해 세 번째 비교 대상 확보
cov = np.cov(Xs, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(-eigvals)   # eigh는 오름차순이라 내림차순으로 재정렬
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]
Z_eig = Xs @ eigvecs_sorted[:, :2]

# 부호(+/-)가 다를 수 있어 값 자체가 아니라 상관계수로 "같은 방향인지"만 비교
pc1_corr_svd_sklearn = np.corrcoef(Z_manual[:, 0], Z[:, 0])[0, 1]
pc1_corr_eig_sklearn = np.corrcoef(Z_eig[:, 0], Z[:, 0])[0, 1]
pc1_corr_svd_eig = np.corrcoef(Z_manual[:, 0], Z_eig[:, 0])[0, 1]

print('\nPC1 상관계수:')
print('  SVD vs sklearn:', pc1_corr_svd_sklearn)
print('  고유분해 vs sklearn:', pc1_corr_eig_sklearn)
print('  SVD vs 고유분해:', pc1_corr_svd_eig)

n_samples = Xs.shape[0]
explained_ratio_svd = (S[:2] ** 2) / np.sum(S ** 2)   # 특이값 제곱 비율 = 설명분산비
explained_ratio_eig = eigvals_sorted[:2] / eigvals_sorted.sum()   # 고유값 비율 = 설명분산비

ratio_compare = pd.DataFrame({
    'SVD': explained_ratio_svd,
    '고유분해': explained_ratio_eig,
    'sklearn': pca2.explained_variance_ratio_,
}, index=['PC1', 'PC2'])
print('\n설명분산비 3방식 비교:')
print(ratio_compare)

print("""
설명:
- 고유벡터(또는 특이벡터)는 방향이 v이든 -v이든 수학적으로 동일한 조건(Av=λv)을 만족하므로,
  계산 알고리즘의 내부 구현에 따라 부호가 다르게 나올 수 있다. 상관계수의 절댓값이나 부호를
  맞춘 뒤 비교해야 이 문제를 피할 수 있다.
- 라이브러리를 그대로 신뢰하기보다 직접 구현한 결과와 대조해 보면 결과가 왜 그렇게 나오는지
  내부 원리를 이해하게 되고, 사용법 오류(잘못된 axis, 정규화 누락 등)를 스스로 발견할 수 있어
  실무에서 결과에 대한 신뢰도를 높일 수 있다.
""")