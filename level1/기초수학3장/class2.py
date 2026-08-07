# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy matplotlib

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 그래프에 한글 라벨을 쓰므로 한글 폰트를 지정합니다.
# (지정하지 않으면 라벨이 사각형(□□□)으로 깨져 보입니다)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 음수 부호가 깨지지 않도록
# Colab이라면 아래 두 줄을 먼저 실행한 뒤 런타임을 재시작하세요.
# !apt -qq install -y fonts-nanum
# !rm -rf ~/.cache/matplotlib


def load_uci(dataset_id, n_retry=3, wait_sec=2.0):
    """UCI에서 데이터를 불러옵니다. 일시적인 네트워크 오류는 몇 번 재시도합니다.

    반환값: (X, y, source)  - source는 'uci' 또는 'fallback'
    """
    for attempt in range(1, n_retry + 1):
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=dataset_id)
            X, y = ds.data.features.copy(), ds.data.targets.copy()
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            return X, y, 'uci'
        except Exception as e:
            print(f'[안내] UCI 로드 실패 ({attempt}/{n_retry}):', e)
            if attempt < n_retry:
                print(f'[안내] {wait_sec}초 후 재시도합니다.')
                time.sleep(wait_sec)

    print('=' * 78)
    print('[경고] UCI Wine Quality 로드에 최종 실패해 대체 데이터(sklearn load_wine)로 진행합니다.')
    print('[경고] load_wine은 특성 수(13개)와 샘플 수가 모두 다른 "별개의" 데이터셋입니다.')
    print('[경고] 따라서 이후 모든 문제의 정답 수치가 교안에 적힌 값과 다르게 나옵니다.')
    print('[경고]   예) PC1 설명분산비  Wine Quality 약 0.275  ->  load_wine 약 0.362')
    print('[경고] 수치를 교안과 맞추려 하지 말고, 절차와 해석이 맞는지를 기준으로 확인하세요.')
    print('=' * 78)
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    return data.data, data.target, 'fallback'


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw, DATA_SOURCE = load_uci(186)   # Wine Quality
X_df = numeric_frame(X_raw)
Xs = StandardScaler().fit_transform(X_df)
print('데이터 출처:', DATA_SOURCE)
print('표준화 데이터:', Xs.shape, '/ 특성 수:', X_df.shape[1])
if DATA_SOURCE == 'fallback':
    print('>> 대체 데이터이므로 아래 문제들의 정답 수치는 교안과 다릅니다.')


# ===================================================
# 문제 1-1 : A = PDP⁻¹ 구성하고 복원하기
# ===================================================
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

P = eigenvectors
D = np.diag(eigenvalues)

A_reconstructed = (P @ D @ np.linalg.inv(P)).real

print('\n--- 문제 1-1 ---')
print('고유값:', eigenvalues)
print('P (고유벡터 행렬):\n', P.real)
print('D (대각행렬):\n', D.real)
print('\nP@D@inv(P):\n', A_reconstructed)
print('원본 A:\n', A)
print('일치 (np.allclose):', np.allclose(A_reconstructed, A))

det_P = np.linalg.det(P)
print('\nP의 행렬식:', det_P, '(0 아님 -> 역행렬 존재:', not np.isclose(det_P, 0), ')')

print("""
설명:
- 대각화(A=PDP⁻¹)는 A라는 변환을 "P로 좌표축을 고유벡터 방향으로 바꾸고 -> D로 그 축마다
  독립적으로 배율만 조정하고 -> P⁻¹로 원래 좌표축으로 되돌리는" 세 단계로 분해하는 것이다.
""")


# ===================================================
# 문제 1-2 : 대각화로 거듭제곱 계산하고, 불가능한 경우 확인하기
# ===================================================
print('--- 문제 1-2 ---')
A2_direct = A @ A
D2 = D ** 2   # 대각행렬은 원소별 제곱 = 실제 제곱과 같음
A2_diag = (P @ D2 @ np.linalg.inv(P)).real
print('A@A:\n', A2_direct)
print('P@D^2@inv(P):\n', A2_diag)
print('일치:', np.allclose(A2_direct, A2_diag))

A10_direct = np.linalg.matrix_power(A, 10)
D10 = D ** 10
A10_diag = (P @ D10 @ np.linalg.inv(P)).real
print('\nA^10 두 방식 일치:', np.allclose(A10_direct, A10_diag))

B = np.array([[1, 1], [0, 1]])
eigvals_B, eigvecs_B = np.linalg.eig(B)
print('\nB 고유값:', eigvals_B)
print('B 고유벡터:\n', eigvecs_B)

rank_B_eigvecs = np.linalg.matrix_rank(eigvecs_B)
cond_B_eigvecs = np.linalg.cond(eigvecs_B)
print('\nB의 고유벡터 행렬 rank:', rank_B_eigvecs, '(변수 수', eigvecs_B.shape[1], '와 비교)')
print('B의 고유벡터 행렬 조건수:', cond_B_eigvecs)

if rank_B_eigvecs < eigvecs_B.shape[1]:
    print('-> rank 부족: 고유벡터가 서로 독립적이지 않아 대각화 불가능 (역행렬 계산은 되지만 결과가 틀림)')
else:
    P_B = eigvecs_B
    D_B = np.diag(eigvals_B)
    B_reconstructed = (P_B @ D_B @ np.linalg.inv(P_B)).real
    print('B 복원 결과:\n', B_reconstructed)
    print('원본 B와 일치:', np.allclose(B_reconstructed, B))

print("""
설명:
- 대각화는 D의 대각원소(고유값)만 제곱/거듭제곱하면 되므로, 행렬을 여러 번 직접 곱하는 것보다
  훨씬 빠르게 거듭제곱을 계산할 수 있다.
- B는 고유벡터가 서로 독립적이지 않아(rank 부족) P가 사실상 역행렬을 가질 자격이 없는데도
  NumPy는 에러 없이 값을 내놓으므로, 복원을 시도하기 전에 rank나 조건수로 미리 판별해야 한다.
- 대칭행렬은 항상 서로 직교하는(따라서 독립적인) 고유벡터를 n개 가지므로 항상 대각화 가능하다.
""")


# ===================================================
# 문제 2-1 : PCA를 NumPy로 직접 구현하기
# ===================================================
cov = np.cov(Xs, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(-eigvals)
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

top2 = eigvecs_sorted[:, :2]
Z_manual = Xs @ top2

print('--- 문제 2-1 ---')
print('Z_manual shape:', Z_manual.shape)

explained_ratio = eigvals_sorted / eigvals_sorted.sum()
print('상위 2개 설명분산비:', explained_ratio[:2])

corr_axes = np.corrcoef(Z_manual[:, 0], Z_manual[:, 1])[0, 1]
print('두 축 간 상관계수:', corr_axes, '(0에 가까움:', np.isclose(corr_axes, 0, atol=1e-8), ')')

print("""
설명:
- 상위 2개 주성분으로 투영한 두 축은 원래 서로 직교하는 고유벡터 방향에서 나온 것이므로,
  투영된 데이터에서도 두 축 사이의 상관관계는 사실상 0이 된다.
""")


# ===================================================
# 문제 2-2 : sklearn PCA와 결과 비교하기
# ===================================================
pca = PCA(n_components=2)
Z_sklearn = pca.fit_transform(Xs)

print('--- 문제 2-2 ---')
print('sklearn 설명분산비:', pca.explained_variance_ratio_)
print('직접 계산 설명분산비:', explained_ratio[:2])
print('일치 (np.allclose):', np.allclose(pca.explained_variance_ratio_, explained_ratio[:2]))

print('\nsklearn components_ vs 직접 구한 고유벡터 (절댓값 비교):')
print('일치:', np.allclose(np.abs(pca.components_), np.abs(top2.T)))

print('\nZ_manual vs Z_sklearn (절댓값 비교):')
print('일치:', np.allclose(np.abs(Z_manual), np.abs(Z_sklearn)))

pca_full = PCA()
Z_full = pca_full.fit_transform(Xs)

sign_match = []
for i in range(eigvecs_sorted.shape[1]):
    same_sign = np.allclose(pca_full.components_[i], eigvecs_sorted[:, i], atol=1e-6)
    opposite_sign = np.allclose(pca_full.components_[i], -eigvecs_sorted[:, i], atol=1e-6)
    sign_match.append('동일' if same_sign else ('반전' if opposite_sign else '불일치'))

sign_df = pd.DataFrame({'주성분': [f'PC{i+1}' for i in range(len(sign_match))], '부호': sign_match})
print('\n성분별 부호 비교:')
print(sign_df)

plt.figure(figsize=(6, 6))
plt.scatter(Z_sklearn[:, 0], Z_sklearn[:, 1], s=8, alpha=.5)
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA 2D 산점도 (sklearn)')
plt.show()

print("""
설명:
- 고유벡터(v)와 -v는 둘 다 Av=λv를 동일하게 만족하므로 수학적으로 완전히 동등한 고유벡터이다.
  어느 부호를 선택할지는 계산 알고리즘의 내부 구현 방식에 따라 달라질 수 있어, sklearn과 직접
  구현한 결과의 부호가 성분마다 다르게(같기도, 반대이기도) 나올 수 있다.
""")


# ===================================================
# 심화 1 문제 3-1 : 누적 설명분산비로 차원 수 결정하기
# ===================================================
print('--- 문제 3-1 ---')
explained_full = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained_full)

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
plt.axhline(0.9, color='red', linestyle='--', label='90% 기준선')
plt.xlabel('주성분 개수'); plt.ylabel('누적 설명분산비')
plt.title('누적 설명분산비')
plt.legend()
plt.show()

n_80 = int(np.searchsorted(cumulative, 0.80) + 1)
n_90 = int(np.searchsorted(cumulative, 0.90) + 1)
n_95 = int(np.searchsorted(cumulative, 0.95) + 1)

print(f'80% 넘는 최소 주성분 개수: {n_80}')
print(f'90% 넘는 최소 주성분 개수: {n_90}')
print(f'95% 넘는 최소 주성분 개수: {n_95}')

n_features = Xs.shape[1]
reduction_df = pd.DataFrame({
    '기준': ['80%', '90%', '95%'],
    '필요 주성분 수': [n_80, n_90, n_95],
    '원본 특성 수': [n_features] * 3,
    '축소율': [f'{(1 - n / n_features) * 100:.1f}%' for n in [n_80, n_90, n_95]],
})
print('\n', reduction_df)

pca_raw = PCA()
pca_raw.fit(X_df.to_numpy())
print('\n표준화 O, PC1 설명분산비:', explained_full[0])
print('표준화 X, PC1 설명분산비:', pca_raw.explained_variance_ratio_[0])

print("""
설명:
- 기준치를 정할 때는 목적에 따라 달라져야 한다. 시각화가 목적이면 2~3차원으로 강제 축소해도
  되지만, 모델 성능 유지가 목적이면 누적 설명분산비가 충분히 높은(90% 이상 등) 지점을 골라야
  정보 손실을 최소화할 수 있다. 또한 표준화를 하지 않으면 값의 스케일(단위)이 큰 특성이 첫
  주성분을 독점해버리는 왜곡이 생기므로, PCA 전에는 반드시 표준화를 해야 한다.
""")