# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy

import numpy as np
import pandas as pd
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
        print('[안내] 대체 데이터로 진행합니다. 성질 검증 방법은 동일합니다.')
        from sklearn.datasets import make_classification
        Xa, ya = make_classification(n_samples=2500, n_features=20,
                                     n_informative=8, random_state=RANDOM_STATE)
        cols = [f'feature_{i}' for i in range(Xa.shape[1])]
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(350)   # Default of Credit Card Clients
X = StandardScaler().fit_transform(numeric_frame(X_raw).iloc[:, :4])   # 특성 4개만 사용
print('데이터 행렬 X:', X.shape)


# ===================================================
# 문제 1-1 : 단위행렬과 대각행렬 적용하기
# ===================================================
A = X[:50]   # 앞 50행
I = np.eye(A.shape[1])   # A의 열 수에 맞는 단위행렬

print('\n--- 문제 1-1 ---')
print('A shape:', A.shape, '/ I shape:', I.shape)
print('A @ I == A 확인 (np.allclose):', np.allclose(A @ I, A))

D = np.diag([1, 10, 0.1, 2])   # 항목별 비중 조정용 대각행렬
AD = A @ D

std_before = A.std(axis=0)
std_after = AD.std(axis=0)

print('\nD (대각행렬):\n', D)
print('A의 각 열 표준편차   :', std_before)
print('A@D의 각 열 표준편차 :', std_after)
print('표준편차 배율(after/before):', std_after / std_before)

print("""
설명:
- A @ I는 A와 완전히 동일하다. 단위행렬은 스칼라의 1처럼 "곱해도 아무것도 바꾸지 않는" 기준점이다.
- 대각행렬 D를 곱하면 각 열(축)의 표준편차가 D의 대각 원소 배율만큼 정확히 커지거나 작아진다.
  대각행렬은 축마다 독립적으로 크기(스케일)만 조정할 뿐, 축 사이를 섞거나 회전시키지 않는다.
""")


# ===================================================
# 문제 1-2 : 데이터에서 대칭행렬 찾기
# ===================================================
G = X.T @ X

print('--- 문제 1-2 ---')
print('G = X.T @ X shape:', G.shape)
print('G == G.T 확인 (np.allclose):', np.allclose(G, G.T))

cov = np.cov(X, rowvar=False)
print('\n공분산행렬 shape:', cov.shape)
print('공분산행렬 대칭 확인 (np.allclose):', np.allclose(cov, cov.T))

print("""
설명:
- 변수 i와 j의 공분산 Cov(i,j)는 "i의 편차 x j의 편차"의 평균으로 계산되는데, 곱셈은 순서를
  바꿔도 값이 같으므로(i의 편차 x j의 편차 = j의 편차 x i의 편차) Cov(i,j) = Cov(j,i)가 항상
  성립한다. 그래서 공분산행렬은 어떤 데이터로 만들든 항상 대칭이다.
""")


# ===================================================
# 문제 2-1 : AB ≠ BA 확인하기
# ===================================================
P = np.array([[1, 2], [0, 1]])
Q = np.array([[1, 0], [3, 1]])

PQ = P @ Q
QP = Q @ P

print('--- 문제 2-1 ---')
print('P @ Q =\n', PQ)
print('Q @ P =\n', QP)
print('P@Q == Q@P (np.array_equal):', np.array_equal(PQ, QP))

P_scaled_I = 2 * np.eye(2)   # 단위행렬의 스칼라배
print('\nP를 2*I로 바꾸면:')
print('(2I) @ Q =\n', P_scaled_I @ Q)
print('Q @ (2I) =\n', Q @ P_scaled_I)
print('(2I)@Q == Q@(2I):', np.array_equal(P_scaled_I @ Q, Q @ P_scaled_I))

D2 = np.diag([2, 5])   # 대각 원소가 서로 다른 대각행렬
print('\n대각 원소가 다른 대각행렬 diag([2,5])로 비교:')
print('D2 @ Q =\n', D2 @ Q)
print('Q @ D2 =\n', Q @ D2)
print('D2@Q == Q@D2:', np.array_equal(D2 @ Q, Q @ D2))

print("""
설명:
- 행렬곱 AB의 (i,j) 원소는 A의 i번째 행과 B의 j번째 열의 내적이다. 순서를 바꿔 BA를 계산하면
  서로 다른 행/열 조합끼리 내적하게 되므로 일반적으로 값이 달라진다(교환법칙 불성립).
- 단위행렬의 스칼라배(2I)는 모든 위치에 같은 배율만 곱하므로 예외적으로 어떤 행렬과 곱해도
  순서가 무관하다. 하지만 대각 원소가 서로 다른 일반 대각행렬(diag([2,5]))은 이 예외에 해당하지
  않아 순서를 바꾸면 결과가 달라진다. 즉 "대각행렬이면 항상 교환된다"는 결론은 성립하지 않는다.
""")


# ===================================================
# 문제 2-2 : 전치 성질로 XᵀX가 대칭인 이유 설명하기
# ===================================================
lhs = (P @ Q).T
rhs = Q.T @ P.T
wrong_order = P.T @ Q.T

print('--- 문제 2-2 ---')
print('(P@Q).T =\n', lhs)
print('Q.T@P.T =\n', rhs)
print('(P@Q).T == Q.T@P.T (np.array_equal):', np.array_equal(lhs, rhs))

print('\nP.T@Q.T =\n', wrong_order)
print('(P@Q).T == P.T@Q.T (순서 유지, 값이 달라야 정상):', np.array_equal(lhs, wrong_order))

XtX = X.T @ X
XtX_transposed = (X.T @ X).T
print('\n(XᵀX)ᵀ == XᵀX (np.allclose):', np.allclose(XtX, XtX_transposed))

print("""
설명:
- (AB)ᵀ = BᵀAᵀ 이므로 (XᵀX)ᵀ = Xᵀ(Xᵀ)ᵀ = XᵀX 가 되어, XᵀX는 데이터가 무엇이든 항상 자기 자신의
  전치와 같은 대칭행렬이 된다.
- 공분산행렬은 평균을 뺀 데이터에 대한 XᵀX 형태로 계산되므로, 이 전치 성질 때문에 공분산행렬도
  항상 대칭일 수밖에 없다.
""")


# ===================================================
# 문제 3-1 : 행렬식과 조건수로 역행렬 존재·안정성 확인하기
# ===================================================
A3 = np.array([[1, 2], [3, 4]])
det_A3 = np.linalg.det(A3)
A3_inv = np.linalg.inv(A3)

print('--- 문제 3-1 ---')
print('A =\n', A3)
print('A 행렬식:', det_A3)
print('A @ A_inv ≈ I (np.allclose):', np.allclose(A3 @ A3_inv, np.eye(2)))

S = np.array([[1, 2], [2, 4]])   # 두 번째 행이 첫 행의 2배 (특이행렬)
det_S = np.linalg.det(S)
print('\nS =\n', S)
print('S 행렬식:', det_S)
try:
    np.linalg.inv(S)
except np.linalg.LinAlgError as e:
    print('S 역행렬 계산 오류:', e)

X_dup = np.hstack([X, X[:, [0]] * 2])   # 첫 번째 컬럼의 2배인 중복 컬럼 추가
print('\nX_dup shape:', X_dup.shape)

XtX_orig = X.T @ X
XtX_dup = X_dup.T @ X_dup

det_orig = np.linalg.det(XtX_orig)
det_dup = np.linalg.det(XtX_dup)
cond_orig = np.linalg.cond(XtX_orig)
cond_dup = np.linalg.cond(XtX_dup)

print(f'\nXᵀX        행렬식: {det_orig:.6e} / 조건수: {cond_orig:.2f}')
print(f'X_dupᵀX_dup 행렬식: {det_dup:.6e} / 조건수: {cond_dup:.2f}')

print("""
설명:
- 행렬식이 0에 가까워지거나 조건수가 급격히 커지면, 역행렬 계산이 수치적으로 불안정해져서
  계수 값이 비정상적으로 커지거나 실행할 때마다 결과가 조금씩 달라질 수 있다.
- 실무에서는 중복이거나 거의 선형종속인 컬럼(예: 월 소득과 연 소득)을 제거하거나, 정규화(릿지
  회귀처럼 대각선에 작은 값을 더하는 방식) 또는 PCA로 차원을 줄여 대응해야 한다.
- 행렬식의 절댓값 크기 자체는 판단 기준이 아니다(데이터 스케일에 따라 커지거나 작아질 수 있음).
  대신 조건수(가장 큰 특이값/가장 작은 특이값의 비율)로 안정성을 판단해야 한다.
""")