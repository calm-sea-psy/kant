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
        print('[안내] 대체 데이터로 진행합니다. shape 규칙과 해석은 동일합니다.')
        from sklearn.datasets import make_classification
        Xa, ya = make_classification(n_samples=3000, n_features=12,
                                     n_informative=6, random_state=RANDOM_STATE)
        cols = [f'feature_{i}' for i in range(Xa.shape[1])]
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(222)   # Bank Marketing
Xn = numeric_frame(X_raw)

FEATURE_COLS = ['age', 'balance', 'duration', 'pdays', 'previous']   # 값이 실제로 흩어져 있는(분산 있는) 컬럼만 명시적으로 선택
FEATURE_COLS = [c for c in FEATURE_COLS if c in Xn.columns] or list(Xn.columns[:5])

X_small = Xn[FEATURE_COLS].sample(n=8, random_state=RANDOM_STATE)   # 앞 8행 대신 무작위 8명 샘플링
X = StandardScaler().fit_transform(X_small)      # 스케일을 맞춰 값 비교를 쉽게
print('데이터 행렬 X:', X.shape)


# ===================================================
# 문제 1-1 : 데이터 행렬 X와 완전연결층 shape 확인하기
# ===================================================
n_samples, n_features = X.shape
n_out = 3   # 출력 점수를 3개 만들고 싶으므로 출력 차원 = 3

# W: (입력 특성 수, 출력 개수) -> X @ W가 성립하려면 W의 행 수가 X의 열 수(n_features)와 같아야 함
# b: (출력 개수,) -> X @ W의 각 행(샘플)에 동일하게 더해지도록(브로드캐스팅) 출력 개수와 같은 길이
W = np.random.randn(n_features, n_out)
b = np.random.randn(n_out)

Y = X @ W + b

print('\n--- 문제 1-1 ---')
print('X shape:', X.shape, '(행=고객 수, 열=특성 수)')
print('W shape:', W.shape, '(행=입력 특성 수와 일치, 열=출력 점수 개수)')
print('b shape:', b.shape, '(출력 점수 개수와 일치)')
print('Y shape:', Y.shape, '(행=고객 수, 열=출력 점수 개수)')

print("""
설명:
- X @ W의 shape은 (샘플 수, 입력 특성 수) @ (입력 특성 수, 출력 개수) = (샘플 수, 출력 개수)로 결정된다.
  안쪽 차원(입력 특성 수)이 서로 일치해야 곱셈이 성립하고, 바깥쪽 차원(샘플 수, 출력 개수)만 결과에 남는다.
- b는 (출력 개수,) 벡터이지만 (샘플 수, 출력 개수) 행렬에 브로드캐스팅되어 모든 샘플에 동일하게 더해진다.
""")


# ===================================================
# 문제 1-2 : 반복문 계산과 행렬곱 결과가 같은지 확인하기
# ===================================================
Y_loop = np.array([x @ W + b for x in X])

print('--- 문제 1-2 ---')
print('반복문 결과 shape:', Y_loop.shape, '/ 행렬곱 결과 shape:', Y.shape)
print('shape 일치:', Y_loop.shape == Y.shape)
print('값 일치 (np.allclose):', np.allclose(Y_loop, Y))

print("""
설명:
- 반복문은 고객 한 명씩 8번 계산을 반복하지만, 행렬곱은 동일한 결과를 한 번의 연산으로 계산한다.
  샘플 수가 커질수록 행렬곱은 내부적으로 최적화된 선형대수 연산(BLAS)을 사용해 반복문보다 훨씬 빠르다.
""")


# ===================================================
# 필수 2 문제 2-1 : shape 오류를 진단하고 전치로 해결하기
# ===================================================
W_wrong = np.random.randn(4, 3)   # 입력 특성 수(5)와 맞지 않는 행 수(4)

print('--- 문제 2-1 ---')
print('X.shape       :', X.shape)
print('W_wrong.shape :', W_wrong.shape)

try:
    X @ W_wrong
except ValueError as e:
    print('오류 발생:', e)

print("""
설명:
- X.shape=(8, 5), W_wrong.shape=(4, 3) 이므로 X의 열 수(5)와 W_wrong의 행 수(4)가 일치하지 않아 오류가 난다.
""")

W_fixed = np.random.randn(X.shape[1], 3)   # 행 수를 X의 열 수(5)에 맞춤
Y_fixed = X @ W_fixed + b
print('W_fixed.shape :', W_fixed.shape)
print('Y_fixed.shape :', Y_fixed.shape, '(정상 계산됨)')

print("""
설명:
- 행렬곱 A @ B가 가능하려면 A의 열 수와 B의 행 수가 같아야 하며, 그때 결과 shape은 (A의 행 수, B의 열 수)가 된다.
""")


# ===================================================
# 문제 2-2 : 전치로 shape 맞추고 전치 성질 검증하기
# ===================================================
print('--- 문제 2-2 ---')
print('X.shape  :', X.shape)
print('X.T.shape:', X.T.shape, '(행과 열이 서로 바뀜)')

lhs = (X @ W).T
rhs = W.T @ X.T
print('(X @ W).T shape:', lhs.shape, '/ W.T @ X.T shape:', rhs.shape)
print('값 일치 (np.allclose):', np.allclose(lhs, rhs))

print('\nX.T @ W.T 시도:')
try:
    X.T @ W.T
except ValueError as e:
    print('오류 발생:', e)
    print(f'(X.T.shape={X.T.shape}의 열 수와 W.T.shape={W.T.shape}의 행 수가 일치하지 않기 때문)')

print("""
설명:
- (AB)^T = B^T A^T 이므로, 전체를 전치하면 곱하는 두 행렬의 순서도 함께 뒤바뀌어야 안쪽 차원이 다시 맞는다.
  그래서 (X @ W).T는 W.T @ X.T와 같지만, 순서를 바꾸지 않은 X.T @ W.T는 차원이 맞지 않아 계산되지 않는다.
""")


# ===================================================
# 심화 1 문제 3-1 : 2층 구조의 shape 흐름과 파라미터 수 계산하기
# ===================================================
def build_layer_params(in_dim, hidden_dim, out_dim):
    W1 = np.random.randn(in_dim, hidden_dim)
    b1 = np.random.randn(hidden_dim)
    W2 = np.random.randn(hidden_dim, out_dim)
    b2 = np.random.randn(out_dim)
    return W1, b1, W2, b2


def run_two_layers(X, W1, b1, W2, b2):
    H = X @ W1 + b1
    O = H @ W2 + b2
    return H, O


def report(tag, X, W1, b1, W2, b2):
    H, O = run_two_layers(X, W1, b1, W2, b2)
    p1 = W1.size + b1.size
    p2 = W2.size + b2.size
    print(f'[{tag}] X{X.shape} -> H{H.shape} -> O{O.shape}')
    print(f'[{tag}] 1층 파라미터 수: {p1} (W1={W1.size} + b1={b1.size})')
    print(f'[{tag}] 2층 파라미터 수: {p2} (W2={W2.size} + b2={b2.size})')
    print(f'[{tag}] 전체 파라미터 수: {p1 + p2}')
    return H, O


print('--- 문제 3-1 ---')
in_dim, hidden_dim, out_dim = X.shape[1], 4, 2
W1, b1, W2, b2 = build_layer_params(in_dim, hidden_dim, out_dim)
report('은닉 4', X, W1, b1, W2, b2)

print("""
예측: 은닉 차원을 4 -> 16으로 늘리면 O의 shape(샘플 수, 출력 차원)은 변하지 않는다.
      하지만 W1, W2가 은닉 차원을 포함하므로 파라미터 수는 크게 늘어난다.
""")

hidden_dim2 = 16
W1b, b1b, W2b, b2b = build_layer_params(in_dim, hidden_dim2, out_dim)
report('은닉 16', X, W1b, b1b, W2b, b2b)