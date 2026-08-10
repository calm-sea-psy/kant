# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo scikit-learn pandas numpy

import numpy as np
import pandas as pd

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
        print('[안내] 대체 센서 데이터로 진행합니다. shape 규칙은 동일합니다.')
        from sklearn.datasets import make_regression
        Xa, ya = make_regression(n_samples=2000, n_features=5, noise=10,
                                 random_state=RANDOM_STATE)
        cols = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear']
        return pd.DataFrame(Xa, columns=cols), pd.Series(ya)


def numeric_frame(X):
    """수치형 컬럼만 남기고 결측값을 중앙값으로 채웁니다."""
    Xn = X.select_dtypes(include='number').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn.fillna(Xn.median(numeric_only=True))


X_raw, y_raw = load_uci(601)
X_df = numeric_frame(X_raw).iloc[:, :5]     # 센서 5개
A_small = X_df.iloc[:24].values             # 24 = 4 x 6 으로 나누어떨어짐
print('센서 데이터:', A_small.shape, '/ 전체 원소 수:', A_small.size)