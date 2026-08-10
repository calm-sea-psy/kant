# 최초 1회만 실행 (새 환경일 때)
# !pip -q install ucimlrepo pandas numpy

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_retail():
    """Online Retail 거래 데이터를 불러오고, 실패하면 같은 구조의 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=352)
        return ds.data.original.copy()
    except Exception as e:
        print('[안내] UCI 로드 실패:', e)
        print('[안내] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다.')
        rng = np.random.default_rng(RANDOM_STATE)
        # 고객마다 구매 건수를 2~12건으로 다르게 배정합니다.
        # 모든 고객이 같은 건수를 갖게 되면 '길이가 다른 시퀀스'라는 실습 전제가 무너집니다.
        customer_ids = []
        for cid in range(1000, 1120):
            customer_ids += [cid] * int(rng.integers(2, 13))
        n = len(customer_ids)
        return pd.DataFrame({
            'InvoiceNo': rng.integers(10000, 10800, n).astype(str),
            'StockCode': rng.choice([f'P{i:03d}' for i in range(120)], n),
            'Quantity': rng.poisson(3, n) + 1,
            'UnitPrice': rng.gamma(2.0, 10.0, n),
            'CustomerID': customer_ids,
        })


retail = load_retail()
df = retail.dropna(subset=['CustomerID', 'StockCode']).copy()
df['StockCode'] = df['StockCode'].astype(str)

# 고객별 구매 상품 시퀀스 (최대 10개)
all_seqs = df.groupby('CustomerID')['StockCode'].apply(lambda s: list(s)[:10])

# padding 효과를 보려면 길이가 서로 달라야 하므로, 길이가 다른 고객 8명을 고릅니다
lengths_all = all_seqs.apply(len)
pick = lengths_all.sort_values().drop_duplicates().index[:8]
seqs = all_seqs.loc[pick]

print('고객 수:', len(seqs))
print('시퀀스 길이:', [len(s) for s in seqs])
print('가장 긴 시퀀스 길이:', max(len(s) for s in seqs))   # 뒤에서 max_len 으로 사용합니다