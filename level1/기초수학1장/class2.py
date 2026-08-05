# 최초 1회만 실행 (새 환경일 때)
# Colab에는 ucimlrepo가 기본 설치되어 있지 않으므로 이 셀을 건너뛰지 마세요.
#!pip install -q ucimlrepo

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_retail():
    """Online Retail 거래 데이터를 불러오고, 실패하면 같은 구조의 대체 데이터를 사용합니다."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print('[경고] ucimlrepo 패키지가 설치되어 있지 않습니다. 위의 pip 설치 셀을 먼저 실행하세요.')
        print('[경고] 지금은 실제 데이터가 아닌 대체(임의 생성) 거래 데이터로 진행됩니다.')
        return _dummy_retail()

    try:
        ds = fetch_ucirepo(id=352)
        return ds.data.original.copy()
    except Exception as e:
        print('[경고] UCI 서버 접속 실패(네트워크·방화벽 확인 필요):', e)
        print('[경고] 동일한 컬럼 구조의 대체 거래 데이터로 진행합니다. 출력값은 교안 예시와 다릅니다.')
        return _dummy_retail()


def _dummy_retail():
    """UCI를 쓸 수 없을 때 사용하는 같은 컬럼 구조의 대체 거래 데이터입니다."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 6000
    return pd.DataFrame({
        'InvoiceNo': rng.integers(10000, 10800, n).astype(str),
        'StockCode': rng.choice([f'P{i:03d}' for i in range(300)], n),
        'Quantity': rng.poisson(3, n) + 1,
        'UnitPrice': rng.gamma(2.0, 10.0, n),
        'CustomerID': rng.integers(1000, 1120, n),
    })


def customer_product_matrix(df, n_customers=60, n_products=150):
    """고객(행) x 상품(열) 구매량 행렬을 만듭니다."""
    df = df.dropna(subset=['CustomerID', 'StockCode']).copy()  # 고객 미상 거래 제외
    pivot = df.pivot_table(index='CustomerID', columns='StockCode',
                           values='Quantity', aggfunc='sum', fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(n_customers).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).head(n_products).index]
    return pivot.astype(float)


retail = load_retail()
M_df = customer_product_matrix(retail, 60, 150)
M = M_df.values
print('고객-상품 행렬:', M.shape)


# ===================================================
# 문제 1-1 : 고객 벡터의 내적 직접 계산하기
# ===================================================
a = M_df.iloc[0].to_numpy()
b = M_df.iloc[1].to_numpy()

print('\n--- 문제 1-1 ---')
print('a shape:', a.shape, '/ b shape:', b.shape)

dot_manual = np.sum(a * b)
dot_np = np.dot(a, b)
dot_at = a @ b

print('np.sum(a*b) :', dot_manual)
print('np.dot(a, b):', dot_np)
print('a @ b       :', dot_at)
print('세 값 모두 일치:', np.isclose(dot_manual, dot_np) and np.isclose(dot_manual, dot_at))
print('결과 타입:', type(dot_manual))

print("""
설명:
- 내적 결과가 스칼라 하나로 나오기 때문에, 두 고객의 구매 벡터를 원소별로 뜯어보지 않고도
  "겹치는 구매 정도"를 하나의 숫자로 요약해 바로 정렬·비교·순위화할 수 있어 유용하다.
""")


# ===================================================
# 문제 1-2 : 내적 값이 구매 규모에 영향을 받는지 확인하기
# ===================================================
dots = M @ a  # dots[i] = M_df.iloc[i] · a
order = np.argsort(-dots)  # 내적 내림차순 위치(position) 목록

print('\n--- 문제 1-2 ---')
print('1위 위치:', order[0], '/ CustomerID:', M_df.index[order[0]], '/ 내적:', dots[order[0]])
print("""
설명:
- 1위가 항상 자기 자신인 이유는 a·a = ‖a‖² 이기 때문이다. 코시-슈바르츠 부등식에 의해
  a·x <= ‖a‖‖x‖ <= ‖a‖‖a‖ = a·a 가 항상 성립하므로, 자기 자신과의 내적을 능가하는 벡터는 없다.
  따라서 순위를 볼 때는 1위(자기 자신)를 제외하고 봐야 의미가 있다.
""")

total_purchase = M_df.sum(axis=1)  # 고객별 총 구매량 (행 합계)

top5_positions = order[order != 0][:5]
top5_dot_df = pd.DataFrame({
    'CustomerID': M_df.index[top5_positions],
    '내적': dots[top5_positions],
    '총구매량': total_purchase.iloc[top5_positions].to_numpy(),
})
print('\n자기 자신 제외 내적 상위 5명:')
print(top5_dot_df)

print('\n전체 고객 총구매량 평균   :', total_purchase.mean())
print('내적 상위 5명 총구매량 평균:', top5_dot_df['총구매량'].mean())

print("""
설명:
- 내적 상위권 고객은 취향이 비슷해서가 아니라 단순히 구매 규모(총 구매량) 자체가 커서
  내적 값이 크게 나올 수 있다는 문제가 있다. 즉 내적만으로는 "성향"과 "규모"를 구분하지 못한다.
""")


# ===================================================
# 문제 2-1 : 코사인 유사도를 공식으로 직접 구현하기
# ===================================================
def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print('\n--- 문제 2-1 ---')
cos_ab = cosine(a, b)
print('0번-1번 고객 코사인 유사도:', cos_ab)

cos_self = cosine(a, a)
cos_scaled = cosine(a, a * 3)

# 서로 겹치는 상품이 없는(내적이 0인) 두 벡터를 인위적으로 구성
nonzero_idx = np.nonzero(a)[0]
c = np.zeros_like(a)
d = np.zeros_like(a)
c[nonzero_idx[0]] = 1.0
d[nonzero_idx[1]] = 1.0
cos_disjoint = cosine(c, d)

print('자기 자신과의 유사도        :', cos_self)
print('3배 스칼라곱 벡터와의 유사도 :', cos_scaled)
print('겹치는 상품이 없는 두 벡터  :', cos_disjoint)

print("""
설명:
- 코사인 유사도 1  : 두 벡터가 완전히 같은 방향 -> 구매 성향이 동일함
- 코사인 유사도 0  : 두 벡터가 겹치는 성분이 전혀 없어 직각 -> 성향이 서로 무관함
- 코사인 유사도 -1 : 완전히 반대 방향 -> 정반대 성향 (구매량처럼 음수가 없는 데이터에서는 나타나기 어려움)
""")


# ===================================================
# 문제 2-2 : 유사 고객 상위 5명 찾기
# ===================================================
print('\n--- 문제 2-2 ---')
sim_matrix = cosine_similarity(M)
print('유사도 행렬 shape:', sim_matrix.shape)

sim_df = pd.DataFrame(sim_matrix, index=M_df.index, columns=M_df.index)

sim_to_0 = sim_df.iloc[0].drop(M_df.index[0]).sort_values(ascending=False)
top5_cos = sim_to_0.head(5)

print('\n0번 고객 기준 코사인 유사도 상위 5명:')
print(top5_cos)

print('\n(비교) 내적 기준 상위 5명 CustomerID :', top5_dot_df['CustomerID'].tolist())
print('       코사인 기준 상위 5명 CustomerID:', top5_cos.index.tolist())

print("""
설명:
- 내적 기준 상위 5명과 코사인 기준 상위 5명의 목록이 달라진다. 내적은 구매 규모(양)에 영향을
  받는 반면, 코사인 유사도는 크기(노름)로 나누어 규모를 제거했기 때문에 순수하게 "구매 성향
  (상품 구성 비율)"만 비교한다.
- 따라서 총구매량이 서로 크게 다른 고객들을 비교할 때는 코사인 유사도가 더 적합하다.
""")


# ===================================================
# 문제 3-1 : 유사 고객 기반 추천 후보 도출하기
# ===================================================
print('\n--- 문제 3-1 ---')
nearest_customer_id = top5_cos.index[0]
neighbor_vec = M_df.loc[nearest_customer_id].copy()
base_vec = M_df.iloc[0]

recommend_scores = neighbor_vec.copy()
recommend_scores[base_vec.to_numpy() > 0] = 0  # 기준 고객이 이미 구매한 상품 제외

candidates = recommend_scores[recommend_scores > 0].sort_values(ascending=False)
print('가장 유사한 이웃 고객:', nearest_customer_id)
print('추천 후보 개수:', len(candidates))
print('\n추천 후보 상위', min(10, len(candidates)), '개:')
print(candidates.head(10))

print("""
설명:
- 이 방식은 "질문 벡터와 가장 유사한 문서(임베딩)를 찾아 그 문서의 정보를 활용하는" 임베딩
  기반 유사도 검색과 원리가 같다. 기준 벡터와 방향이 가장 비슷한 이웃 벡터를 찾은 뒤, 그
  이웃이 가진 정보(여기서는 구매 상품)를 기준 대상에게 재사용/추천한다는 점에서 동일하다.
""")