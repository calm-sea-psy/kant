import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# 같은 seed를 고정하면 분할 차이와 모델 차이를 섞지 않고 결과를 다시 확인할 수 있습니다.
# 아래 검사는 원본 행 인덱스를 사용하므로 분할 뒤 행 순서가 달라도 누수를 찾아냅니다.

def assert_disjoint(*frames):
    """여러 분할의 원본 인덱스가 서로 겹치지 않는지 확인합니다."""
    index_sets = [set(frame.index) for frame in frames]
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            assert index_sets[i].isdisjoint(index_sets[j])


# 두 자료는 scikit-learn 내장 실제 데이터이므로 외부 다운로드나 합성 fallback이 없습니다.
# 회귀 데이터는 target 통계 계산 전에 먼저 세 분할로 나눕니다.
# 첫 분할의 40%를 다시 반으로 나누므로 train/validation/test 비율은 60/20/20입니다.
diabetes = load_diabetes(as_frame=True)
X_reg, y_reg = diabetes.data, diabetes.target
X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=SEED
)
X_reg_valid, X_reg_test, y_reg_valid, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.5, random_state=SEED
)
assert_disjoint(X_reg_train, X_reg_valid, X_reg_test)

# scikit-learn 원본의 malignant=0을 탐지 대상인 positive class 1로 바꿉니다.
# 분류에서는 stratify를 사용해 세 분할의 악성 비율이 크게 흔들리지 않게 합니다.
breast = load_breast_cancer(as_frame=True)
X_cls = breast.data
y_cls = (breast.target == 0).astype(int)
X_cls_train, X_cls_temp, y_cls_train, y_cls_temp = train_test_split(
    X_cls, y_cls, test_size=0.4, stratify=y_cls, random_state=SEED
)
X_cls_valid, X_cls_test, y_cls_valid, y_cls_test = train_test_split(
    X_cls_temp,
    y_cls_temp,
    test_size=0.5,
    stratify=y_cls_temp,
    random_state=SEED,
)
assert_disjoint(X_cls_train, X_cls_valid, X_cls_test)

# 행 수와 양성 비율은 분할·target 변환이 의도대로 되었는지 확인하는 첫 점검값입니다.
print("Diabetes split:", len(X_reg_train), len(X_reg_valid), len(X_reg_test))
print("Cancer split:", len(X_cls_train), len(X_cls_valid), len(X_cls_test))
print("Cancer positive rate:", round(float(y_cls.mean()), 4))


def build_regression_report(X_train, y_train, X_valid, y_valid):
    """회귀 모델과 train 평균 기준 모델의 validation 지표를 반환합니다."""
    # 함수에 들어온 두 분할이 실제로 겹치지 않는지 재확인합니다.
    # 상단에서 이미 검증했지만, 이 함수만 따로 재사용/테스트될 수도 있으므로 다시 확인합니다.
    assert_disjoint(X_train, X_valid)

    # StandardScaler로 각 특성을 평균 0, 표준편차 1로 맞춘 뒤 LinearRegression을 학습합니다.
    # 두 단계를 Pipeline으로 묶으면 스케일러 통계(fit)가 항상 train에서만 계산되고,
    # validation/test에는 transform만 적용되어 데이터 누수를 구조적으로 막을 수 있습니다.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    # fit은 반드시 train에만 적용합니다. validation은 아직 모델이 한 번도 보지 않은 상태여야 합니다.
    model.fit(X_train, y_train)
    # 학습된 모델로 validation 입력에 대한 예측값을 만듭니다. 여기서 처음으로 validation을 사용합니다.
    model_pred = model.predict(X_valid)

    # 기준 모델은 "가장 단순한 모델도 못 이기면 의미가 없다"를 확인하기 위한 하한선입니다.
    # 기준 모델은 validation을 보지 않고 train target 평균만 그대로 반복합니다.
    baseline_value = y_train.mean()
    # model_pred와 같은 길이·shape의 배열을 만들어, 모든 validation 샘플에 대해
    # 동일한 baseline_value를 예측한 것처럼 취급합니다.
    baseline_pred = np.full_like(model_pred, fill_value=baseline_value, dtype=float)

    def metrics(y_true, y_pred):
        # MAE: 오차 절댓값의 평균. 이상치 영향이 상대적으로 작고 단위가 target과 동일합니다.
        mae = mean_absolute_error(y_true, y_pred)
        # RMSE: 오차 제곱의 평균에 제곱근을 씌운 값. 큰 오차에 더 큰 페널티를 줍니다.
        # mean_squared_error는 기본적으로 MSE(제곱값)를 반환하므로 np.sqrt로 직접 RMSE를 만듭니다.
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # R²: 기준(평균)만 예측했을 때 대비 분산을 얼마나 설명하는지를 나타내는 비율.
        # 1에 가까울수록 좋고, 0이면 평균 예측과 다를 바 없으며 음수면 평균보다도 못합니다.
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}

    # 같은 validation target(y_valid)과 같은 지표 함수를 사용해 모델과 기준 모델을 "공정하게" 비교합니다.
    model_metrics = metrics(y_valid, model_pred)
    baseline_metrics = metrics(y_valid, baseline_pred)

    # 모델이 기준 모델보다 나은지 최소한의 조건(RMSE 개선)을 assertion으로 강제합니다.
    # 이 조건이 깨지면 "모델이 평균만 예측하는 것보다도 못하다"는 뜻이므로 바로 실패시켜야 합니다.
    assert model_metrics["rmse"] < baseline_metrics["rmse"]

    # 학습된 파이프라인(model)까지 함께 반환해, 호출부에서 재학습 없이 재사용할 수 있게 합니다.
    return {
        "model": model,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_value": baseline_value,
    }


def build_classification_report(X_train, y_train, X_valid, y_valid):
    """분류 Pipeline과 validation 확률·지표를 반환합니다."""
    # 회귀 함수와 마찬가지로, 이 함수에 들어온 두 분할이 겹치지 않는지 다시 확인합니다.
    assert_disjoint(X_train, X_valid)

    # StandardScaler로 30개 특성의 스케일을 맞춘 뒤 LogisticRegression으로 분류합니다.
    # LogisticRegression은 스케일에 민감하므로(경사하강 수렴 속도·정규화 강도에 영향) 스케일링이 특히 중요합니다.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        # 특성 30개 기준으로 기본 max_iter(100)에서는 수렴 경고가 날 수 있어 여유 있게 늘렸습니다.
        ("classifier", LogisticRegression(max_iter=10_000)),
    ])
    # 학습은 train에만 적용합니다.
    clf.fit(X_train, y_train)

    # predict_proba는 [클래스0 확률, 클래스1 확률] 두 열을 반환합니다.
    # y=1(악성)을 positive class로 정의했으므로, 우리가 필요한 "악성일 확률"은 두 번째 열([:, 1])입니다.
    probability = clf.predict_proba(X_valid)[:, 1]
    # 문제에서 지정한 고정 임계값 0.5로 확률을 0/1 예측으로 변환합니다.
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        # Accuracy: 전체 중 맞춘 비율. 클래스 비율이 불균형하면(여기선 악성 37%) 오해를 줄 수 있어 단독 사용은 위험합니다.
        "accuracy": accuracy_score(y_valid, prediction),
        # Precision: 악성이라고 예측한 것 중 실제 악성 비율. 오탐(양성을 악성으로 잘못 판단)을 줄이는 정도를 봅니다.
        "precision": precision_score(y_valid, prediction),
        # Recall: 실제 악성 중 놓치지 않고 잡아낸 비율. 병원 시나리오에서는 이 값을 낮추면 안 됩니다(악성을 놓치는 비용이 큼).
        "recall": recall_score(y_valid, prediction),
        # F1: Precision과 Recall의 조화평균. 두 값을 동시에 요약해서 볼 때 사용합니다.
        "f1": f1_score(y_valid, prediction),
        # ROC-AUC: 임계값을 바꿔가며 본 순위 성능. 확률 자체(probability)를 사용하므로 0.5 고정 임계값에 좌우되지 않습니다.
        "roc_auc": roc_auc_score(y_valid, probability),
        # AP(Average Precision): PR 곡선 아래 면적에 대응. 양성이 소수 클래스일 때 ROC-AUC보다 변별력이 좋을 수 있습니다.
        "average_precision": average_precision_score(y_valid, probability),
    }
    # 여섯 지표 모두 정의상 0~1 사이여야 하므로, 계산 실수(예: label 반전, 잘못된 확률 열 사용)를 조기에 잡기 위한 안전장치입니다.
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

    # fitted clf와 validation 확률을 함께 반환해, 이후 choose_threshold 단계에서
    # 같은 모델·같은 확률을 재사용하고 test 단계에서도 재학습 없이 이 clf를 그대로 쓸 수 있게 합니다.
    return {
        "model": clf,
        "probability": probability,
        "metrics": metrics,
    }


def choose_threshold(y_true, probability, minimum_recall=0.90):
    """Recall 정책을 만족하는 validation 임계값과 비교표를 반환합니다."""
    # probability는 반드시 build_classification_report에서 나온 validation 확률이어야 합니다.
    # test 확률을 여기 넣으면 test를 임계값 선택(모델 결정)에 사용하는 것이 되어 데이터 누수가 됩니다.

    # 0.05 ~ 0.95 구간을 0.01 간격으로 촘촘히 훑습니다.
    # np.arange(0.05, 0.95, 0.01)만 쓰면 부동소수점 오차로 마지막 값(0.95)이 빠질 수 있어 1e-9를 더해 포함시키고,
    # round(2)로 0.860000000001 같은 부동소수점 잔여 오차를 깔끔한 소수 둘째 자리로 정리합니다.
    thresholds = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)

    rows = []
    for threshold in thresholds:
        # 확률이 임계값 이상이면 악성(1)으로, 미만이면 양성(0)으로 판정합니다.
        # 임계값을 낮출수록 더 많은 샘플을 악성으로 분류하므로 일반적으로 Recall은 오르고 Precision은 내려갑니다.
        prediction = (probability >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            # zero_division=0: 특정 임계값에서 양성 예측이 하나도 없거나 실제 양성이 없을 때
            # 0으로 나누는 경고/예외 대신 0.0을 반환하도록 해 표 생성이 중간에 끊기지 않게 합니다.
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
        })
    # 임계값별 지표를 표(DataFrame)로 모아, 필요하면 전체 트레이드오프를 그대로 확인할 수 있게 합니다.
    table = pd.DataFrame(rows)

    # 운영 정책(Recall >= minimum_recall)을 만족하지 못하는 임계값은 후보에서 제외합니다.
    candidates = table[table["recall"] >= minimum_recall]
    if candidates.empty:
        # 정책을 만족하는 임계값이 하나도 없다면, 기준을 몰래 낮추지 말고 실패를 명시적으로 알립니다.
        # 여기서 임계값을 억지로 골라 반환하면 운영 정책을 어기는 모델을 그대로 배포하게 될 위험이 있습니다.
        raise RuntimeError(
            f"validation Recall >= {minimum_recall} 정책을 만족하는 임계값이 없습니다."
        )

    # 정책을 만족하는 후보들 중에서 다음 우선순위로 하나만 고릅니다.
    # 1) F1이 가장 높은 임계값(Precision·Recall 균형이 가장 좋은 지점)
    # 2) F1이 동률이면 Precision이 더 높은 쪽(같은 재현율에서 오탐이 더 적은 쪽)
    # 3) 그래도 동률이면 임계값이 더 낮은 쪽을 선택해 재현율 쪽에 더 안전하게 치우치도록 함
    ranked = candidates.sort_values(
        by=["f1", "precision", "threshold"],
        ascending=[False, False, True],
    )
    chosen_threshold = float(ranked.iloc[0]["threshold"])

    # table 전체(모든 임계값), candidates(정책 통과분), chosen_row(최종 선택)를 함께 반환해
    # 보고서 작성이나 디버깅 시 왜 이 임계값이 선택됐는지 추적할 수 있게 합니다.
    return {
        "chosen_threshold": chosen_threshold,
        "table": table,
        "candidates": candidates,
        "chosen_row": ranked.iloc[0].to_dict(),
    }


# 스크립트를 직접 실행했을 때만 아래 데모 흐름을 돌립니다(다른 파일에서 import할 때는 실행되지 않음).
if __name__ == "__main__":
    # --- 필수 1: 회귀 모델 vs train 평균 기준 모델 ---------------------------------
    regression_report = build_regression_report(
        X_reg_train, y_reg_train, X_reg_valid, y_reg_valid
    )
    print("\n[회귀 validation 지표]")
    # round(v, 3): 화면 출력을 보기 좋게 다듬기 위한 것일 뿐, 내부 계산에는 원래 정밀도를 그대로 사용합니다.
    print("model   :", {k: round(v, 3) for k, v in regression_report["model_metrics"].items()})
    print("baseline:", {k: round(v, 3) for k, v in regression_report["baseline_metrics"].items()})

    # --- 필수 2: 분류 임계값 지표 + 순위 지표 ---------------------------------------
    classification_report_ = build_classification_report(
        X_cls_train, y_cls_train, X_cls_valid, y_cls_valid
    )
    print("\n[분류 validation 지표 @0.5]")
    print({k: round(v, 4) for k, v in classification_report_["metrics"].items()})

    # --- 심화 1: Recall 정책 기반 임계값 선택 ---------------------------------------
    # 여기서 넘기는 probability는 위에서 계산한 "validation" 확률입니다. test 확률이 아님에 주의합니다.
    threshold_result = choose_threshold(
        y_cls_valid, classification_report_["probability"], minimum_recall=0.90
    )
    chosen = threshold_result["chosen_row"]
    print("\n[임계값 정책 보고]")
    print("- 정책: validation Recall >= 0.90")
    print(f"- 선택 임계값: {chosen['threshold']:.2f}")
    print(
        "- validation Precision / Recall / F1: "
        f"{chosen['precision']:.4f} / {chosen['recall']:.4f} / {chosen['f1']:.4f}"
    )

    # 여기서부터가 "test를 한 번만, 마지막에 확인"하는 구간입니다.
    # 선택에 쓰인 fitted clf를 그대로 사용해 test 확률을 계산합니다(재학습 금지).
    # 만약 여기서 clf.fit(...)을 다시 호출하면, train과 다른 조건으로 학습된 모델이
    # validation에서 고른 임계값과 짝이 맞지 않게 되어 비교 자체가 무의미해집니다.
    clf = classification_report_["model"]
    test_probability = clf.predict_proba(X_cls_test)[:, 1]
    # validation에서 확정된 chosen_threshold를 그대로 test에 적용합니다(test로 임계값을 다시 고르지 않음).
    test_prediction = (test_probability >= threshold_result["chosen_threshold"]).astype(int)
    test_metrics = {
        "average_precision": average_precision_score(y_cls_test, test_probability),
        "precision": precision_score(y_cls_test, test_prediction),
        "recall": recall_score(y_cls_test, test_prediction),
        "f1": f1_score(y_cls_test, test_prediction),
    }
    print(
        "- test AP / Precision / Recall / F1: "
        f"{test_metrics['average_precision']:.4f} / {test_metrics['precision']:.4f} / "
        f"{test_metrics['recall']:.4f} / {test_metrics['f1']:.4f}"
    )
    # 아래 두 문장은 md의 "임계값 정책 보고" 양식이 요구하는 설명을 그대로 채운 것입니다.
    print("- 같은 fitted 모델을 유지한 이유: 임계값 선택에 쓰인 확률 분포와 test 확률 분포를 같은 결정 경계에서 비교하기 위함입니다.")
    print("- test를 보고 다시 선택하지 않는 이유: test를 선택에 다시 쓰면 test가 사실상 validation이 되어 성능이 낙관적으로 부풀려집니다.")