import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# 1) Генератор тестовых транзакций
def generate_transactions(num=100, users=20, countries=("A", "B", "C", "D")):
    """
    Генерирует список транзакций:
    transaction_id, user_id, amount, timestamp, merchant_country
    """
    now = datetime.now()
    txns = []

    for i in range(1, num + 1):
        user_id = f"U{random.randint(1, users)}"
        country = random.choice(countries)

        amount = round(random.choice([
            random.uniform(5, 3000),
            random.uniform(3000, 12000),
            random.uniform(12000, 30000)
        ]), 2)

        ts = now - timedelta(minutes=random.randint(0, 180), seconds=random.randint(0, 59))

        txns.append({
            "transaction_id": f"T{i:06d}",
            "user_id": user_id,
            "amount": amount,
            "timestamp": ts,
            "merchant_country": country
        })

    txns.sort(key=lambda x: x["timestamp"])
    return txns


# 2) Rule-based Fraud Detection + отчёт
def fraud_detection(
    transactions,
    amount_limit=10000,
    freq_limit=5,
    window_minutes=60,
    high_risk_countries=("C",),
):
    """
    Статусы:
    - APPROVED
    - FLAGGED: High Amount
    - FLAGGED: High Frequency
    - BLOCKED: High Risk Country & Amount
    """
    results = []
    user_recent = defaultdict(deque)
    window = timedelta(minutes=window_minutes)
    high_risk_set = set(high_risk_countries)

    for txn in transactions:
        uid = txn["user_id"]
        ts = txn["timestamp"]

        dq = user_recent[uid]
        while dq and (ts - dq[0]) > window:
            dq.popleft()
        dq.append(ts)

        is_high_freq = len(dq) > freq_limit

        is_high_risk_country = txn["merchant_country"] in high_risk_set
        effective_limit = amount_limit / 2 if is_high_risk_country else amount_limit
        is_high_amount = txn["amount"] > effective_limit

        if is_high_risk_country and is_high_amount:
            status = "BLOCKED: High Risk Country & Amount"
        elif is_high_amount:
            status = "FLAGGED: High Amount"
        elif is_high_freq:
            status = "FLAGGED: High Frequency"
        else:
            status = "APPROVED"

        results.append({
            **txn,
            "effective_amount_limit": effective_limit,
            "txns_in_last_hour": len(dq),
            "is_high_risk_country": int(is_high_risk_country),
            "is_high_amount": int(is_high_amount),
            "is_high_freq": int(is_high_freq),
            "status": status,
        })

    return results


# 3) Синтетическая разметка fraud=1/0 (для демо ML)
def make_labels_for_ml(rule_results, noise=0.05):
    """
    В реальности разметка берётся из чарджбэков/расследований/кейсов.
    Здесь — демо: считаем fraud=1, если:
      - BLOCKED => почти всегда fraud
      - FLAGGED по сумме/частоте => часто fraud
      - APPROVED => редко fraud
    + небольшой шум, чтобы модель была более «живой».
    """
    y = []
    for r in rule_results:
        if r["status"].startswith("BLOCKED"):
            p = 0.95
        elif r["status"] == "FLAGGED: High Amount":
            p = 0.55
        elif r["status"] == "FLAGGED: High Frequency":
            p = 0.40
        else:
            p = 0.05

        # шум: слегка «размываем» вероятности
        p = min(0.99, max(0.01, p + random.uniform(-noise, noise)))
        y.append(1 if random.random() < p else 0)
    return y


# 4) Формирование датасета признаков для ML
def build_ml_dataset(rule_results):
    """
    Возвращает X (list[dict]) и список имён фич.
    Мы не используем transaction_id (это идентификатор, не признак).
    """
    X = []
    for r in rule_results:
        ts: datetime = r["timestamp"]
        X.append({
            "amount": float(r["amount"]),
            "merchant_country": r["merchant_country"],
            "txns_in_last_hour": int(r["txns_in_last_hour"]),
            "is_high_risk_country": int(r["is_high_risk_country"]),
            "hour_of_day": int(ts.hour),
            "day_of_week": int(ts.weekday()),  # 0=Mon ... 6=Sun
        })
    return X


# 5) Обучение ML-модели (scikit-learn)
def train_ml_model(X, y, random_state=42):
    """
    Модель: RandomForest + OneHotEncoder для страны.
    Выводит метрики и возвращает pipeline.
    """
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # Преобразуем списки словарей в DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    categorical = ["merchant_country"]
    numeric = ["amount", "txns_in_last_hour", "is_high_risk_country", "hour_of_day", "day_of_week"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=2
    )

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n=== ML EVALUATION ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.3f}")
    except ValueError:
        print("ROC-AUC: not available (need both classes in y_test)")

    # feature importances (для RF) — аккуратно достанем имена после one-hot
    ohe = clf.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical))
    feature_names = cat_feature_names + numeric

    importances = clf.named_steps["model"].feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\nTop-10 feature importances:")
    for name, val in ranked[:10]:
        print(f"  {name:25s} {val:.4f}")

    return clf


# 6) Печать rule-based отчёта
def print_rule_report(rule_results, limit=30):
    print("\n=== RULE-BASED REPORT (first rows) ===")
    for r in rule_results[:limit]:
        ts_str = r["timestamp"].isoformat(sep=" ", timespec="seconds")
        print(
            f"{r['transaction_id']} | {r['user_id']} | "
            f"{r['amount']:>9.2f} | {ts_str} | {r['merchant_country']} | "
            f"eff_limit={r['effective_amount_limit']:.0f} | "
            f"last_hour={r['txns_in_last_hour']} | {r['status']}"
        )
    if len(rule_results) > limit:
        print(f"... ({len(rule_results) - limit} more rows)")


# запуск кода
if __name__ == "__main__":
    random.seed(42)

    # 1) Генерируем данные
    txns = generate_transactions(num=300, users=30, countries=("A", "B", "C", "D"))

    # 2) Rule-based детекция (с регуляторным снижением лимита для high-risk)
    rule_results = fraud_detection(
        txns,
        amount_limit=10000,
        freq_limit=5,
        window_minutes=60,
        high_risk_countries=("C",),   # "C" — high risk
    )

    # Печать части отчёта
    print_rule_report(rule_results, limit=25)

    # 3) Делаем синтетические метки fraud=1/0
    y = make_labels_for_ml(rule_results, noise=0.05)

    # 4) Признаки для ML
    X = build_ml_dataset(rule_results)

    # 5) Обучаем модель и печатаем метрики
    _clf = train_ml_model(X, y, random_state=42)

    # Пример: применим модель к нескольким последним транзакциям и покажем риск-скор
    print("\n=== ML SCORING (last 10 transactions) ===")
    last10 = X[-10:]
    last10_df = pd.DataFrame(last10)
    proba = _clf.predict_proba(last10_df)[:, 1]
    for i, p in enumerate(proba, start=1):
        print(f"txn(last-{11 - i}) fraud_proba={p:.3f}")

