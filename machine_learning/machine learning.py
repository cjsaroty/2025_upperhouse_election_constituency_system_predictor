import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from category_encoders import CatBoostEncoder
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib

plt.rcParams["font.family"] = "MS Gothic"

# ============================
# データ読み込み
# ============================
df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip()

rename_dict = {
    "衆参すべての当選回数": "衆参当選回数",
    "参議院の当選回数": "参議院当選回数",
    "衆議院の当選回数": "衆議院当選回数",
    "大きな政府か小さな政府か(1に近いほど小さな政府/5に近いほど大きな政府)": "政府規模",
    "出生地からの立候補か(0が出生地から立候補、1が出生地から立候補ではない)": "出生地外立候補フラグ",
    "秘書経験の有無(0が秘書経験あり、1が秘書経験なし)": "秘書経験フラグ",
    "地方議会経験の有無(0が地方議会経験あり、1が地方議会経験なし)": "地方議会経験フラグ"
}
df = df.rename(columns=rename_dict)

# ============================
# 目的変数
# ============================
df["当落"] = df["当落"].map({"当": 1, "落": 0})
y = df["当落"]

# ============================
# 特徴量セット（議席数 → 都道府県）
# ============================
features = [
    "年齢", "性別", "衆参当選回数",
    "参議院当選回数", "衆議院当選回数",
    "都道府県",
    "争点1位", "争点2位", "争点3位",
    "政府規模", "出生地外立候補フラグ",
    "秘書経験フラグ", "地方議会経験フラグ",
    "職業(分類)"
]
X = df[features].copy()

# ============================
# CatBoost Encoding（カテゴリ列）
# ============================
cat_cols = ["党派", "元現新", "争点1位", "争点2位", "争点3位", "職業(分類)", "都道府県"]

# 元のカテゴリ列を X に残しておく
for col in cat_cols:
    if col not in X.columns:
        X[col] = df[col]

cbe = CatBoostEncoder()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_cbe = np.zeros((len(X), len(cat_cols)))
for tr_idx, va_idx in kf.split(X):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    cbe.fit(X_tr[cat_cols], y_tr)
    X_cbe[va_idx, :] = cbe.transform(X_va[cat_cols]).values

# 変換結果を新しい列として追加
for i, col in enumerate(cat_cols):
    X[f"{col}_cbe"] = X_cbe[:, i]

# 元の object 型列は削除（LightGBM は object 型を扱えない）
X = X.drop(columns=cat_cols)

# ============================
# LightGBM 学習
# ============================
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 255,
    "max_depth": -1,
    "min_data_in_leaf": 1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "class_weight": None,
    "verbose": -1,
    "seed": 42
}
train_data = lgb.Dataset(X, y)

model = lgb.train(
    params,
    train_data,
    num_boost_round=50000
)

# ============================
# 学習データで予測
# ============================
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob >= 0.5).astype(int)

# ============================
# 評価
# ============================
print("\n*** LightGBM 評価（学習データ） ***")
print(confusion_matrix(y, y_pred))
print(f"Accuracy : {accuracy_score(y, y_pred):.3f}")
print(f"Precision: {precision_score(y, y_pred):.3f}")
print(f"Recall   : {recall_score(y, y_pred):.3f}")
print(f"F1-score : {f1_score(y, y_pred):.3f}")

# ============================
# 特徴量重要度
# ============================
lgb.plot_importance(model, figsize=(8, 10))
plt.title("LightGBM Feature Importance")
plt.show()

# ============================
# モデル保存
# ============================
model_package = {
    "model": model,
    "features": list(X.columns),
    "cbe_cols": cat_cols,
    "cbe": cbe
}

joblib.dump(model_package, "./machine learning/model_package.pkl")
print("モデル保存完了：machine learning/model_package.pkl")
