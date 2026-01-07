import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==============================
# 1. データ読み込み
# ==============================
# CSVファイル名を指定
df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

# ===== 列名クリーニング（必須）=====
df.columns = (
    df.columns
    .str.replace(r"\s+", "", regex=True)  # 改行・全角・半角スペース削除
    .str.replace("（", "(", regex=False)
    .str.replace("）", ")", regex=False)
    .str.replace("、", ",", regex=False)
)

# ==============================
# 2. 新人候補に限定
# ==============================
df_new = df[
    (df["元現新"] == "新") &
    (df["衆参すべての当選回数"] == 0) &
    (df["衆議院当選回数"] == 0) &
    (df["参議院当選回数"] == 0)
].copy()

# ==============================
# 3. 説明変数・目的変数
# ==============================
target_col = "当落"   # 当選=1、落選=0

feature_cols = [
    "年齢",
    "性別",
    "党派",
    "都道府県",
    "争点1位",
    "争点2位",
    "争点3位",
    "政府規模",
    "出生地外立候補フラグ",
    "秘書経験フラグ",
    "地方議会経験フラグ",
    "職業(分類)"
]


X = df_new[feature_cols]
y = df_new[target_col]

# ==============================
# 4. 数値変数・カテゴリ変数分離
# ==============================
numeric_features = [
    "年齢",
    "争点1位",
    "争点2位",
    "争点3位",
    "政府規模",
    "出生地外立候補フラグ",
    "秘書経験フラグ",
    "地方議会経験フラグ"
]

for col in numeric_features:
    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")



categorical_features = [
    "性別",
    "党派",
    "都道府県",
    "職業(分類)"
]

# ==============================
# 5. 前処理 + ロジスティック回帰
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# ==============================
# 6. 学習・評価
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ==============================
# 7. 当選しやすい要因の抽出
# ==============================
feature_names = (
    numeric_features +
    list(
        pipeline.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
)

coefficients = pipeline.named_steps["model"].coef_[0]

importance_df = pd.DataFrame({
    "要因": feature_names,
    "係数": coefficients
}).sort_values(by="係数", ascending=False)

print("\n=== 新人が当選しやすい要因（正の影響が強い順） ===")
print(importance_df.head(20))

print("\n=== 新人が当選しにくい要因（負の影響が強い順） ===")
print(importance_df.tail(20))

# CSVとして保存
importance_df.to_csv("newcomer_winning_factors.csv", index=False)
