import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
import matplotlib.pyplot as plt

# ============================
# 日本語表示対応
# ============================
plt.rcParams["font.family"] = "MS Gothic"  # WindowsならMS Gothic、MacならAppleGothicなど

# ============================
# 争点入力時にNaN表示を回避
# ============================

# ============================
# データ読み込み
# ============================
df = pd.read_excel(
    r"C:\Users\owner\OneDrive\デスクトップ\2025_upperhouse_election_predictor\Data\2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# 列名の前後スペース削除
df.columns = df.columns.str.strip()

# 列名を短くして使いやすくする
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
# ラベルエンコード
# ============================
label_cols = ["党派", "元現新", "争点1位", "争点2位", "争点3位", "職業(分類)"]
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

df["当落"] = df["当落"].map({"当": 1, "落": 0})

# ============================
# 説明変数と目的変数
# ============================
X = df[
    [
        "年齢",
        "性別",
        "党派",
        "元現新",
        "衆参当選回数",
        "参議院当選回数",
        "衆議院当選回数",
        "議席数",
        "争点1位",
        "争点2位",
        "争点3位",
        "政府規模",
        "出生地外立候補フラグ",
        "秘書経験フラグ",
        "地方議会経験フラグ",
        "職業(分類)"
    ]
]

y = df["当落"]

# ============================
# 学習データとテストデータに分割
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 評価関数
# ============================
def evaluate_model(name, y_test, y_pred):
    print(f"\n*** {name} 評価 ***")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")

# ============================
# RandomForest + GridSearch
# ============================
rf_param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid=rf_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=1,
)

rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
evaluate_model("RandomForest (best)", y_test, y_pred_rf)

# ============================
# 特徴量重要度プロット
# ============================
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=True).plot(kind="barh", figsize=(8, 6))
plt.title("ランダムフォレスト特徴量重要度")
plt.tight_layout()
plt.show()

# ============================
# ユーザー入力による予測（改修版）
# ============================
print("\n*** 当選確率予測アプリ（16特徴量：年齢追加） ***")

candidate_name = input("立候補者のフルネームを入力してください: ")

def get_input(prompt, cast_type=int):
    return cast_type(input(prompt))

# ★ 年齢を追加（性別の前）
age = get_input("年齢を入力してください（整数）: ")

gender = get_input("性別を入力してください（男性=0, 女性=1）: ")
party = input(f"党派を入力してください {list(encoders['党派'].classes_)}: ")
status = input(f"元現新を入力してください {list(encoders['元現新'].classes_)}: ")

senkyo_all = get_input("衆参すべての当選回数: ")
sangiin = get_input("参議院の当選回数: ")
shugiin = get_input("衆議院の当選回数: ")

seats = get_input("議席数を入力してください: ")

issue1 = input(f"争点1位を入力してください {list(encoders['争点1位'].classes_)}: ")
issue2 = input(f"争点2位を入力してください {list(encoders['争点2位'].classes_)}: ")
issue3 = input(f"争点3位を入力してください {list(encoders['争点3位'].classes_)}: ")

gov_scale = get_input("大きな政府(5) 小さな政府(1) の尺度 (1〜5): ")
birth_flag = get_input("出生地から立候補か?（0=はい,1=いいえ）: ")
secretary_flag = get_input("秘書経験あり?（0=あり,1=なし）: ")
local_flag = get_input("地方議会経験あり?（0=あり,1=なし）: ")

job = input(f"職業分類を入力してください {list(encoders['職業(分類)'].classes_)}: ")

# エンコード
party_encoded = encoders["党派"].transform([party])[0]
status_encoded = encoders["元現新"].transform([status])[0]
issue1_encoded = encoders["争点1位"].transform([issue1])[0]
issue2_encoded = encoders["争点2位"].transform([issue2])[0]
issue3_encoded = encoders["争点3位"].transform([issue3])[0]
job_encoded = encoders["職業(分類)"].transform([job])[0]

# ============================
# ★ DataFrame 作成（年齢を最初に追加）
# ============================
X_input = pd.DataFrame(
    [[
        age,                 # ←★追加
        gender,
        party_encoded,
        status_encoded,
        senkyo_all,
        sangiin,
        shugiin,
        seats,
        issue1_encoded,
        issue2_encoded,
        issue3_encoded,
        gov_scale,
        birth_flag,
        secretary_flag,
        local_flag,
        job_encoded
    ]],
    columns=X.columns
)

# 予測
prob = best_rf_model.predict_proba(X_input)[0][1]
label = "当選" if prob >= 0.5 else "落選"

print(f"\n立候補者：{candidate_name}")
print(f"当選確率：{prob*100:.2f}% → 予測ラベル：{label}")
print("（使用モデル：ランダムフォレスト）")