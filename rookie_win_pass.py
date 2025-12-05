# ===============================================================
# candidate_analysis_shap_xgb_full15_fixed.py
# 15 特徴量すべてを「必ず可視化」する完全版
# ===============================================================

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

# ==================== 日本語フォント ====================
font_path = "C:/Windows/Fonts/meiryo.ttc"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ==================== データ ====================
INPUT_PATH = "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
data = pd.read_excel(INPUT_PATH, engine="openpyxl")

data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(r"\s+", "", regex=True)

print("Excel列名:", list(data.columns))

# ==================== 特徴量15個 ====================
features_master = [
    '年齢','性別','党派',
    '衆参当選回数','参議院当選回数','衆議院当選回数',
    '議席数','争点1位','争点2位','争点3位',
    '政府規模','出生地外立候補フラグ',
    '秘書経験フラグ','地方議会経験フラグ','職業(分類)'
]

target = "当落"

# ==================== 存在しない列を自動補完（重要!!!） ====================
missing = [c for c in features_master if c not in data.columns]
if missing:
    print("\n⚠️ 以下の特徴量が Excel に存在しません → 自動でゼロ列を追加します")
    print(missing)
    for col in missing:
        data[col] = 0  # 全部ゼロで補完

# これで 15 列が必ず揃う
features = features_master.copy()

# ==================== 目的変数0/1変換 ====================
y_raw = data[target]

uniq = list(pd.Series(y_raw.dropna().unique()).astype(str))
mapping = {}
for val in uniq:
    s = str(val)
    if "当" in s:
        mapping[val] = 1
    elif "落" in s:
        mapping[val] = 0
    else:
        mapping[val] = 0  # その他は0にまとめる

y = y_raw.map(mapping).astype(int)
print("\n目的変数の分布:")
print(y.value_counts())

# ==================== 欠損値補完 ====================
for col in features:
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna("不明")

# ==================== ラベルエンコーディング ====================
label_encoders = {}
for col in features:
    if not pd.api.types.is_numeric_dtype(data[col]):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# ==================== X, y ====================
X = data[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== モデル ====================
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
print(f"\nモデル精度: {(model.predict(X_test)==y_test).mean():.3f}")

# ==================== SHAP ====================
explainer = shap.TreeExplainer(model)
shap_arr = explainer.shap_values(X)

# shap が list の場合（XGBoost あるある）
if isinstance(shap_arr, list):
    shap_arr = shap_arr[1] if len(shap_arr) == 2 else np.array(shap_arr).sum(axis=0)

shap_abs_mean = np.abs(shap_arr).mean(axis=0)

# ==================== 15列揃っていることを最終強制 ====================
if len(shap_abs_mean) != 15:
    print("\n⚠️ SHAP 出力が 15 個ではなかったため補正します")
    fixed = np.zeros(15)
    n = min(15, len(shap_abs_mean))
    fixed[:n] = shap_abs_mean[:n]
    shap_abs_mean = fixed

# ==================== 可視化 ====================
df_imp = pd.DataFrame({
    "特徴量": features,
    "平均絶対SHAP値": shap_abs_mean
}).sort_values("平均絶対SHAP値")

plt.figure(figsize=(12, 10))
plt.barh(df_imp["特徴量"], df_imp["平均絶対SHAP値"])
plt.title("SHAP 重要度（15特徴量すべて表示）")

ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)

plt.tight_layout()
plt.savefig("output/shap_bar_fixed15.png", dpi=200)
plt.show()

print("\n保存しました: output/shap_bar_fixed15.png")
