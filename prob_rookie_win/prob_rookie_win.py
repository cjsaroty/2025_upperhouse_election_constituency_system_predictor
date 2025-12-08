# candidate_analysis_safe_shap_jp_safe_columns.py
import pandas as pd
import numpy as np
import lightgbm as lgb
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

# ---------- 日本語フォント設定 ----------
font_path = "C:/Windows/Fonts/meiryo.ttc"  # Windows環境
import warnings

warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

# ==================== 日本語フォント ====================
font_path = "C:/Windows/Fonts/meiryo.ttc"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ---------- データ読み込み ----------
data = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

# 列名を正規化（前後空白・不可視文字削除）
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(r'\s+', '', regex=True)

print("Excelの列名一覧:")
print(list(data.columns))

# ---------- 特徴量リスト（元現新を除いた15個） ----------
features = [
    '年齢','性別','党派',
    '衆参すべての当選回数','参議院の当選回数','衆議院の当選回数',
    '議席数','争点1位','争点2位','争点3位',
    '大きな政府か小さな政府か','出生地からの立候補か',
    '秘書経験の有無','地方議会経験の有無','職業'
]
target = '当落'

# ---------- 存在しない列を自動除外 ----------
existing_features = [col for col in features if col in data.columns]
missing = set(features) - set(existing_features)
if missing:
    print(f"存在しない列を除外しました: {missing}")
features = existing_features

# ---------- 欠損値補完 ----------
for col in features:
    if data[col].dtype in [np.int64, np.float64]:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna(data[col].mode()[0])

# ---------- カテゴリ変数のラベルエンコード ----------
cat_features = ['性別','党派','争点1位','争点2位','争点3位','職業']
for col in cat_features:
    if col in features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# ---------- 学習用データ ----------
X = data[features]
y = data[target]

# ---------- 学習用・テスト用に分割 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- LightGBM分類器 ----------
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X_train, y_train)

# ---------- 予測 ----------
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\nモデル精度: {accuracy:.3f}")

# ---------- 学習に使った特徴量を確認 ----------
print("LightGBMで学習した特徴量:", model.feature_name_)

# ---------- 特徴量重要度 ----------
importances = pd.DataFrame({
    '特徴量': features,
    '重要度': model.feature_importances_
}).sort_values(by='重要度', ascending=False)
print("\n特徴量重要度:")
print(importances)

# ---------- SHAP可視化 ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 二値分類の場合はクラス1(当選)のSHAP値を使用
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

# summary_plot に全列を渡す
shap.summary_plot(shap_values_to_plot, X, show=True, plot_size=(12,8))

# y軸ラベルに日本語フォントを適用
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

# ---------- 当選しやすい属性の提案 ----------
def suggest_best_attributes(data, features, target):
    suggestions = {}
    for col in features:
        if data[col].dtype in [np.int64, np.float64]:
            best_val = data.loc[data[target]==1, col].mean()
        else:
            best_val = data[data[col].notna()].groupby(col)[target].mean().idxmax()
        suggestions[col] = best_val
    return suggestions

best_attrs = suggest_best_attributes(data, features, target)
print("\n当選しやすい属性の提案:")
for k, v in best_attrs.items():
    print(f"{k}: {v}")

plt.tight_layout()
plt.savefig("output/shap_bar_fixed15.png", dpi=200)
plt.show()

print("\n保存しました: output/shap_bar_fixed15.png")

