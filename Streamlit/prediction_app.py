# 当落予測ダッシュボード
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

jp_font = "MS Gothic"

plt.rcParams.update({
    "font.family": jp_font,
    "axes.unicode_minus": False
})

# SHAP はオプション（インストールされていない環境でも動くよう try/except）
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

plt.rcParams["font.family"] = "MS Gothic"
sns.set_style("whitegrid")
shap_option = st.sidebar.checkbox("SHAPを表示する")

# ---------------------------
# ユーティリティ
# ---------------------------
@st.cache_data
def load_excel(uploaded):
    return pd.read_excel(uploaded, engine="openpyxl")

def safe_rename(df):
    df = df.copy()
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
    for k, v in rename_dict.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    return df

def coerce_target_series(y):
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float)
    y_ser = y.fillna("").astype(str).str.strip()
    mapping = {
        "当選": 1, "落選": 0, "当": 1, "落": 0,
        "合格": 1, "不合格": 0,
        "win": 1, "lose": 0, "W": 1, "L": 0,
        "True": 1, "False": 0, "true": 1, "false": 0
    }
    mapped = y_ser.map(mapping)
    if mapped.notnull().all():
        return mapped.astype(float)
    def try_numeric(val):
        try:
            return float(val)
        except:
            return np.nan
    numeric_candidate = y_ser.map(try_numeric)
    combined = mapped.copy()
    combined[pd.isna(combined)] = numeric_candidate[pd.isna(combined)]
    if combined.notnull().all():
        return combined.astype(float)
    le = LabelEncoder()
    try:
        encoded = le.fit_transform(y_ser)
        return pd.Series(encoded.astype(float), index=y_ser.index)
    except Exception:
        raise ValueError("目的変数を数値に変換できませんでした。'当選/落選' などの形式にしてください。")

def prepare_label_encoders(df, label_cols):
    encoders = {}
    for col in label_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            le = LabelEncoder()
            df[col] = df[col].fillna("nan").astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders

def apply_cbe_kfold(X, y, label_cols, n_splits=5, random_state=42):
    X = X.copy().reset_index(drop=True)
    y = y.reset_index(drop=True)
    valid_label_cols = [c for c in label_cols if c in X.columns]
    if len(valid_label_cols) == 0:
        return X, None
    cbe = CatBoostEncoder()
    kf = KFold(n_splits=max(2, int(n_splits)), shuffle=True, random_state=random_state)
    X_cbe = np.zeros((len(X), len(valid_label_cols)))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        cbe.fit(X_tr[valid_label_cols], y_tr)
        transformed = cbe.transform(X_va[valid_label_cols])
        X_cbe[va_idx, :] = transformed.values
    cbe.fit(X[valid_label_cols], y)
    for i, col in enumerate(valid_label_cols):
        X[f"{col}_cbe"] = X_cbe[:, i]
    return X, cbe

# ---------------------------
# モデル保存フォルダ
# ---------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# UI設定
# ---------------------------
st.set_page_config(layout="wide", page_title="当落予測ダッシュボード (SHAP 統合)")
st.sidebar.title("ナビゲーション")
page = st.sidebar.radio(
    "ページ選択",
    ["Overview", "Candidate Prediction",
    "Feature Analysis", "Model Management"]
)

# ---------------------------
# ファイルアップロード
# ---------------------------
st.sidebar.write("---")
uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロード", type=["xlsx"])
df = None
if uploaded_file is not None:
    df = load_excel(uploaded_file)
    df = safe_rename(df)
    if "当落" in df.columns:
        try:
            df["当落"] = coerce_target_series(df["当落"])
        except Exception as e:
            st.error(f"目的変数 '当落' を数値化できませんでした: {e}")

# ---------------------------
# Overview
# ---------------------------
if page == "Overview":
    st.title("Overview — データ確認")
    if df is None:
        st.info("左のサイドバーから Excel ファイルをアップロードしてください。")
    else:
        st.subheader("データプレビュー")
        st.dataframe(df.head())
        st.write("形状:", df.shape)
        st.subheader("統計概要（数値列）")
        st.dataframe(df.describe())
        st.subheader("欠損値の簡易チェック")
        miss = df.isnull().sum().sort_values(ascending=False).head(20)
        st.dataframe(miss[miss > 0])

# ---------------------------
# Candidate Prediction
# ---------------------------
if page == "Candidate Prediction":
    st.title("新規候補の当落予測")

    if df is None:
        st.info("まず Excel データをアップロードしてください。")
        st.stop()

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_files.sort(reverse=True)

    if not model_files:
        st.warning("学習済みモデルが見つかりません。")
        st.stop()

    selected_model = st.selectbox("使用するモデル", model_files)

    mdl = joblib.load(os.path.join(MODEL_DIR, selected_model))
    model = mdl["model"]
    features_used = mdl["features"]
    label_encoders = mdl.get("label_encoders", {})
    label_cols = mdl.get("label_cols", [])
    cbe = mdl.get("cbe", None)

    st.write("### 候補者情報を入力してください")

    # ===== 入力フォーム（16特徴量固定） =====
    input_data = {}

    input_data["年齢"] = st.selectbox(
        "年齢", sorted(df["年齢"].dropna().unique())
    )

    input_data["性別"] = st.selectbox(
        "性別", sorted(df["性別"].dropna().unique())
    )

    input_data["党派"] = st.selectbox(
        "党派", sorted(df["党派"].dropna().astype(str).unique())
    )

    input_data["元現新"] = st.selectbox(
        "元現新", sorted(df["元現新"].dropna().astype(str).unique())
    )

    input_data["衆参当選回数"] = st.selectbox(
        "衆参すべての当選回数", sorted(df["衆参当選回数"].dropna().unique())
    )

    input_data["参議院当選回数"] = st.selectbox(
        "参議院の当選回数", sorted(df["参議院当選回数"].dropna().unique())
    )

    input_data["衆議院当選回数"] = st.selectbox(
        "衆議院の当選回数", sorted(df["衆議院当選回数"].dropna().unique())
    )

    input_data["都道府県"] = st.selectbox(
        "都道府県", sorted(df["都道府県"].dropna().astype(str).unique())
    )

    input_data["争点1位"] = st.selectbox(
        "争点1位", sorted(df["争点1位"].dropna().astype(str).unique())
    )

    input_data["争点2位"] = st.selectbox(
        "争点2位", sorted(df["争点2位"].dropna().astype(str).unique())
    )

    input_data["争点3位"] = st.selectbox(
        "争点3位", sorted(df["争点3位"].dropna().astype(str).unique())
    )

    input_data["政府規模"] = st.selectbox(
        "政府規模（1=小さい政府〜5=大きい政府）",
        sorted(df["政府規模"].dropna().unique())
    )

    input_data["出生地外立候補フラグ"] = st.selectbox(
        "出生地外立候補 (0=地元 / 1=地元以外)", [0, 1]
    )

    input_data["秘書経験フラグ"] = st.selectbox(
        "秘書経験 (0=あり / 1=なし)", [0, 1]
    )

    input_data["地方議会経験フラグ"] = st.selectbox(
        "地方議会経験 (0=あり / 1=なし)", [0, 1]
    )

    input_data["職業(分類)"] = st.selectbox(
        "職業(分類)", sorted(df["職業(分類)"].dropna().astype(str).unique())
    )

    # ===== 予測 =====
    if st.button("当選確率を予測"):
        X_new = pd.DataFrame([input_data])

        # LabelEncoder
        for col, le in label_encoders.items():
            if col in X_new.columns:
                X_new[col] = le.transform(X_new[col].astype(str))

        # CatBoostEncoder
        if cbe is not None and label_cols:
            for col in label_cols:
                X_new[f"{col}_cbe"] = cbe.transform(X_new[[col]]).iloc[:, 0]

        # モデル入力順に整形
        X_new = X_new.reindex(columns=features_used, fill_value=0)

        prob = float(model.predict(X_new)[0])

        st.subheader("予測結果")
        st.metric("当選確率", f"{prob:.3f}")
        st.write("判定:", "🟢 当選" if prob >= 0.5 else "🔴 落選")



# ---------------------------
# Feature Analysis
# ---------------------------
elif page == "Feature Analysis":
    st.title("Feature Analysis — 特徴量分析（当選率ベース）")

    if df is None:
        st.info("データをアップロードしてください。")
        st.stop()

    # =========================
    # 数値列 → 値別 当選率（区間化なし）
    # =========================
    st.subheader("数値列別 当選率（値そのまま）")

    num_cols = [
        "年齢",
        "衆参当選回数",
        "参議院当選回数",
        "衆議院当選回数",
        "政府規模"
    ]

    chosen_num = st.selectbox("数値列を選択", options=num_cols)

    try:
        tmp = df[[chosen_num, "当落"]].dropna()

        # 数値そのものを x 軸にして平均を計算
        # 同じ値が複数ある場合の平均を計算
        agg_num = tmp.groupby(chosen_num)["当落"].mean().sort_index()

        st.dataframe(agg_num.rename("当選率"))
        st.bar_chart(agg_num)  # x軸が数値列の値

    except Exception as e:
        st.error(f"数値列別当選率の計算に失敗しました: {e}")


    # =========================
    # カテゴリ列 → 当選率
    # =========================
    st.subheader("カテゴリ列別 当選率")

    cat_cols = [
        "性別",
        "党派",
        "元現新",
        "都道府県",
        "争点1位",
        "争点2位",
        "争点3位",
        "出生地外立候補フラグ",
        "秘書経験フラグ",
        "地方議会経験フラグ",
        "職業(分類)"
    ]

    chosen_cat = st.selectbox("カテゴリ列を選択", options=cat_cols)

    try:
        agg_cat = (
            df.groupby(chosen_cat)["当落"]
            .mean()
            .sort_values(ascending=False)
        )

        st.dataframe(agg_cat.rename("当選率"))
        st.bar_chart(agg_cat)

    except Exception as e:
        st.error(f"カテゴリ別当選率の計算に失敗しました: {e}")


# ---------------------------
# Model Management
# ---------------------------
elif page == "Model Management":
    st.title("Model Management — モデル管理")
    st.write("保存済みモデル一覧")
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_files.sort(reverse=True)
    st.dataframe(pd.DataFrame({"model": model_files}))
    sel = st.selectbox("操作モデルを選択", options=[""] + model_files)
    if sel:
        path = os.path.join(MODEL_DIR, sel)
        st.write("モデルパス:", path)
        if st.button("ダウンロード"):
            with open(path, "rb") as f:
                bytes_io = io.BytesIO(f.read())
            st.download_button("Download model", data=bytes_io, file_name=sel)
        if st.button("削除"):
            os.remove(path)
            st.success("削除しました。再読み込みしてください。")
