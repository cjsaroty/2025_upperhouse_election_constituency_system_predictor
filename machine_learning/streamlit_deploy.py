# app.py
# 修正版 — 「当落」が object 型で groupby.mean() が失敗する問題を解消し、
# 学習時の前処理（CBE で追加された特徴量）と予測時の特徴列整合性を保つよう修正済み。
import streamlit as st
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
import joblib
import shap

plt.rcParams["font.family"] = "MS Gothic"
sns.set_style("whitegrid")

# ---------------------------
# ユーティリティ
# ---------------------------
@st.cache_data
def load_excel(uploaded):
    # pandas が openpyxl を使って読み込む
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
    """
    目的変数 y を安全に数値化する：
    - 既に数値ならそのまま
    - '当選','落選','当','落','当選 ', '落選 ' などをマップ
    - '1','0' の文字を数値に変換
    - True/False を 1/0 に
    - 上記で変換できなければ LabelEncoder を使う（最後の手段）
    戻り値は pandas.Series（数値 or 整数）
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float)

    y_ser = y.fillna("").astype(str).str.strip()

    # 代表的ラベルのマップ
    mapping = {
        "当選": 1, "落選": 0, "当": 1, "落": 0,
        "合格": 1, "不合格": 0,
        "win": 1, "lose": 0, "W": 1, "L": 0,
        "True": 1, "False": 0, "true": 1, "false": 0
    }

    # 小文字化キー対応等
    mapped = y_ser.map(mapping)
    if mapped.notnull().all():
        return mapped.astype(float)

    # マップで一部しか置換されない場合は、数値文字列を変換
    def try_numeric(val):
        try:
            # 例えば "1" -> 1.0, "0" -> 0.0
            return float(val)
        except:
            return np.nan

    numeric_candidate = y_ser.map(try_numeric)
    # numeric_candidate が全て NaN でなければそれを採用（混在している場合は元の mapped を優先で併合）
    combined = mapped.copy()
    combined[pd.isna(combined)] = numeric_candidate[pd.isna(combined)]

    if combined.notnull().all():
        return combined.astype(float)

    # まだ混在しているなら、当/落が混在しているケースを優先的に map して残りは LabelEncoder
    # 最後の手段：LabelEncoder
    le = LabelEncoder()
    try:
        encoded = le.fit_transform(y_ser)
        return pd.Series(encoded.astype(float), index=y_ser.index)
    except Exception:
        # 最終的に 0/1 に分けられない場合はエラーを投げる
        raise ValueError("目的変数を数値に変換できませんでした。'当選/落選' などの形式にしてください。")

def prepare_label_encoders(df, label_cols):
    """
    LabelEncoder を各カテゴリ列に適用（文字列・カテゴリ向け）。
    数値列に対しては何もしない。
    戻り値: (df_encoded, encoders_dict)
    """
    encoders = {}
    for col in label_cols:
        if col in df.columns:
            # 数値列はスキップ（を文字列としてラベル化したくないため）
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            le = LabelEncoder()
            df[col] = df[col].fillna("nan").astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders

def apply_cbe_kfold(X, y, label_cols, n_splits=5, random_state=42):
    """
    CatBoostEncoder を k-fold で適用して各カテゴリ列に対応する *_cbe 列を追加する。
    X（DataFrame）はコピーして返す。
    戻り値: (X_with_cbe, fitted_cbe_encoder_object)
    """
    X = X.copy().reset_index(drop=True)
    y = y.reset_index(drop=True)
    # 有効な label_cols をフィルタ
    valid_label_cols = [c for c in label_cols if c in X.columns]
    if len(valid_label_cols) == 0:
        return X, None

    cbe = CatBoostEncoder()
    kf = KFold(n_splits=max(2, int(n_splits)), shuffle=True, random_state=random_state)
    X_cbe = np.zeros((len(X), len(valid_label_cols)))

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        # fit on train folds
        cbe.fit(X_tr[valid_label_cols], y_tr)
        # transform validation fold
        transformed = cbe.transform(X_va[valid_label_cols])
        # transformed may be DataFrame
        X_cbe[va_idx, :] = transformed.values

    # full fit on all data for future transform
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
# サイドバー・ヘッダー
# ---------------------------
st.set_page_config(layout="wide", page_title="当落予測ダッシュボード")
st.sidebar.title("ナビゲーション")
page = st.sidebar.radio(
    "ページ選択",
    ["Overview", "Train Model", "Candidate Prediction",
     "Party / Region Analysis", "Feature Analysis", "Model Management"]
)

# ---------------------------
# 共通：ファイルアップロード
# ---------------------------
st.sidebar.write("---")
uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロード", type=["xlsx"])
df = None
if uploaded_file is not None:
    df = load_excel(uploaded_file)
    df = safe_rename(df)
    # ここで '当落' が存在すれば強制的に数値化しておく（以降どこでも安全に mean() などが使える）
    if "当落" in df.columns:
        try:
            df["当落"] = coerce_target_series(df["当落"])
        except Exception as e:
            st.error(f"目的変数 '当落' を数値化できませんでした: {e}")

# ---------------------------
# ページ：Overview
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
# ページ：Train Model
# ---------------------------
elif page == "Train Model":
    st.title("Train Model — モデル学習")
    if df is None:
        st.info("データをアップロードしてからこちらで学習を実行してください。")
    else:

        # 🔥 修正ポイント：LabelEncoder 適用対象に 3 つのフラグ列を追加
        label_cols = [
            c for c in [
                "党派", "元現新", "争点1位", "争点2位", "争点3位",
                "職業(分類)", "出生地外立候補フラグ",
                "秘書経験フラグ", "地方議会経験フラグ"
            ]
            if c in df.columns
        ]

        # 目的変数
        if "当落" in df.columns:
            default_index = list(df.columns).index("当落")
        else:
            default_index = 0

        target_col = st.sidebar.selectbox(
            "目的変数を選択",
            options=[c for c in df.columns],
            index=default_index
        )

        st.write("目的変数:", target_col)

        # 特徴量選択
        features = st.multiselect(
            "特徴量を選択（デフォルト推奨）",
            options=[c for c in df.columns if c != target_col],
            default=[c for c in [
                "年齢","性別","党派","元現新","衆参当選回数","参議院当選回数",
                "衆議院当選回数","議席数","争点1位","争点2位","争点3位",
                "政府規模","出生地外立候補フラグ","秘書経験フラグ",
                "地方議会経験フラグ","職業(分類)"
            ] if c in df.columns]
        )

        test_size = st.sidebar.slider("検証データ割合（test_size）", 0.1, 0.5, 0.2)

        # ハイパーパラメータ
        st.sidebar.write("LightGBM ハイパーパラメータ")
        lr = st.sidebar.number_input("learning_rate", min_value=0.0001, max_value=0.5,
                                     value=0.05, format="%.4f")
        num_leaves = st.sidebar.slider("num_leaves", 8, 512, 64)
        n_estimators = st.sidebar.number_input(
            "num_boost_round 最大",
            min_value=100, max_value=100000,
            value=5000, step=100
        )
        early_stopping_rounds = st.sidebar.number_input(
            "early_stopping_rounds",
            min_value=10, max_value=1000, value=100
        )
        val_frac_for_cbe = st.sidebar.slider("CBE 用 kfold 分割数", 3, 10, 5)

        if st.button("前処理＆学習を実行"):
            with st.spinner("前処理中..."):
                X = df[features].copy()
                y = df[target_col].copy()

                # 目的変数を安全に数値化
                try:
                    y = coerce_target_series(y)
                except Exception as e:
                    st.error(f"目的変数の変換に失敗しました: {e}")
                    st.stop()

                # LabelEncoder 適用（カテゴリ列のみ）
                X, encoders = prepare_label_encoders(X, label_cols)

                # CBE（k-fold）でカテゴリ特徴を数値化して *_cbe 列を追加
                X, cbe = apply_cbe_kfold(X, y, label_cols, n_splits=val_frac_for_cbe)

                # 学習に使う最終的な特徴列（CBE で追加された列も含める）
                features_used = X.columns.tolist()

                # split
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                except Exception as e:
                    st.error(f"train_test_split に失敗しました（stratify 可能か確認してください）: {e}")
                    st.stop()

                # LightGBM dataset
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "learning_rate": lr,
                    "num_leaves": num_leaves,
                    "verbose": -1,
                    "seed": 42
                }

            with st.spinner("LightGBM 学習中..."):
                # 学習（コールバック方式で early stopping）
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=int(n_estimators),
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=int(early_stopping_rounds)),
                        lgb.log_evaluation(period=50)
                    ]
                )

            # 評価
            y_val_pred_prob = model.predict(X_val)
            y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
            st.subheader("評価（検証データ）")
            st.write("混同行列")
            st.write(confusion_matrix(y_val, y_val_pred))
            st.write({
                "Accuracy": accuracy_score(y_val, y_val_pred),
                "Precision": precision_score(y_val, y_val_pred, zero_division=0),
                "Recall": recall_score(y_val, y_val_pred, zero_division=0),
                "F1": f1_score(y_val, y_val_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_val, y_val_pred_prob)
            })

            # 保存（モデル + 前処理情報を保存）
            model_name = f"lgb_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_obj = {
                "model": model,
                # 保存する特徴量は「実際に学習に渡した列」を保存（CBE 列を含む）
                "features": features_used,
                "cbe": cbe,
                "label_cols": label_cols,
                "label_encoders": encoders
            }
            joblib.dump(save_obj, os.path.join(MODEL_DIR, model_name))
            st.success(f"学習完了。モデルを保存しました: {model_name}")
            st.session_state["latest_model_path"] = os.path.join(MODEL_DIR, model_name)

# ---------------------------
# ページ：Candidate Prediction
# ---------------------------
elif page == "Candidate Prediction":
    st.title("Candidate Prediction — 新規候補の当落予測")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_files.sort(reverse=True)
    selected_model = st.selectbox("モデルを選択（最新を選ぶ）", options=model_files)

    if not selected_model:
        st.info("まだ学習済みモデルがありません。まず Train Model で学習してください。")
    else:
        mdl = joblib.load(os.path.join(MODEL_DIR, selected_model))
        model = mdl["model"]
        features_used = mdl["features"]  # ここは学習時に保存した実際の特徴列
        cbe = mdl.get("cbe", None)
        label_cols = mdl.get("label_cols", [])
        label_encoders = mdl.get("label_encoders", {})

        st.write("使用モデル:", selected_model)
        st.write("入力フォームに値を入れてください。")

        # 入力フォーム作成（学習時の元の特徴列名がわからない場合に備え、
        # 数値っぽい名前は number_input、'性別' などは selectbox、その他は text_input を使う）
        input_data = {}
        # 学習時に使った特徴列が多い場合、フォームが長くなる点に注意
        for f in features_used:
            # skip CBE columns (these are generated automatically)
            if f.endswith("_cbe"):
                # 生成はモデル側で行うのでユーザー入力は不要 — 後で別途作成
                continue
            if f in ["年齢", "衆参当選回数", "参議院当選回数", "衆議院当選回数", "議席数", "政府規模"]:
                input_data[f] = st.number_input(f, value=0)
            elif f in ["性別"]:
                input_data[f] = st.selectbox(f, options=[0, 1])
            else:
                input_data[f] = st.text_input(f, value="")

        if st.button("予測する"):
            # DataFrame を作る
            X_new = pd.DataFrame([input_data])

            # LabelEncoder を適用（学習時に用いた encoders に合わせる）
            for col, le in label_encoders.items():
                if col in X_new.columns:
                    try:
                        X_new[col] = le.transform(X_new[col].fillna("nan").astype(str))
                    except Exception:
                        # 学習時に見たカテゴリでない場合は -1 を入れておく（または 0）
                        X_new[col] = 0

            # CBE 特徴作成（学習時に fit した cbe があれば transform）
            if cbe is not None and label_cols:
                for col in label_cols:
                    if col in X_new.columns:
                        try:
                            # cbe.transform expects DataFrame with label cols
                            transformed = cbe.transform(X_new[[col]])
                            X_new[f"{col}_cbe"] = transformed.iloc[:, 0]
                        except Exception:
                            X_new[f"{col}_cbe"] = 0
                    else:
                        # 欠けている列は 0 で埋める
                        X_new[f"{col}_cbe"] = 0

            # 最終的にモデルが期待する列順に整え、足りない列は 0 で補完
            X_new = X_new.reindex(columns=[c for c in features_used if not c.endswith("_cbe")] + [c for c in features_used if c.endswith("_cbe")], fill_value=0)
            # 注意: モデルは学習時に features_used の順で学習しているはずなので、その順を維持して渡す
            X_new = X_new.reindex(columns=features_used, fill_value=0)

            # モデルによっては predict が (n_samples,) ではなく (n_samples,1) の可能性があるが一般的には (n,)
            try:
                prob = model.predict(X_new)[0]
            except Exception as e:
                st.error(f"モデル予測中にエラーが発生しました: {e}")
                st.write("入力データ（デバッグ）:")
                st.dataframe(X_new)
                st.stop()

            st.metric("当選確率", f"{prob:.3f}")
            st.write("閾値0.5判定:", "当" if prob >= 0.5 else "落")

# ---------------------------
# ページ：Party / Region Analysis
# ---------------------------
elif page == "Party / Region Analysis":
    st.title("Party / Region Analysis — 政党・地域の集計")
    if df is None:
        st.info("データをアップロードしてください。")
    else:
        group_by = st.selectbox(
            "グループ化",
            options=[c for c in ["党派", "議席数", "地域", "選挙区"] if c in df.columns]
        )
        if st.button("集計実行"):
            # 当落が数値であることを保証（Overview の読み込み時点で coerce しているはず）
            try:
                summary = df.groupby(group_by).agg({"当落": ["mean", "count"]})
                summary.columns = ["当選確率", "候補数"]
                summary = summary.sort_values("当選確率", ascending=False)
                st.dataframe(summary.head(100))
                st.bar_chart(summary["当選確率"])
            except Exception as e:
                st.error(f"集計に失敗しました: {e}")

# ---------------------------
# ページ：Feature Analysis
# ---------------------------
elif page == "Feature Analysis":
    st.title("Feature Analysis — 特徴量分析")
    if df is None:
        st.info("データをアップロードしてください。")
    else:
        st.subheader("単変量分布")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.info("数値列が存在しません。")
        else:
            choose = st.selectbox("数値列を選択", options=num_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[choose].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("カテゴリ別当選率")
        # カテゴリ列の判定を厳密に（object または unique < 30）
        cat_cols = [c for c in df.columns if (df[c].dtype == object or df[c].nunique() < 30)]
        if len(cat_cols) == 0:
            st.info("カテゴリ列が見つかりません。")
        else:
            chosen_cat = st.selectbox("カテゴリ列を選ぶ", options=cat_cols)
            try:
                # 当落が数値であることを前提に mean をとる（coerce していれば安全）
                agg = df.groupby(chosen_cat)["当落"].mean().sort_values(ascending=False)
                st.dataframe(agg)
                st.bar_chart(agg)
            except Exception as e:
                st.error(f"カテゴリ別当選率の計算に失敗しました: {e}")
                st.write("選択列のデータ型とサンプル:")
                st.write(df[chosen_cat].head())
                st.write("当落列の型・サンプル:")
                st.write(df["当落"].dtype)
                st.write(df["当落"].head())

# ---------------------------
# ページ：Model Management
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
