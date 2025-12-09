# app.py
# 当落予測ダッシュボード（SHAP 統合版）
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
    ["Overview", "Train Model", "Candidate Prediction",
     "Party / Region Analysis", "Feature Analysis", "Model Management"]
)

# ---------------------------
# ファイルアップロード
# ---------------------------
st.sidebar.write("---")
uploaded_file = st.sidebar.file_uploader(
    "Excelファイルをアップロード",
    type=["xlsx"],
    key="excel_uploader_sidebar"
)
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
# Train Model
# ---------------------------
elif page == "Train Model":
    st.title("Train Model — モデル学習（SHAP オプション）")
    if df is None:
        st.info("データをアップロードしてからこちらで学習を実行してください。")
    else:
        # Label 列候補
        label_cols = [
            c for c in [
                "党派", "元現新", "争点1位", "争点2位", "争点3位",
                "職業(分類)", "出生地外立候補フラグ",
                "秘書経験フラグ", "地方議会経験フラグ"
            ]
            if c in df.columns
        ]

        # 目的変数選択
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

        # 「新人のみ」フィルタ（チェックボックス）
        filter_new = False
        if "元現新" in df.columns:
            filter_new = st.sidebar.checkbox("新人のみで学習する (元現新 == '新')", value=False)

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

        # SHAP 有無オプション（計算コストが高いので明示的に）
        shap_option = False
        if SHAP_AVAILABLE:
            shap_option = st.sidebar.checkbox("学習後に SHAP を計算・表示する（計算負荷あり）", value=False)
        else:
            st.sidebar.write("SHAP がインストールされていません。requirements.txt に 'shap' を追加してください。")

        if st.button("前処理＆学習を実行"):
            with st.spinner("前処理中..."):
                df_train = df.copy()
                if filter_new:
                    if "元現新" in df_train.columns:
                        df_train = df_train[df_train["元現新"].astype(str).str.strip() == "新"].copy()
                        if df_train.empty:
                            st.error("新人データが存在しません。")
                            st.stop()
                    else:
                        st.error("'元現新' 列が存在しません。")
                        st.stop()

                X = df_train[features].copy()
                y = df_train[target_col].copy()

                try:
                    y = coerce_target_series(y)
                except Exception as e:
                    st.error(f"目的変数の変換に失敗しました: {e}")
                    st.stop()

                # LabelEncoder 適用
                X, encoders = prepare_label_encoders(X, label_cols)

                # CBE
                X, cbe = apply_cbe_kfold(X, y, label_cols, n_splits=val_frac_for_cbe)

                features_used = X.columns.tolist()

                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=test_size, stratify=y, random_state=42
                    )
                except Exception as e:
                    st.error(f"train_test_split に失敗しました（stratify 可能か確認してください）: {e}")
                    st.stop()

                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "learning_rate": 0.01,
                    "num_leaves": 255,     # 木の容量最大
                    "max_depth": -1,       # 深さ制限なし
                    "min_data_in_leaf": 1, # 葉1サンプルで柔軟に
                    "feature_fraction": 1.0,
                    "bagging_fraction": 1.0,
                    "bagging_freq": 0,
                    "lambda_l1": 0.0,
                    "lambda_l2": 0.0,
                    "class_weight": None,
                    "verbose": -1,
                    "seed": 42
                }

            with st.spinner("LightGBM 学習中..."):
                model = lgb.train(
                    params,
                    train_data,
                     num_boost_round=50000,
                )

            # 評価表示
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

            # モデル保存
            model_name = f"lgb_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            save_obj = {
                "model": model,
                "features": features_used,
                "cbe": cbe,
                "label_cols": label_cols,
                "label_encoders": encoders
            }
            joblib.dump(save_obj, os.path.join(MODEL_DIR, model_name))
            st.success(f"学習完了。モデルを保存しました: {model_name}")
            st.session_state["latest_model_path"] = os.path.join(MODEL_DIR, model_name)

 # -----------------------
# SHAP 計算＆表示（オプション）
# -----------------------
if shap_option and SHAP_AVAILABLE:
    with st.spinner("SHAP を計算中（データ量により数分かかることがあります）..."):
        try:
            # X_val をサンプリング（大きい場合は 500 行に制限）
            max_shap = 500
            if len(X_val) > max_shap:
                X_shap = X_val.sample(n=max_shap, random_state=42)
                st.write(f"SHAP はサンプル {max_shap} 行で計算します（表示はサンプル）")
            else:
                X_shap = X_val

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            # shap_values の型はモデル・バージョンで変わる場合がある。ndarray に揃える
            if isinstance(shap_values, list):
                shap_vals_for_plot = shap_values[0]
            else:
                shap_vals_for_plot = shap_values

            # ---------- Summary Plot（サンプル） ----------
            st.write("### SHAP Summary Plot（サンプル）")
            # 余白を確保するために図を大きめにして左マージンを広げる
            fig_summary = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals_for_plot, X_shap, show=False)
            # 長い特徴名対策に左マージンを拡張（日本語ラベル向け）
            try:
                fig_summary.tight_layout()
            except Exception:
                pass
            # 経験則的に左を広げる（日本語の長い列名が切れるのを防ぐ）
            try:
                fig_summary.subplots_adjust(left=0.35)
            except Exception:
                pass
            st.pyplot(fig_summary)
            plt.close(fig_summary)

            # ---------- Bar Plot（平均絶対値） ----------
            st.write("### SHAP Bar Plot（平均絶対値）")
            # 特徴量数に応じて縦サイズを調整（多いほど縦長に）
            n_features = X_shap.shape[1] if hasattr(X_shap, "shape") else len(X_shap.columns)
            height = max(4, min(0.35 * n_features, 20))  # 上限をつける
            fig_bar = plt.figure(figsize=(10, height))
            shap.summary_plot(shap_vals_for_plot, X_shap, plot_type="bar", show=False)
            try:
                fig_bar.tight_layout()
            except Exception:
                pass
            try:
                fig_bar.subplots_adjust(left=0.35)
            except Exception:
                pass
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        except Exception as e:
            st.error(f"SHAP 計算中にエラーが発生しました: {e}")


# ---------------------------
# Candidate Prediction
# ---------------------------
elif page == "Candidate Prediction":
    st.title("Candidate Prediction — 新規候補の当落予測（SHAP で要因分析）")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_files.sort(reverse=True)
    selected_model = st.selectbox("モデルを選択（最新を選ぶ）", options=model_files)

    if not selected_model:
        st.info("まだ学習済みモデルがありません。まず Train Model で学習してください。")
    else:
        mdl = joblib.load(os.path.join(MODEL_DIR, selected_model))
        model = mdl["model"]
        features_used = mdl["features"]
        cbe = mdl.get("cbe", None)
        label_cols = mdl.get("label_cols", [])
        label_encoders = mdl.get("label_encoders", {})

        st.write("使用モデル:", selected_model)
        st.write("入力フォームに値を入れてください。")

        input_data = {}
        for f in features_used:
            if f.endswith("_cbe"):
                continue
            if f in ["年齢", "衆参当選回数", "参議院当選回数", "衆議院当選回数", "議席数", "政府規模"]:
                input_data[f] = st.number_input(f, value=0)
            elif f in ["性別"]:
                input_data[f] = st.selectbox(f, options=[0, 1])
            else:
                input_data[f] = st.text_input(f, value="")

        shap_individual = False
        if SHAP_AVAILABLE:
            shap_individual = st.checkbox("この候補者の SHAP force plot を表示する (要 shap)", value=False)
        else:
            st.write("※ SHAP がインストールされていません。個別要因は表示できません。")

        if st.button("予測する"):
            X_new = pd.DataFrame([input_data])

            # LabelEncoder 適用
            for col, le in label_encoders.items():
                if col in X_new.columns:
                    try:
                        X_new[col] = le.transform(X_new[col].fillna("nan").astype(str))
                    except Exception:
                        X_new[col] = 0

            # CBE 特徴作成
            if cbe is not None and label_cols:
                for col in label_cols:
                    if col in X_new.columns:
                        try:
                            transformed = cbe.transform(X_new[[col]])
                            X_new[f"{col}_cbe"] = transformed.iloc[:, 0]
                        except Exception:
                            X_new[f"{col}_cbe"] = 0
                    else:
                        X_new[f"{col}_cbe"] = 0

            # 整列
            X_new = X_new.reindex(columns=[c for c in features_used if not c.endswith("_cbe")] + [c for c in features_used if c.endswith("_cbe")], fill_value=0)
            X_new = X_new.reindex(columns=features_used, fill_value=0)

            try:
                prob_arr = model.predict(X_new)
                prob = float(prob_arr[0]) if hasattr(prob_arr, "__len__") else float(prob_arr)
            except Exception as e:
                st.error(f"モデル予測中にエラーが発生しました: {e}")
                st.write("入力データ（デバッグ）:")
                st.dataframe(X_new)
                st.stop()

            st.metric("当選確率", f"{prob:.3f}")
            st.write("閾値0.5判定:", "当" if prob >= 0.5 else "落")

            # 個別 SHAP 表示（オプション）
            if shap_individual and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_new)
                    if isinstance(shap_values, list):
                        sv = shap_values[0]
                    else:
                        sv = shap_values

                    st.subheader("この候補者の SHAP force plot（要 HTML 描画）")
                    # force_plot を HTML 化して埋め込む
                    force_html = shap.force_plot(explainer.expected_value, sv, X_new, matplotlib=False)
                    components.html(force_html.html(), height=300)
                except Exception as e:
                    st.error(f"個別 SHAP の作成でエラーが発生しました: {e}")
            elif shap_individual and not SHAP_AVAILABLE:
                st.warning("SHAP がインストールされていないため表示できません。")

# ---------------------------
# Party / Region Analysis
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
            try:
                summary = df.groupby(group_by).agg({"当落": ["mean", "count"]})
                summary.columns = ["当選確率", "候補数"]
                summary = summary.sort_values("当選確率", ascending=False)
                st.dataframe(summary.head(100))
                st.bar_chart(summary["当選確率"])
            except Exception as e:
                st.error(f"集計に失敗しました: {e}")

# ---------------------------
# Feature Analysis
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
        cat_cols = [c for c in df.columns if (df[c].dtype == object or df[c].nunique() < 30)]
        if len(cat_cols) == 0:
            st.info("カテゴリ列が見つかりません。")
        else:
            chosen_cat = st.selectbox("カテゴリ列を選ぶ", options=cat_cols)
            try:
                agg = df.groupby(chosen_cat)["当落"].mean().sort_values(ascending=False)
                st.dataframe(agg)
                st.bar_chart(agg)
            except Exception as e:
                st.error(f"カテゴリ別当選率の計算に失敗しました: {e}")
                st.write(df[chosen_cat].head())
                st.write(df["当落"].dtype)
                st.write(df["当落"].head())

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
# ---------------------------
# Newcomer Winning Patterns
# ---------------------------
elif page == "Newcomer Winning Patterns":
    st.title("Newcomer Winning Patterns — 新人の勝ちパターン分析")

    if df is None:
        st.info("データをアップロードしてください。")
        st.stop()

    # 元現新フィルタが存在するか確認
    if "元現新" not in df.columns:
        st.error("この分析には '元現新' 列が必要です。")
        st.stop()

    # 新人だけ抽出
    df_new = df[df["元現新"].astype(str).str.strip() == "新"].copy()
    if df_new.empty:
        st.error("新人データが存在しません。")
        st.stop()

    st.write(f"新人候補データ数: {len(df_new)} 名")

    # 分析対象となるカテゴリ列
    candidate_cols = [
        c for c in df_new.columns
        if (df_new[c].dtype == object or df_new[c].nunique() <= 20)
        and c not in ["当落", "元現新"]
    ]

    st.subheader("特徴量ごとの新人当選率（単変量）")

    # 単変量当選率ランキング
    result_list = []
    for col in candidate_cols:
        try:
            tmp = df_new.groupby(col)["当落"].mean().sort_values(ascending=False)
            result_list.append((col, tmp))
        except:
            continue

    # ランキングを表示
    for col, series in result_list:
        st.markdown(f"### 📌 {col} 別 当選率")
        st.dataframe(series)
        st.bar_chart(series)

    st.write("---")

    # 多変量の勝ちパターン（2変量組み合わせ）
    st.subheader("組み合わせ勝ちパターン（多変量：2つの特徴）")

    top_k = st.slider("表示する上位パターン数", 5, 50, 10)

    pattern_rows = []
    for col1 in candidate_cols:
        for col2 in candidate_cols:
            if col1 >= col2:
                continue
            try:
                grp = df_new.groupby([col1, col2])["当落"].mean()
                grp = grp.reset_index().sort_values("当落", ascending=False)
                pattern_rows.append((f"{col1} × {col2}", grp.head(top_k)))
            except:
                pass

    # 表示（多すぎるので 10組まで）
    for title, patt in pattern_rows[:10]:
        st.markdown(f"### 🔥 {title} の勝ちパターン（Top {top_k}）")
        st.dataframe(patt)
# app.py
# 当落予測ダッシュボード（SHAP 統合版）
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

# SHAP はオプション（インストールされていない環境でも動くよう try/except）
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

plt.rcParams["font.family"] = "MS Gothic"
sns.set_style("whitegrid")
shap_option = st.sidebar.checkbox("SHAPを表示する", key="show_shap")


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
    ["Overview", "Train Model", "Candidate Prediction",
     "Party / Region Analysis", "Feature Analysis", "Model Management"],
    key="page_select"
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
# Train Model
# ---------------------------
elif page == "Train Model":
    st.title("Train Model — 過学習 LightGBM モデル学習（SHAP オプション）")

    if df is None:
        st.info("データをアップロードしてからこちらで学習を実行してください。")
    else:
        # ラベルエンコード対象
        label_cols = ["党派", "元現新", "争点1位", "争点2位", "争点3位", "職業(分類)"]

        # 目的変数
        if "当落" in df.columns:
            target_col = "当落"
        else:
            target_col = st.sidebar.selectbox("目的変数を選択", df.columns)

        st.write("目的変数:", target_col)

        # 特徴量（あなたのモデルと同一）
        features = [
            "年齢", "性別", "党派", "元現新", "衆参当選回数",
            "参議院当選回数", "衆議院当選回数", "議席数",
            "争点1位", "争点2位", "争点3位",
            "政府規模", "出生地外立候補フラグ",
            "秘書経験フラグ", "地方議会経験フラグ",
            "職業(分類)"
        ]
        features = [f for f in features if f in df.columns]

        if st.button("学習開始（過学習 LightGBM）"):
            with st.spinner("前処理中..."):
                df_train = df.copy()

                X = df_train[features].copy()
                y = df_train[target_col].copy()

                # 当落 → 数値化
                y = y.map({"当": 1, "落": 0, 1: 1, 0: 0}).astype(int)

                # ============================
                # LabelEncoder（あなたのモデルと同一）
                # ============================
                encoders = {}
                for col in label_cols:
                    if col in X.columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le

                # ============================
                # CatBoostEncoder（KFold）
                # ============================
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cbe = CatBoostEncoder()

                X_cbe = np.zeros((len(X), len(label_cols)))
                valid_cols = [c for c in label_cols if c in X.columns]

                for tr_idx, va_idx in kf.split(X):
                    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                    y_tr = y.iloc[tr_idx]

                    cbe.fit(X_tr[valid_cols], y_tr)
                    X_cbe[va_idx, :] = cbe.transform(X_va[valid_cols]).values

                for i, col in enumerate(valid_cols):
                    X[f"{col}_cbe"] = X_cbe[:, i]

                # ============================
                # LightGBM（過学習設定）
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
                    "verbose": -1,
                    "seed": 42
                }

                train_data = lgb.Dataset(X, y)

            with st.spinner("LightGBM 学習中...（50000ラウンド）"):
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=50000
                )

            # ============================
            # 評価（学習データ）
            # ============================
            y_pred_prob = model.predict(X)
            y_pred = (y_pred_prob >= 0.5).astype(int)

            st.subheader("評価（学習データ）")
            st.write("混同行列")
            st.write(confusion_matrix(y, y_pred))

            st.write({
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1": f1_score(y, y_pred, zero_division=0)
            })

            # ============================
            # 特徴量重要度
            # ============================
            st.subheader("特徴量重要度")
            fig, ax = plt.subplots(figsize=(8, 10))
            lgb.plot_importance(model, ax=ax, max_num_features=30)
            st.pyplot(fig)

            # ============================
            # モデル保存
            # ============================
            save_path = f"models/overfit_lgb_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(
                {"model": model, "features": X.columns.tolist(), "encoders": encoders, "cbe": cbe},
                save_path
            )
            st.success(f"学習完了！モデルを保存しました: {save_path}")


 # -----------------------
# SHAP 計算＆表示（オプション）
# -----------------------
if shap_option and SHAP_AVAILABLE:
    with st.spinner("SHAP を計算中（データ量により数分かかることがあります）..."):
        try:
            # X_val をサンプリング（大きい場合は 500 行に制限）
            max_shap = 500
            if len(X_val) > max_shap:
                X_shap = X_val.sample(n=max_shap, random_state=42)
                st.write(f"SHAP はサンプル {max_shap} 行で計算します（表示はサンプル）")
            else:
                X_shap = X_val

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            # shap_values の型はモデル・バージョンで変わる場合がある。ndarray に揃える
            if isinstance(shap_values, list):
                shap_vals_for_plot = shap_values[0]
            else:
                shap_vals_for_plot = shap_values

            # ---------- Summary Plot（サンプル） ----------
            st.write("### SHAP Summary Plot（サンプル）")
            # 余白を確保するために図を大きめにして左マージンを広げる
            fig_summary = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals_for_plot, X_shap, show=False)
            # 長い特徴名対策に左マージンを拡張（日本語ラベル向け）
            try:
                fig_summary.tight_layout()
            except Exception:
                pass
            # 経験則的に左を広げる（日本語の長い列名が切れるのを防ぐ）
            try:
                fig_summary.subplots_adjust(left=0.35)
            except Exception:
                pass
            st.pyplot(fig_summary)
            plt.close(fig_summary)

            # ---------- Bar Plot（平均絶対値） ----------
            st.write("### SHAP Bar Plot（平均絶対値）")
            # 特徴量数に応じて縦サイズを調整（多いほど縦長に）
            n_features = X_shap.shape[1] if hasattr(X_shap, "shape") else len(X_shap.columns)
            height = max(4, min(0.35 * n_features, 20))  # 上限をつける
            fig_bar = plt.figure(figsize=(10, height))
            shap.summary_plot(shap_vals_for_plot, X_shap, plot_type="bar", show=False)
            try:
                fig_bar.tight_layout()
            except Exception:
                pass
            try:
                fig_bar.subplots_adjust(left=0.35)
            except Exception:
                pass
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        except Exception as e:
            st.error(f"SHAP 計算中にエラーが発生しました: {e}")


# ---------------------------
# Candidate Prediction
# ---------------------------
elif page == "Candidate Prediction":
    st.title("Candidate Prediction — 新規候補の当落予測（SHAP で要因分析）")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    model_files.sort(reverse=True)
    selected_model = st.selectbox("モデルを選択（最新を選ぶ）", options=model_files)

    if not selected_model:
        st.info("まだ学習済みモデルがありません。まず Train Model で学習してください。")
    else:
        mdl = joblib.load(os.path.join(MODEL_DIR, selected_model))
        model = mdl["model"]
        features_used = mdl["features"]
        cbe = mdl.get("cbe", None)
        label_cols = mdl.get("label_cols", [])
        label_encoders = mdl.get("label_encoders", {})

        st.write("使用モデル:", selected_model)
        st.write("入力フォームに値を入れてください。")

        input_data = {}
        for f in features_used:
            if f.endswith("_cbe"):
                continue
            if f in ["年齢", "衆参当選回数", "参議院当選回数", "衆議院当選回数", "議席数", "政府規模"]:
                input_data[f] = st.number_input(f, value=0)
            elif f in ["性別"]:
                input_data[f] = st.selectbox(f, options=[0, 1])
            else:
                input_data[f] = st.text_input(f, value="")

        shap_individual = False
        if SHAP_AVAILABLE:
            shap_individual = st.checkbox("この候補者の SHAP force plot を表示する (要 shap)", value=False)
        else:
            st.write("※ SHAP がインストールされていません。個別要因は表示できません。")

        if st.button("予測する"):
            X_new = pd.DataFrame([input_data])

            # LabelEncoder 適用
            for col, le in label_encoders.items():
                if col in X_new.columns:
                    try:
                        X_new[col] = le.transform(X_new[col].fillna("nan").astype(str))
                    except Exception:
                        X_new[col] = 0

            # CBE 特徴作成
            if cbe is not None and label_cols:
                for col in label_cols:
                    if col in X_new.columns:
                        try:
                            transformed = cbe.transform(X_new[[col]])
                            X_new[f"{col}_cbe"] = transformed.iloc[:, 0]
                        except Exception:
                            X_new[f"{col}_cbe"] = 0
                    else:
                        X_new[f"{col}_cbe"] = 0

            # 整列
            X_new = X_new.reindex(columns=[c for c in features_used if not c.endswith("_cbe")] + [c for c in features_used if c.endswith("_cbe")], fill_value=0)
            X_new = X_new.reindex(columns=features_used, fill_value=0)

            try:
                prob_arr = model.predict(X_new)
                prob = float(prob_arr[0]) if hasattr(prob_arr, "__len__") else float(prob_arr)
            except Exception as e:
                st.error(f"モデル予測中にエラーが発生しました: {e}")
                st.write("入力データ（デバッグ）:")
                st.dataframe(X_new)
                st.stop()

            st.metric("当選確率", f"{prob:.3f}")
            st.write("閾値0.5判定:", "当" if prob >= 0.5 else "落")

            # 個別 SHAP 表示（オプション）
            if shap_individual and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_new)
                    if isinstance(shap_values, list):
                        sv = shap_values[0]
                    else:
                        sv = shap_values

                    st.subheader("この候補者の SHAP force plot（要 HTML 描画）")
                    # force_plot を HTML 化して埋め込む
                    force_html = shap.force_plot(explainer.expected_value, sv, X_new, matplotlib=False)
                    components.html(force_html.html(), height=300)
                except Exception as e:
                    st.error(f"個別 SHAP の作成でエラーが発生しました: {e}")
            elif shap_individual and not SHAP_AVAILABLE:
                st.warning("SHAP がインストールされていないため表示できません。")

# ---------------------------
# Party / Region Analysis
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
            try:
                summary = df.groupby(group_by).agg({"当落": ["mean", "count"]})
                summary.columns = ["当選確率", "候補数"]
                summary = summary.sort_values("当選確率", ascending=False)
                st.dataframe(summary.head(100))
                st.bar_chart(summary["当選確率"])
            except Exception as e:
                st.error(f"集計に失敗しました: {e}")

# ---------------------------
# Feature Analysis
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
        cat_cols = [c for c in df.columns if (df[c].dtype == object or df[c].nunique() < 30)]
        if len(cat_cols) == 0:
            st.info("カテゴリ列が見つかりません。")
        else:
            chosen_cat = st.selectbox("カテゴリ列を選ぶ", options=cat_cols)
            try:
                agg = df.groupby(chosen_cat)["当落"].mean().sort_values(ascending=False)
                st.dataframe(agg)
                st.bar_chart(agg)
            except Exception as e:
                st.error(f"カテゴリ別当選率の計算に失敗しました: {e}")
                st.write(df[chosen_cat].head())
                st.write(df["当落"].dtype)
                st.write(df["当落"].head())

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
# ---------------------------
# Newcomer Winning Patterns
# ---------------------------
elif page == "Newcomer Winning Patterns":
    st.title("Newcomer Winning Patterns — 新人の勝ちパターン分析")

    if df is None:
        st.info("データをアップロードしてください。")
        st.stop()

    # 元現新フィルタが存在するか確認
    if "元現新" not in df.columns:
        st.error("この分析には '元現新' 列が必要です。")
        st.stop()

    # 新人だけ抽出
    df_new = df[df["元現新"].astype(str).str.strip() == "新"].copy()
    if df_new.empty:
        st.error("新人データが存在しません。")
        st.stop()

    st.write(f"新人候補データ数: {len(df_new)} 名")

    # 分析対象となるカテゴリ列
    candidate_cols = [
        c for c in df_new.columns
        if (df_new[c].dtype == object or df_new[c].nunique() <= 20)
        and c not in ["当落", "元現新"]
    ]

    st.subheader("特徴量ごとの新人当選率（単変量）")

    # 単変量当選率ランキング
    result_list = []
    for col in candidate_cols:
        try:
            tmp = df_new.groupby(col)["当落"].mean().sort_values(ascending=False)
            result_list.append((col, tmp))
        except:
            continue

    # ランキングを表示
    for col, series in result_list:
        st.markdown(f"### 📌 {col} 別 当選率")
        st.dataframe(series)
        st.bar_chart(series)

    st.write("---")

    # 多変量の勝ちパターン（2変量組み合わせ）
    st.subheader("組み合わせ勝ちパターン（多変量：2つの特徴）")

    top_k = st.slider("表示する上位パターン数", 5, 50, 10)

    pattern_rows = []
    for col1 in candidate_cols:
        for col2 in candidate_cols:
            if col1 >= col2:
                continue
            try:
                grp = df_new.groupby([col1, col2])["当落"].mean()
                grp = grp.reset_index().sort_values("当落", ascending=False)
                pattern_rows.append((f"{col1} × {col2}", grp.head(top_k)))
            except:
                pass

    # 表示（多すぎるので 10組まで）
    for title, patt in pattern_rows[:10]:
        st.markdown(f"### 🔥 {title} の勝ちパターン（Top {top_k}）")
        st.dataframe(patt)
