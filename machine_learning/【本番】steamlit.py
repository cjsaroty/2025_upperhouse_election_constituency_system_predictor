import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "MS Gothic"


# ==========================================================
# Streamlit アプリ開始
# ==========================================================
def main():

    st.title("🌸 2025参院選・当落予測アプリ（LightGBM + CBE）")
    st.write("Excel データを読み込んで当落予測モデルを構築します。")

    st.sidebar.header("設定")

    uploaded_file = st.sidebar.file_uploader("Excel ファイルをアップロード", type=["xlsx"])

    if uploaded_file is None:
        st.info("左のサイドバーから Excel をアップロードしてください。")
        return

    # ==========================================================
    # データ読み込み
    #==========================================================
    df = pd.read_excel(uploaded_file, engine="openpyxl")
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

    st.subheader("📊 データプレビュー")
    st.dataframe(df.head())

    # ==========================================================
    # ラベルエンコード
    #==========================================================
    label_cols = ["党派", "元現新", "争点1位", "争点2位", "争点3位", "職業(分類)"]
    encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df["当落"] = df["当落"].map({"当": 1, "落": 0})

    # ==========================================================
    # 特徴量
    #==========================================================
    features = [
        "年齢", "性別", "党派", "元現新", "衆参当選回数",
        "参議院当選回数", "衆議院当選回数", "議席数",
        "争点1位", "争点2位", "争点3位",
        "政府規模", "出生地外立候補フラグ",
        "秘書経験フラグ", "地方議会経験フラグ",
        "職業(分類)"
    ]

    X = df[features].copy()
    y = df["当落"].copy()

    # ==========================================================
    # CatBoost Encoding（KFold leak防止）
    #==========================================================
    st.sidebar.write("---")
    st.sidebar.subheader("学習設定")

    if st.sidebar.button("学習を実行する"):

        with st.spinner("エンコーディング中..."):
            cbe = CatBoostEncoder()
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            X_cbe = np.zeros((len(X), len(label_cols)))
            for tr_idx, va_idx in kf.split(X):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr = y.iloc[tr_idx]
                cbe.fit(X_tr[label_cols], y_tr)
                X_cbe[va_idx, :] = cbe.transform(X_va[label_cols]).values

            for i, col in enumerate(label_cols):
                X[f"{col}_cbe"] = X_cbe[:, i]

        # ==========================================================
        # 過学習 LightGBM
        #==========================================================
        params_overfit = {
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
            "seed": 42,
        }

        train_data = lgb.Dataset(X, y)

        with st.spinner("LightGBM 学習中...（50,000 ラウンド）"):
            model = lgb.train(
                params_overfit,
                train_data,
                num_boost_round=50000
            )

        # ==========================================================
        # 予測
        #==========================================================
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # ==========================================================
        # 評価
        #==========================================================
        st.subheader("📈 評価結果（学習データ）")

        cm = confusion_matrix(y, y_pred)
        st.write("### 混同行列")
        st.write(cm)

        st.write("### 指標")
        st.write(f"Accuracy : {accuracy_score(y, y_pred):.3f}")
        st.write(f"Precision: {precision_score(y, y_pred):.3f}")
        st.write(f"Recall   : {recall_score(y, y_pred):.3f}")
        st.write(f"F1-score : {f1_score(y, y_pred):.3f}")

        # ==========================================================
        # 特徴量重要度
        #==========================================================
        st.subheader("🔍 特徴量重要度（Feature Importance）")
        fig, ax = plt.subplots(figsize=(8, 12))
        lgb.plot_importance(model, ax=ax, max_num_features=30)
        st.pyplot(fig)

        st.success("学習が完了しました！")


if __name__ == "__main__":
    main()
