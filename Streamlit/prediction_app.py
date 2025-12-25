# ============================
# Streamlit 当落予測アプリ
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

st.set_page_config(page_title="参議院選挙 当落予測", layout="centered")

# ============================
# モデル読み込み
# ============================
MODEL_PATH = "../machine_learning/model_package.pkl"

mdl = joblib.load(MODEL_PATH)

model = mdl["model"]
features = mdl["features"]
label_encoders = mdl["label_encoders"]
cbe_cols = mdl["cbe_cols"]
cbe = mdl["cbe"]

# ============================
# UI
# ============================
st.title("🗳 参議院選挙 当落予測")
st.write("すべての項目を入力すると、当選確率を予測します。")

# ============================
# 入力フォーム
# ============================
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("年齢", 25, 90, 50)
        gender = st.selectbox("性別", [0, 1])  # 0:男性 1:女性
        party = st.selectbox("党派", label_encoders["党派"].classes_)
        status = st.selectbox("元現新", label_encoders["元現新"].classes_)
        total_win = st.number_input("衆参当選回数", 0, 20, 0)
        house_win = st.number_input("参議院当選回数", 0, 10, 0)
        lower_win = st.number_input("衆議院当選回数", 0, 10, 0)

    with col2:
        seats = st.number_input("議席数", 1, 50, 1)
        issue1 = st.selectbox("争点1位", label_encoders["争点1位"].classes_)
        issue2 = st.selectbox("争点2位", label_encoders["争点2位"].classes_)
        issue3 = st.selectbox("争点3位", label_encoders["争点3位"].classes_)
        gov = st.slider("政府規模（1:小さな政府 / 5:大きな政府）", 1, 5, 3)
        birth = st.selectbox("出生地外立候補", [0, 1])
        secretary = st.selectbox("秘書経験なし", [0, 1])
        local = st.selectbox("地方議会経験なし", [0, 1])
        job = st.selectbox("職業(分類)", label_encoders["職業(分類)"].classes_)

    submit = st.form_submit_button("予測する")

# ============================
# 予測処理
# ============================
if submit:

    # 1行DataFrame作成
    X_new = pd.DataFrame([{
        "年齢": age,
        "性別": gender,
        "党派": party,
        "元現新": status,
        "衆参当選回数": total_win,
        "参議院当選回数": house_win,
        "衆議院当選回数": lower_win,
        "議席数": seats,
        "争点1位": issue1,
        "争点2位": issue2,
        "争点3位": issue3,
        "政府規模": gov,
        "出生地外立候補フラグ": birth,
        "秘書経験フラグ": secretary,
        "地方議会経験フラグ": local,
        "職業(分類)": job
    }])

    # ============================
    # LabelEncoder
    # ============================
    for col, le in label_encoders.items():
        X_new[col] = le.transform(X_new[col].astype(str))

    # ============================
    # CatBoostEncoder（★修正点）
    # ============================
    X_cbe = cbe.transform(X_new[cbe_cols])

    for col in cbe_cols:
        X_new[f"{col}_cbe"] = X_cbe[col].values

    # ============================
    # 特徴量順・型合わせ
    # ============================
    X_new = X_new.reindex(columns=features, fill_value=0)
    X_new = X_new.astype(float)

    # ============================
    # 予測
    # ============================
    prob = float(model.predict(X_new)[0])

    st.subheader("📊 予測結果")
    st.metric("当選確率", f"{prob*100:.1f} %")

    if prob >= 0.5:
        st.success("✅ 当選の可能性が高い")
    else:
        st.error("❌ 落選の可能性が高い")
