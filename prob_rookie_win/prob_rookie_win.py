import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer


def load_data():
    return pd.read_excel(
        "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
        engine="openpyxl"
    )


def preprocess(df):
    df = df[df["元現新"] == "新"].copy()

    # 当落の正規化
    df["当落"] = df["当落"].astype(str).str.strip()
    df.loc[df["当落"].str.contains("当", na=False), "当落"] = 1
    df.loc[df["当落"].str.contains("落", na=False), "当落"] = 0
    df["当落"] = pd.to_numeric(df["当落"], errors="coerce")
    df = df.dropna(subset=["当落"])
    df["当落"] = df["当落"].astype(int)

    # カテゴリ列を文字列化して欠損を「欠損」に置換
    categorical_features = [
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
        "職業(分類)",
    ]
    for col in categorical_features:
        df[col] = df[col].astype(str).fillna("欠損")

    return df



def build_pipeline():
    numeric_features = [
        "年齢",
        "衆参すべての当選回数",
        "参議院当選回数",
        "衆議院当選回数",
    ]

    categorical_features = [
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
        "職業(分類)",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="欠損")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, numeric_features, categorical_features


def analyze(df):
    X = df.drop(columns=["当落", "候補者氏名", "元現新"])
    y = df["当落"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    pipeline, numeric_features, categorical_features = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("=== 分類性能 ===")
    print(classification_report(y_test, pipeline.predict(X_test)))

    # 特徴量名取得
    ohe = (
        pipeline
        .named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    feature_names = (
        numeric_features
        + list(ohe.get_feature_names_out(categorical_features))
    )

    coef = pipeline.named_steps["model"].coef_[0]
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": coef})
        .sort_values("coefficient", ascending=False)
    )

    coef_df.to_csv("rookie_win_factor_coefficients.csv", index=False)

    print("\n=== 新人当選にプラスに働く要因（上位） ===")
    print(coef_df.head(15))

    print("\n=== 新人当選にマイナスに働く要因（下位） ===")
    print(coef_df.tail(15))


def main():
    df = load_data()
    df = preprocess(df)
    analyze(df)


if __name__ == "__main__":
    main()
