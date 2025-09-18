from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font=["Meiryo"])

# データ読み込み
df = pd.read_excel("sangiinn_2025_cleaning.xlsx", engine="openpyxl")

# ラベルエンコーディング
le_party = LabelEncoder()
df['党派'] = le_party.fit_transform(df['党派'])

le_status = LabelEncoder()
df['元現新'] = le_status.fit_transform(df['元現新'])

le_job = LabelEncoder()
df['職業(分類)'] = le_job.fit_transform(df['職業(分類)'])

df['当落'] = df['当落'].map({'当': 1, '落': 0})

X = df[['年齢', '性別', '党派', '元現新', '議席数', '職業(分類)']]
y = df['当落']

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 評価関数
def evaluate_model(name, y_test, y_pred):
    print(f"\n*** {name} 評価 ***")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")

# ロジスティック回帰
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
evaluate_model("ロジスティック回帰", y_test, y_pred_lr)

# ランダムフォレスト（グリッドサーチ）
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',  
    n_jobs=1
)

rf_grid.fit(X_train, y_train)

best_rf_model = rf_grid.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
evaluate_model("ランダムフォレスト (グリッドサーチ)", y_test, y_pred_rf)

# ランダムフォレスト特徴量重要度
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(8,6))
plt.title("ランダムフォレスト特徴量重要度")
plt.show()

# ユーザー入力による当選確率予測（ランダムフォレスト使用）
print("\n*** 当選確率予測アプリ ***")
name = input("名前を入力してください: ")
age = int(input("年齢を入力してください: "))
gender = int(input("性別を入力してください（男性=0, 女性=1）: "))
party = input(f"党派を入力してください {list(le_party.classes_)}: ")
status = input(f"元現新を入力してください {list(le_status.classes_)}: ")
seats = int(input("議席数を入力してください: "))
job = input(f"職業を入力してください {list(le_job.classes_)}: ")

try:
    party_encoded = le_party.transform([party])[0]
    status_encoded = le_status.transform([status])[0]
    job_encoded = le_job.transform([job])[0]

    X_input = np.array([[age, gender, party_encoded, status_encoded, seats, job_encoded]])
    
    prob = best_rf_model.predict_proba(X_input)[0][1]
    pred_label = "当選" if prob >= 0.5 else "落選"
    
    print(f"\n{name}さんの当選確率は {prob*100:.2f}% → 予測ラベル: {pred_label}")
    print("（使用モデル: ランダムフォレスト）")

except ValueError:
    print("入力した文字列が学習データに存在しません。正しい値を入力してください。")
