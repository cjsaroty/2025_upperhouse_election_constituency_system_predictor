import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set(font=["Meiryo"])

# ==================================================
# 1. データ読み込み
# ==================================================
file_path = "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
df = pd.read_excel(file_path)

# ==================================================
# 2. クロス集計（職業 × 当落）
# ==================================================
job_outcome = pd.crosstab(df["職業(分類)"], df["当落"])

# 「その他」を最後に回す
order = [job for job in job_outcome.index if job != "その他"] + ["その他"]
job_outcome = job_outcome.reindex(order)

print("\n▼ クロス集計（職業 × 当落）")
print(job_outcome)

# ==================================================
# 3. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(job_outcome)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（職業と当落は独立ではない）")
else:
    print("→ 有意差なし（職業と当落は独立とみなせる）")

# 期待度数の確認（重要）
expected_df = pd.DataFrame(
    expected,
    index=job_outcome.index,
    columns=job_outcome.columns
)

print("\n▼ 期待度数（5未満が多い場合は解釈に注意）")
print(expected_df.round(2))

# ==================================================
# 4. 当選率の計算
# ==================================================
job_outcome["当選率"] = job_outcome["当"] / job_outcome.sum(axis=1)

# ==================================================
# 5. 当選率の棒グラフ
# ==================================================
plt.figure(figsize=(12, 6))

sns.barplot(
    x=job_outcome.index,
    y=job_outcome["当選率"]
)

plt.title("職業ごとの当選率")
plt.xlabel("職業")
plt.ylabel("当選率")
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{int(y * 100)}%")
)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()
