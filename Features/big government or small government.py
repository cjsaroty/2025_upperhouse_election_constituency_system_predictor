import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency

sns.set(font=["Meiryo"])

# ▼ データ読み込み
file_path = "Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
df = pd.read_excel(file_path)

# ▼ 当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# ▼ 政府タイプ列名を設定
government_col = "政府規模"

# ▼ カイ二乗検定
# クロス集計表（政府タイプ × 当落）
contingency = pd.crosstab(df[government_col], df["当落フラグ"])
print("\n▼ クロス集計表（政府規模 × 当落）")
print(contingency)

# χ²検定
chi2, p, dof, expected = chi2_contingency(contingency)
print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（政府規模と当選は独立ではない）")
else:
    print("→ 有意差なし（政府規模と当選は独立とみなせる）")

# ▼ 政府規模ごとの当選確率
win_rate_by_government = df.groupby(government_col)["当落フラグ"].mean().sort_index()

# グラフ描画
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x=win_rate_by_government.index, y=win_rate_by_government.values, palette="viridis", ax=ax)

# x軸ラベルを具体的に（必要に応じて変更）
ax.set_xticklabels([
    "小さな政府に近い",
    "どちらかといえば小さな政府",
    "どちらともいえない",
    "どちらかといえば大きな政府",
    "大きな政府に近い"
], rotation=0, fontsize=10)

# y軸をパーセント表示
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)
ax.set_xlabel("大きな政府か小さな政府か", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 棒グラフの値ラベルを追加
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

# タイトル設定
plt.title("政府の大きさの志向性と当選確率の関係", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
