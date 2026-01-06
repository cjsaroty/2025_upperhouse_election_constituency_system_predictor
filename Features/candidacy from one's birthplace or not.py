import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import os
from scipy.stats import chi2_contingency

# --- グラフの日本語フォント設定 ---
sns.set(font=["Meiryo"])

# --- データ読み込み（相対パス） ---
file_path = "Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} が見つかりません。パスを確認してください。")

df = pd.read_excel(file_path, engine="openpyxl")

# --- 当落を数値化（当選=1, 落選=0） ---
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# --- 列名短縮（出生地外立候補フラグ） ---
df = df.rename(columns={"出生地外立候補フラグ": "出生地外立候補"})

# --- 出生地から立候補ごとの当選率計算 ---
win_rate_by_birthplace = df.groupby("出生地外立候補")["当落フラグ"].mean().sort_index()

# --- カイ二乗検定 ---
contingency_table = pd.crosstab(df["出生地外立候補"], df["当落フラグ"])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("=== カイ二乗検定結果 ===")
print("クロス集計表:\n", contingency_table)
print(f"カイ二乗統計量 = {chi2:.3f}")
print(f"p値 = {p:.3f}")
print(f"自由度 = {dof}")
print("期待度数:\n", expected)
if p < 0.05:
    print("→ 出生地からの立候補と当選には有意な関係があります（p<0.05）")
else:
    print("→ 出生地からの立候補と当選に有意な関係はありません（p≥0.05）")

# --- グラフ描画 ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x=win_rate_by_birthplace.index,
    y=win_rate_by_birthplace.values,
    palette="coolwarm",
    ax=ax
)

# --- y軸をパーセント表示 ---
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)

# --- x軸ラベル設定 ---
ax.set_xticks([0, 1])
ax.set_xticklabels(["出生地から立候補", "出生地以外から立候補"])
ax.set_xlabel("出生地からの立候補の有無", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# --- 棒グラフに値ラベル追加 ---
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=10)

# --- タイトルにp値を表示 ---
plt.title(f"出生地からの立候補と当選確率の関係", fontsize=14, pad=20)

plt.tight_layout()
plt.show()
