import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

sns.set(font=["Meiryo"])

df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

#当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

#党派ごとの当選率
party_win_rate = df.groupby("党派")["当落フラグ"].mean().sort_values(ascending=False)

#グラフの土台を作成
fig, ax1 = plt.subplots(figsize=(12, 6))

#棒グラフを描画（当選確率）
sns.barplot(x=party_win_rate.index, y=party_win_rate.values, palette="viridis", ax=ax1)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax1.set_ylim(0, 1)
ax1.set_ylabel("当選確率 (%)", fontsize=12)

#棒グラフの値ラベルを付記
for container in ax1.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax1.bar_label(container, labels=labels, padding=2, fontsize=9)

#タイトルと横軸ラベル
plt.title("党派別当選確率", fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

#グラフを表示
plt.tight_layout()
plt.show()