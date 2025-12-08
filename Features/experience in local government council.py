import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

sns.set(font=["Meiryo"])

# データ読み込み
df = pd.read_excel(
    "C:/Users/frontier-Python/Desktop/2025_upperhouse_election_constituency_system_predictor/Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# 当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# 地方議会経験ごとの当選率を計算
# 0=地方議会経験あり, 1=地方議会経験なし
win_rate_by_local_council = df.groupby("地方議会経験の有無(0が地方議会経験あり、1が地方議会経験なし)")["当落フラグ"].mean().sort_index()

# グラフ描画
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x=win_rate_by_local_council.index,
    y=win_rate_by_local_council.values,
    palette="coolwarm",
    ax=ax
)

# y軸をパーセント表示
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)

# x軸ラベルをわかりやすくする
ax.set_xticklabels(["地方議会経験あり", "地方議会経験なし"])
ax.set_xlabel("地方議会経験の有無", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 棒グラフの値ラベルを追加
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=10)

# タイトル設定
plt.title("地方議会経験の有無と当選確率の関係", fontsize=14, pad=20)

# レイアウト調整
plt.tight_layout()
plt.show()
