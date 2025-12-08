import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

sns.set(font=["Meiryo"])

# データ読み込み
file_path = r"C:\Users\owner\OneDrive\デスクトップ\2025_upperhouse_election_predictor\Data\2025_upperhouse_election_constituency_system_cleaning.xlsx"

df = pd.read_excel(file_path)

# 当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# 「大きな政府か小さな政府か」列の名前を仮に '政府タイプ' とする
# 1:小さな政府, 2:どちらかといえば小さな政府, 3:どちらともいえない, 4:どちらかといえば大きな政府, 5:大きな政府
# 既に数値化されている前提
# 過去当選回数ではなく、ここでは政府タイプごとの当選確率を計算
win_rate_by_government = df.groupby("大きな政府か小さな政府か(1に近いほど小さな政府/5に近いほど大きな政府)")["当落フラグ"].mean().sort_index()

# グラフ描画
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=win_rate_by_government.index, y=win_rate_by_government.values, palette="viridis", ax=ax)

# x軸ラベルを具体的に
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
ax.set_xlabel("政府の規模感", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 棒グラフの値ラベルを追加
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

# タイトル設定
plt.title("政府規模感と当選確率の関係", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
