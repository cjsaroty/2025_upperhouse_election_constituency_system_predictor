import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font=["Meiryo"])

df = pd.read_excel("../Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

#当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

#議席数ごとの当選率を計算
seat_win_rate = df.groupby('議席数')["当落フラグ"].mean() * 100 

#棒グラフ（横長のサイズ）
plt.figure(figsize=(14,5))  # 横長に変更
sns.barplot(x=seat_win_rate.index, y=seat_win_rate.values, palette="viridis")
plt.title('議席数ごとの当選確率', fontsize=14)
plt.xlabel('議席数')
plt.ylabel('当選確率 (%)')
plt.ylim(0,100)

#棒の上に当選率ラベルを表示
for i, v in enumerate(seat_win_rate.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.show()
