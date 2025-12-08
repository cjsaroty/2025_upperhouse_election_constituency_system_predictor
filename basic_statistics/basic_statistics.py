import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

sns.set(font=["Meiryo"])

df = pd.read_excel("../Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

#基本統計量の確認
desc = df.describe(include='all')
desc = desc.round(2).fillna("–")

# プロット用に設定
fig, ax = plt.subplots(figsize=(12, desc.shape[0]*0.5))
ax.axis('off')  # 軸は非表示

# テーブルとして描画
tbl = ax.table(cellText=desc.round(2).values, 
            colLabels=desc.columns, 
            rowLabels=desc.index, 
            cellLoc='center', 
            loc='center')

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(desc.columns))))

plt.tight_layout()
plt.show()