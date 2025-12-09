import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font=["Meiryo"])

# データ読み込み
df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")
# 政党名をわかりやすく変換（例：略称→フルネーム）
party_name_mapping = {
    "自民": "自由民主党",
    "立民": "立憲民主党",
    "公明": "公明党",
    "共産": "日本共産党",
    "維新": "日本維新の会",
    "国民": "国民民主党",
    "れいわ": "れいわ新選組",
    "社民": "社会民主党",
    "N党": "NHK党",
    "その他": "その他"
}

df["党派"] = df["党派"].map(lambda x: party_name_mapping.get(x, x))

# 政府タイプと政党ごとの人数を集計
party_count_by_government = df.groupby(["大きな政府か小さな政府か(1に近いほど小さな政府/5に近いほど大きな政府)", "党派"]).size().reset_index(name="人数")

# グラフ描画
plt.figure(figsize=(12, 6))
sns.barplot(
    x="大きな政府か小さな政府か(1に近いほど小さな政府/5に近いほど大きな政府)",
    y="人数",
    hue="党派",
    data=party_count_by_government,
    palette="Set2"
)

# X軸ラベルをわかりやすく
plt.xticks(
    [0, 1, 2, 3, 4],
    [
        "小さな政府に近い",
        "どちらかといえば小さな政府",
        "どちらともいえない",
        "どちらかといえば大きな政府",
        "大きな政府に近い"
    ],
    rotation=0
)

plt.xlabel("政府の規模感", fontsize=12)
plt.ylabel("人数", fontsize=12)
plt.title("政府規模感と所属政党の関係", fontsize=14, pad=20)
plt.legend(title="党派", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



