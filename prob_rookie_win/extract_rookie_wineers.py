import pandas as pd

# ============================
# データ読み込み
# ============================
input_path = "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
output_path = "./Data/rookie_winners.xlsx"

df = pd.read_excel(input_path, engine="openpyxl")
df.columns = df.columns.str.strip()

# ============================
# 新人かつ当選者を抽出
# ============================
rookie_winners = df[
    (df["元現新"] == "新") &
    (df["当落"] == "当")
].copy()

# ============================
# 並び替え
# ============================
# 都道府県 → 年齢順
if "都道府県" in rookie_winners.columns:
    rookie_winners = rookie_winners.sort_values(
        by=["都道府県", "年齢"],
        ascending=[True, True]
    )

# ============================
# 出力
# ============================
rookie_winners.to_excel(output_path, index=False)

print("新人当選者の抽出が完了しました")
print(f"件数：{len(rookie_winners)}")
print(f"出力先：{output_path}")
