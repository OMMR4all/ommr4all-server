import pandas as pd

df = pd.read_excel("/tmp/eval_data.xls", sheet_name="Text Docs Line Ignored", header=3)
df2 = pd.read_excel("/tmp/eval_data.xls", sheet_name="Syllable Docs", header=3)

s = df.loc[df['Zeilen Fehler'] != 0]
t = df2.loc[df2['Doc_id'].isin(s['Doc_id'])]
t.to_excel("/tmp/wrong_sylls.xlsx")