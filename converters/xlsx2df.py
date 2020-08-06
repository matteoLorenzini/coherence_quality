import pandas as pd

df = pd.read_excel ('excel/coherence.ods.xlsx', sheet_name='VAW', skiprows='1')

#df = df.dropna()

print(df)

df.to_csv('vaw.tsv', sep = '\t')
df.to_excel ('archeo_dataset.xlsx', index = None, header=True)