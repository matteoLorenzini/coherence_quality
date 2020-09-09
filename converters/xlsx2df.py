import pandas as pd

df = pd.read_excel ('../excel/test.xlsx', sheet_name='VAW_labelled', skiprows='1')

#df = df.dropna()

print(df)

df.to_csv('../inputFile/vaw_labelled.tsv', sep = '\t',index=False)
#df.to_excel ('archeo_dataset.xlsx', index = None, header=True)