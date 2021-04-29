import pandas as pd
import fasttext as ft
import numpy as np
import re
from numpy import nan


# here you load the csv into pandas dataframe
df=pd.read_csv('../input_data/vaw_dataset_coherence.csv')

# here you load your fasttext module
model=ft.load_model('../model/icon_wikipedia_.bin')

# here the baseline
baseline = ['__label__Religione_e_Magia','__label__Mitologia_classica_e_storia_antica','__label__Societa_civilizzazione_cultura']

gold = ['__label__Religione_e_Magia','__label__Natura','__label__Essere_umano_uomo_in_generale','__label__Societa_civilizzazione_cultura','__label__Idee_e_concetti_astratti','__label__Storia','__label__Bibbia_storie_dal_Vecchio_e_dal_Nuovo_Testamento','__label__Letteratura','__label__Mitologia_classica_e_storia_antica']

# line by line, you make the predictions and store them in a list
predictions=[]
for line in df['subject']:
    pred_label=model.predict(line, k=6, threshold=0.001)  
    predictions.append(pred_label)

# you add the list to the dataframe, then save the datframe to new csv
df[['prediction','value']]=predictions

df = df.applymap(str)

df['prediction'].str.replace(r"\(.*\)","")

df['value'] = df['value'].str.replace(' ', ', ')

# new data frame with split value columns 
df[['label_1','label_2','label_3','label_4','label_5','label_6']] = df.prediction.str.split("', '", n = 7, expand = True)
#df[['label_1','label_2','label_3']] = df.prediction.str.split("', '", n = 4, expand = True)

df.fillna("no_prediction", inplace = True)

# Only the  

argument_cols = ['label_1','label_2','label_3']

boolean_idx = df[argument_cols].apply(
    lambda arg_column: df['manual_label'].combine(arg_column, lambda token, arg: token in arg)
)

selected_vals = df[argument_cols][boolean_idx]
selected_vals = selected_vals.replace(np.nan, '', regex=True)
selected_vals = selected_vals.applymap(str)
df['suggested_label'] = selected_vals["label_1"].astype(str) + selected_vals["label_2"]+ selected_vals["label_3"]


df = df.replace(r'^\s*$', np.nan, regex=True)
df.loc[df['suggested_label'].isnull(),'suggested_label'] = "Null"#df['label_1']

df['P@1'] = df.apply(lambda x: x.manual_label in x.label_1, axis=1)
df["P@1"] = df["P@1"].astype(int)
df["P@3"] = df['suggested_label'].apply(lambda x: 1 if any(i in x for i in gold) else 0)


df['baseline@1'] = df.apply(lambda x: x.manual_label in x.baseline_1, axis=1)
df["baseline@1"] = df["baseline@1"].astype(int)

df["baseline@3"] = df['manual_label'].apply(lambda x: 1 if any(i in x for i in baseline) else 0)
output = df[['subject','manual_label','suggested_label','P@1','P@3','baseline@1','baseline@3']]

print(output)

# Results 

print(df['baseline@1'].value_counts(normalize=True).mul(100).astype(str)+'%')
print(df['baseline@3'].value_counts(normalize=True).mul(100).astype(str)+'%')
print(df['P@1'].value_counts(normalize=True).mul(100).astype(str)+'%')
print(df['P@3'].value_counts(normalize=True).mul(100).astype(str)+'%')


output.to_csv('csv_file_new_pred.csv',sep=',',index=False)

df.to_csv('csv_file_df_pred.csv',sep=',',index=False)
