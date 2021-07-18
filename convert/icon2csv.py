# importing panda library 
import pandas as pd
import numpy as np
import glob 

# readinag given csv file 

path = r'../iconclass' # use your path
all_files = glob.glob(path + "/*.txt")

li = []

for filename in all_files:
    df = pd.read_csv(filename, delimiter="|",  names=["id", "Iconografia"])
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame['id_1']= frame['id'].str[:1]

def assign_label(i):
    if i == "0":
        return 'arte_astratta'
    if i == "1":
        return 'Religione_e_Magia'
    if i == "2":
        return 'Natura'
    if i == "3":
        return 'Essere_umano_uomo_in_generale'
    if i == "4":
        return 'Societa_civilizzazione_cultura'
    if i == "5":
        return 'Idee_e_concetti_astratti'
    if i == "6":
        return 'Storia'
    if i == "7":
        return 'Bibbia_storie_dal_Vecchio_e_dal_Nuovo_Testamento'
    if i == "8":
        return 'Letteratura'
    if i == "9":
        return 'Mitologia_classica_e_storia_antica'
    

frame['Tema'] = frame['id_1'].apply(assign_label)

print(frame)

header = ["Iconografia",  "Tema"]
# storing this dataframe in a csv file 
frame.to_csv('icon.csv', 
				index = None, 
                columns = header,
                sep = ';') 
