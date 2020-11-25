import pandas as pd
import fasttext as ft

# here you load the csv into pandas dataframe
df=pd.read_csv('coherence.csv')

# here you load your fasttext module
model=ft.load_model('icon.bin')

# line by line, you make the predictions and store them in a list
predictions=[]
for line in df['subject']:
    pred_label=model.predict(line, k=-1, threshold=0.5) 
    predictions.append(pred_label)

# you add the list to the dataframe, then save the datframe to new csv
df[['prediction','value']]=predictions
print(df)
df.to_csv('csv_file_w_pred.csv',sep=',',index=False)