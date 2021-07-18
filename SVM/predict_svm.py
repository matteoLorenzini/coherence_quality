import pandas as pd
import pickle

from sklearn import svm
import numpy as np

r_filenameTSV = "../input_data/vaw_dataset_coherence.tsv"

tsv_read = pd.read_csv(r_filenameTSV, sep='\t', names=["vector"])

df = pd.DataFrame(tsv_read)

df = pd.DataFrame(df.vector.str.split(" ", 1).tolist(), columns=['manual_label', 'vector'])

y = pd.DataFrame([df.manual_label]).astype(int).to_numpy().reshape(-1, 1).ravel()
print(y.shape)

X = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df['vector']])
print(X.astype(float).to_numpy())

# here the baseline
baseline = ['1','9','4']

gold = ['1','2','3','4','5','6','7','8','9']

predictions=[]
with open('model.pkl', 'rb') as f:     
        clf = pickle.load(f)
        
for line in X:
        output=clf.predict(X)
        output_prob=pd.DataFrame(clf.predict_proba(X),columns=clf.classes_)

        output_prob = pd.DataFrame(output_prob.columns.values[np.argsort(-output_prob.values, axis=1)[:, :3]], 
                  index=df.index,
                  columns = ['label_1','label_2','label_3']).reset_index()
     
df['prediction'] = output
#print(df)
#print(output_prob)

final = pd.concat([df[['manual_label', 'prediction']], output_prob], axis=1)      


final['baseline_1'] = '1'
final['baseline_2'] = '9'
final['baseline_3'] = '4'

print(final)
final = final.astype(int)

baseline_p1=0
baseline_p3=0
pred_p1=0
pred_p3=0

rows=final.shape[0]
for i in range(rows):
    manual_is_base1 = final.manual_label == final.baseline_1
    manual_is_base2 = final.manual_label == final.baseline_2
    manual_is_base3 = final.manual_label == final.baseline_3
    manual_is_anybase = manual_is_base1 | manual_is_base2 | manual_is_base3

    manual_is_pred1 = final.manual_label == final.label_1
    manual_is_pred2 = final.manual_label == final.label_2
    manual_is_pred3 = final.manual_label == final.label_3
    manual_is_anypred = manual_is_pred1 | manual_is_pred2 | manual_is_pred3

    baseline_p1 = sum(manual_is_base1)
    baseline_p3 = sum(manual_is_anybase)

    pred_p1 = sum(manual_is_pred1)
    pred_p3 = sum(manual_is_anypred)

print("Raw counts: B@1={} B@3={} P@1={} P@3={}",baseline_p1,baseline_p3,pred_p1,pred_p3)
print("Percentages: B@1={} B@3={} P@1={} P@3={}",baseline_p1/rows,baseline_p3/rows,pred_p1/rows,pred_p3/rows)
