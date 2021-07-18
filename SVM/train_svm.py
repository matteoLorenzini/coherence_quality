import pandas as pd
import pickle
from time import time
from sklearn import svm


r_filenameTSV = "../input_data/iconclas.tsv"


#DF 300 dimension start

tsv_read = pd.read_csv(r_filenameTSV, sep='\t', names=["vector"])

df = pd.DataFrame(tsv_read)

df = pd.DataFrame(df.vector.str.split(" ", 1).tolist(), columns=['label', 'vector'])

print(df)


y = pd.DataFrame([df.label]).astype(int).to_numpy().reshape(-1, 1).ravel()
print(y.shape)

X = pd.DataFrame([dict(y.split(':') for y in x.split()) for x in df['vector']])
print(X.astype(float).to_numpy())



start = time()

clf = svm.SVC(kernel='rbf',
              C=51,
              gamma=1,
              probability=True
              )
clf.fit(X, y)  

# save
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)

# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

clf2.predict(X[0:1])