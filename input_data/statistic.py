import pandas as pd
import matplotlib.pyplot as plt

#dataset
set = pd.read_csv ('../SVM/vaw_svm_prediction.csv',sep=',',names=['manual_label','prediction','index','label_1','label_2','label_3','baseline_1','baseline_2','baseline_3'
])

df = pd.DataFrame(set)

print(df)

prediction = df.groupby(['prediction']).size().reset_index(name='counts').sort_values("counts", ascending=False)
label_1 = df.groupby(['label_1']).size().reset_index(name='counts').sort_values("counts", ascending=False)
label_2 = df.groupby(['label_2']).size().reset_index(name='counts').sort_values("counts", ascending=False)
label_3 = df.groupby(['label_3']).size().reset_index(name='counts').sort_values("counts", ascending=False)

#order = stat.sort_values("counts", ascending=False)

print(prediction)
print(label_1)
print(label_2)
print(label_3)

fig, ((ax1, ax2), (ax3, ax4)) =  plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1 = prediction.plot.bar(x='prediction', y='counts', rot=0)
ax2 = label_1.plot.bar(x='label_1', y='counts', rot=0)
ax3 = label_2.plot.bar(x='label_2', y='counts', rot=0)
ax4 = label_3.plot.bar(x='label_3', y='counts', rot=0)

plt.show()
'''
ax = stat.plot.bar(x='label_3', y='counts', rot=0)

plt.xticks(rotation=90,fontsize=8)
plt.title("Dataset Structure")
plt.xlabel("Categories")
plt.ylabel("Number of subject per category")
plt.show()
'''
