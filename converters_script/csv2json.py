import pandas as pd
csv_data = pd.read_csv("input_data/vaw_dataset_coherence_cos.csv", sep = ";")
print(csv_data)

csv_data.to_json("vaw_dataset_coherence _SVM_.json", orient = "records")