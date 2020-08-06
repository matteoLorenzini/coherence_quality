
import pandas as pd
import sys

for f in sys.argv[1:]:
	data = pd.read_csv(f, low_memory=False)
	data.to_csv(f+'.tsv', sep='\t')

