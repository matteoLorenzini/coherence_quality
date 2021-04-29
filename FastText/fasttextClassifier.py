import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import numpy as np
import codecs
from time import time
from nltk.corpus import stopwords
from nltk import download
import csv
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import re
import requests
import sys
import argparse
import gzip
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create FastText embeddings for descriptions')
    parser.add_argument('base', help='base path for the train / test files')

    args = parser.parse_args()

    descriptions = args.base
    print("Processing file {} containing descriptions.".format(descriptions))

    start = time()

    print("Processing data")

    final_pred = list()
    final_test = list()

    for i in range(10):
        i_str = '{:02d}'.format(i)

        train_file = descriptions + ".train." + i_str
        test_file = descriptions + ".test." + i_str

        print("Cross-validating on iteration {}".format(i_str))

        start = time()

    #Default embeddings

        model = fasttext.train_supervised(input=train_file,
                                   lr=1.0,
                                   epoch=100,
                                   wordNgrams=2,
                                   bucket=200000,
                                   dim=50,
                                   loss='hs')

'''      
    #Pre-trained embeddings wikipedia uncomment to use

        model = fasttext.train_supervised(
                                input=train_file,
                                lr=1.0, epoch=100,
                                wordNgrams=2, bucket=200000, dim=300, loss='hs',
                                pretrainedVectors='cc_300.vec')

'''    
print("Training time: {}".format(time()-start))
start = time()

with open(test_file, 'r') as f:
            test_desc = f.readlines()

listPred = []
listLabel = []
for line in test_desc:
    if line.startswith("__label__Religione_e_Magia "):
        desc = line[len("__label__Religione_e_Magia "):]
        label = 1
    elif line.startswith("__label__Natura "):
        desc = line[len("__label__Natura "):]
        label = 2
    elif line.startswith("__label__Essere_umano_uomo_in_generale "):
        desc = line[len("__label__Essere_umano_uomo_in_generale"):]
        label = 3
    elif line.startswith("__label__Societa_civilizzazione_cultura "):
        desc = line[len("__label__Societa_civilizzazione_cultura"):]
        label = 4
    elif line.startswith("__label__Idee_e_concetti_astratti"):
        desc = line[len("__label__Idee_e_concetti_astratti"):]
        label = 5
    elif line.startswith("__label__Storia"):
        desc = line[len("__label__Storia"):]
        label = 6
    elif line.startswith("__label__Bibbia_storie_dal_Vecchio_e_dal_Nuovo_Testamento"):
        desc = line[len("__label__Bibbia_storie_dal_Vecchio_e_dal_Nuovo_Testamento"):]
        label = 7
    elif line.startswith("__label__Letteratura"):
        desc = line[len("__label__Letteratura"):]
        label = 8
    elif line.startswith("__label__Mitologia_classica_e_storia_antica"):
        desc = line[len("__label__Mitologia_classica_e_storia_antica"):]
        label = 9
    elif line.strip():
        print("<EMPTY?")
        print(line)
        print(">")
    else:
        print("<ERROR reading test")
        print(line)
        print(">")

    predLabel = model.predict(desc.rstrip("\n\r"))[0][0];
    if predLabel == "__label__Religione_e_Magia":
        pred = 1
    elif predLabel == "__label__Natura":
        pred = 2
    elif predLabel == "__label__Essere_umano_uomo_in_generale":
        pred = 3
    elif predLabel == "__label__Societa_civilizzazione_cultura":
        pred = 4
    elif predLabel == "__label__Idee_e_concetti_astratti":
        pred = 5
    elif predLabel == "__label__Storia":
        pred = 6
    elif predLabel == "__label__Bibbia_storie_dal_Vecchio_e_dal_Nuovo_Testamento":
        pred = 7
    elif predLabel == "__label__Letteratura":
        pred = 8
    elif predLabel == "__label__Mitologia_classica_e_storia_antica":
        pred = 9
    else:
        print("ERROR in prediction")

    listPred.append(pred)
    listLabel.append(label)

print("Testing time: {}".format(time() - start))

final_pred.extend(listPred)
final_test.extend(listLabel)

print(final_test)
print(final_pred)

output = descriptions + ".eval.gz"
with gzip.open(output, "wb") as f:
    np.savetxt(f, (final_test, final_pred), fmt='%i')