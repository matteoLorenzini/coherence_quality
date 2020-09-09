import fasttext
from nltk import ngrams
import numpy as np
import pandas as pd


ftModelFile = "cc.it.300.bin"

print("Loading FastText\n")
model_ft = fasttext.load_model(ftModelFile)

print("Loading stopwords\n")
stopwords = []
with open("stopwords.txt") as file:
    for line in file:
        line = line.strip().lower()
        if len(line) == 0:
            continue
        stopwords.append(line)


def CleanStopWords(sentence):
    sentenceSplitted = sentence.split(" http://dh-server.fbk.eu:19003/simp-engines/tae/simpform")  # METTERE UN VERO TOKENIZZATORE
    sentence = [w for w in sentenceSplitted if w not in stopwords]
    return (sentence)

df_complete = pd.DataFrame()

with open("inputFile/vaw_labelled.tsv", "r") as f:
    for line in f:
        parts = line.split("\t")

        # DESCRIPTION
        mydesc = parts[1]

        sentencearray = CleanStopWords(mydesc.lower())
        sentence = ' '.join(sentencearray)

        embeddings_descr = model_ft.get_sentence_vector(sentence)

        print(sentence, embeddings_descr[:3])
        print()

        bigrams = ngrams(sentence.split(), 2)
        trigrams = ngrams(sentence.split(), 3)

        # SUBJECT

        mysubj = parts[2]

        subjarray = CleanStopWords(mysubj.lower())
        subject = ' '.join(subjarray)

        embeddings_subj = model_ft.get_sentence_vector(subject)

        print(subject, embeddings_subj[:3])
        print()

        def cos_sim(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot_product / (norm_a * norm_b)

        a = embeddings_descr
        b = embeddings_subj

        cosine = [(cos_sim(a, b))]

        print(cosine)

        max_cosine_bigr = None
        min_cosine_bigr = None

        for b in bigrams:
            emb = model_ft.get_sentence_vector("".join(b))
            print(b, emb[:3])

            a = emb
            b = embeddings_subj

            cosine_bigr = [(cos_sim(a, b))]
            print("Cosine bigrams:", cosine_bigr)

            if max_cosine_bigr is None or max_cosine_bigr < cosine_bigr:
                max_cosine_bigr = cosine_bigr

            if min_cosine_bigr is None or min_cosine_bigr > cosine_bigr:
                min_cosine_bigr = cosine_bigr

        print(f"Min cosine bigrams: {min_cosine_bigr}")
        print(f"Max cosine bigrams: {max_cosine_bigr}")

        max_cosine_trig = None
        min_cosine_trig = None

        for t in trigrams:
            emb = model_ft.get_sentence_vector("".join(t))
            print(t, emb[:3])

            a = emb
            b = embeddings_subj

            cosine_trig = [(cos_sim(a, b))]
            print("Cosine trigrams:", cosine_trig)

            if max_cosine_trig is None or max_cosine_trig < cosine_trig:
                max_cosine_trig = cosine_trig


            if min_cosine_trig is None or min_cosine_trig > cosine_trig:
                min_cosine_trig = cosine_trig



        print(f"Min cosine trigrams: {min_cosine_trig}")
        print(f"Max cosine trigrams: {max_cosine_trig}")

        df_cosine = pd.DataFrame(cosine, columns=['Cosine'])

        df_min_bigr = pd.DataFrame(min_cosine_bigr, columns=['Cos_min_bigr'])

        df_max_bigr = pd.DataFrame(max_cosine_bigr, columns=['Cos_max_bigr'])

        df_min_trig = pd.DataFrame(min_cosine_trig,columns = ['Cos_min_trig'])

        df_max_trig = pd.DataFrame(max_cosine_trig,columns = ['Cos_max_trig'])


        data_set = pd.concat([ df_cosine['Cosine'],df_min_bigr['Cos_min_bigr'], df_max_bigr['Cos_max_bigr'],df_min_trig['Cos_min_trig'], df_max_trig['Cos_max_trig']], axis=1, keys=['Cosine','Cos_min_bigr','Cos_max_bigr','Cos_min_trig', 'Cos_max_trig'])

        df_complete = df_complete.append(data_set)

    df_complete.to_csv('vaw_labelled.csv')






