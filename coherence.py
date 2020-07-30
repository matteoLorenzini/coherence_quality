import fasttext
from nltk import ngrams




ftModelFile = "cc.it.300.bin"

print("Loading FastText\n")
model_ft  = fasttext.load_model(ftModelFile)

print("Loading stopwords\n")
stopwords = []
with open("stopwords.txt") as file:
    for line in file: 
        line = line.strip().lower()
        if len(line) == 0:
            continue
        stopwords.append(line)

def CleanStopWords (sentence):
    sentenceSplitted = sentence.split(" ")  #METTERE UN VERO TOKENIZZATORE 
    sentence = [w for w in sentenceSplitted if w not in stopwords]
    return(sentence)

# figura maschile schematizzata
#mytext = "Bronzetto rappresentante una figura schematica di uomo seduto a gambe appena divaricate con i gomiti puntati sulle ginocchia e le mani davanti alla testa"
# mytext = "Bronzetto rappresentante una figura schematica di uomo seduto a gambe appena divaricate con i gomiti puntati sulle ginocchia e le mani davanti alla testa, le estremità delle braccia si uniscono in maniera indistinta, sicchè non si può precisare se le meni reggono qualcosa davanti alla bocca o se sono congiunte sotto il mento. Nella testa cilindrica e coronata da una calotta piatta ad indicare la capigliatura, non ci sono indicazioni di occhi, naso e nuca né delle dita nelle mani e nei piedi Il busto è esile un po' appiattito le membra si piegano descrivendo un arco acuto senza però dar risalto alle giunture Appartiene ad un tipo diffuso in Grecia nei Balcani nell Italia centrale in tutta l età del ferro"
with open ("excel/dataset.csv.tsv","r") as f:
    for line in f:
        parts = line.split("\t")
        mytext = parts[2]
        
        mysubj = parts[3]

        #AGGIUNGERE RIMOZIONE PUNTEGGIATURA
        sentencearray = CleanStopWords(mytext.lower())
        sentence = ' '.join(sentencearray)

        embeddings = model_ft.get_sentence_vector(sentence)

        print(sentence,embeddings[:3])
        print()

        bigrams  = ngrams(sentence.split(), 2)
        trigrams = ngrams(sentence.split(), 3)

        for t in trigrams:
        	emb = model_ft.get_sentence_vector(''.join(t))
        	print (t,emb[:3])

        for b in bigrams: 
        	emb = model_ft.get_sentence_vector(''.join(b))
        	print (b,emb[:3])

        def cos_similarity (mytext, mysubj):

            fasttext_model = fasttext.load_model(ftModelFile)

            descr_emb = np.mean([fasttext_model[x] for word in mytext for x in word.split()if x not in stopwords], axis=0)

            subj_emb = np.mean([fasttext_model[x] for word in mysubj for x in word.split()if x not in stopwords], axis=0)

            return "Similarity:", 1- scipy.spatial.distance.cosine(descr_emb, subj_emb)

        print(cos_similarity)