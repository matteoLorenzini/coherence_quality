# Subject prediction in cultural heritage visual arts works

  

## Resources 

* [FastText](https://fasttext.cc/)
* [ScikitLearn](https://scikit-learn.org/stable/index.html)
* [Wikipedia Embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html)
* [Annotated dataset](https://figshare.com/articles/dataset/Annotated_dataset_to_assess_the_accuracy_of_the_textual_description_of_cultural_heritage_records/13359104)

## Dependencies 

* Python3
* Pandas
* Pickle

## Pipeline FastText

* Input data
  * [Iconclass](https://github.com/matteoLorenzini/coherence/blob/master/input_data/icon.txt): pre labelled Iconclass dataset. The resources are already labelled according to FastText syntax using the prefix __label__ e.g. __label__Religione_e_Magia
  * [TestSet](https://github.com/matteoLorenzini/coherence/blob/master/input_data/vaw_test_dataset_coherence_baseline.csv): 500 high-quality descriptions used as test dataset for subject prediction. Descriptions are from the [Annotated dataset](https://figshare.com/articles/dataset/Annotated_dataset_to_assess_the_accuracy_of_the_textual_description_of_cultural_heritage_records/13359104) 

* Train the model
  * Use the [train_model.py](https://github.com/matteoLorenzini/coherence/blob/master/FastText/train_model.py) script to create the prediction model
* Run the prediction model
  * Run the [predict_fasttext.py](https://github.com/matteoLorenzini/coherence/blob/master/FastText/predict_fasttext.py) script to obtain the top 3 predicted subjects

## Pipeline SVM

* Input data
  * [Iconclass](https://github.com/matteoLorenzini/coherence/blob/master/input_data/icon.txt): pre labelled Iconclass dataset.
  * [TestSet](https://github.com/matteoLorenzini/coherence/blob/master/input_data/vaw_test_dataset_coherence_baseline.csv): 500 high-quality descriptions used as test dataset for subject prediction. Descriptions are from the [Annotated dataset](https://figshare.com/articles/dataset/Annotated_dataset_to_assess_the_accuracy_of_the_textual_description_of_cultural_heritage_records/13359104)

* Create the word embeddings
  * Run the script [csv2json.py](https://github.com/matteoLorenzini/coherence/blob/master/convert/csv2json.py) to convert the test dataset and the iconclass dataset in .json
  * Run the script [convert.py](https://github.com/matteoLorenzini/coherence/blob/master/convert/converter.py) to convert the .json(s) file of the test and iconclass datasets in .tsv word embeddings
* Train the model
  * Run the script [train_svm.py](https://github.com/matteoLorenzini/coherence/blob/master/SVM/train_svm.py) to train the prediction model and obtain the .pkl file
* Run the prediction model
  * Run the [predict_svm.py]((https://github.com/matteoLorenzini/coherence/blob/master/SVM/predict_svm.py)) to obtain the top 3 predicted subjects

[![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/) [![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)