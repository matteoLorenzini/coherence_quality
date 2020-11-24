from sklearn.metrics import classification_report
import numpy as np
import os
import argparse


def readEvalData(evaldata):

    data=np.loadtxt(evaldata)
    return data[0], data[1]


def score(evaldata, output):

    test,pred=readEvalData(evaldata)
    print(test)
    print(pred)

    print(classification_report(test, pred, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('evaldata', help='numpy file with predicted and test classes')

    args = parser.parse_args()

    evaldata = args.evaldata
    print("Processing dataset {}".format(evaldata))
    assert os.path.exists(evaldata), "dataset not found"

    if evaldata.endswith('.gz'):
        evaldata_mod = evaldata[:-3]
    else:
        evaldata_mod = evaldata

    output = evaldata_mod + ".score"
    score(evaldata, output)
