import joblib
import json
import os
from argparse import ArgumentParser
from feature_extractor import load_all_features
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize


def main(args):
    labels = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
              'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']
    lda: LinearDiscriminantAnalysis = joblib.load(os.path.join(args.model_dir, 'lda.model'))
    classifier: OneVsRestClassifier = joblib.load(os.path.join(args.model_dir, 'svm.model'))
    X_test_raw = load_all_features(file_path=args.test_feat_dir, combined=True)
    X_test = lda.transform(X_test_raw)
    del X_test_raw
    X_test_n = normalize(X_test)

    y_predict = classifier.predict(X_test_n)
    filenames = os.listdir(args.test_dir)
    result = {}
    for n in range(len(y_predict)):
        result[filenames[n]] = labels[y_predict[n]]

    with open(os.path.join(args.output_dir, 'result.json'), 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--test_feat_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
