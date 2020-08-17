import joblib
import os
from argparse import ArgumentParser
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from feature_extractor import load_all_features


def main(args):
    # Load data
    dataset_info = datasets.load_files(args.train_dir, None, None, False, False)
    raw_features = load_all_features(file_path=args.feat_dir, combined=True)

    if args.validate:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            raw_features, dataset_info.target, test_size=1 - args.train_ratio, random_state=args.random_state)
    else:
        X_train_raw, y_train = shuffle(raw_features, dataset_info.target, random_state=args.random_state)

    del raw_features

    # LDA dimension reduction
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=18)
    X_train = lda.fit_transform(X_train_raw, y_train)
    del X_train_raw

    if args.validate:
        X_test = lda.transform(X_test_raw)
        del X_test_raw

    # Normalize features
    X_train_n = normalize(X_train)
    if args.validate:
        X_test_n = normalize(X_test)

    # Train classifier
    classifier = OneVsRestClassifier(SVC())
    classifier.fit(X_train_n, y_train)
    print('Train set score:', classifier.score(X_train_n, y_train))
    if args.validate:
        print('Test set score:', classifier.score(X_test_n, y_test))

    # Save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    joblib.dump(lda, os.path.join(args.save_dir, 'lda.model'))
    joblib.dump(classifier, os.path.join(args.save_dir, 'svm.model'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    main(args)
