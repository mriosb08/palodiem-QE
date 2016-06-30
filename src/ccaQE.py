import sys
import argparse
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
import re
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
from scipy import stats
import numpy as np
import scipy as sp


def main(args):
    (training_file, label_file, test_file, u_file, e, c, output_file, components) = args
    X_training = load_feat(training_file)
    n = len(X_training)
    U = load_feat(u_file)
    y_training = [float(line.strip()) for line in open(label_file)]
   
    U = np.asarray(U)
    X_training = np.asarray(X_training)
    #X = preprocessing.normalize(X, norm='l2')
    y_training = np.asarray(y_training)
    
    X_test = load_feat(test_file)
    y_test = [float(line.strip()) for line in open(test_label)]
    X_test = np.asarray(X_test)
    X_test[np.isnan(X_test)] = 0.0
    #test_X = preprocessing.normalize(test_X, norm='l2')
    y_test = np.asarray(y_test)

    
    cca = CCA(n_components=components, max_iter=50)
    (X_cca, U_cca) = cca.fit_transform(X_training[:1000], U[:1000])
    X_test_cca = cca.transform(X_test)
    
    svr = SVR(C=c, epsilon=e, kernel='rbf')
    svr.fit(X_cca, y_training[:1000])    
    pred = svr.predict(X_test_cca)
    
 
    with open(output_file, 'w') as output:
        for p in pred:
            print >>output, p
    return


def load_feat(feat_file):
    X = []
    with open(feat_file) as feat:
        for line in feat:
            cols = re.split('\s+', line.strip())
            features = [float(i) for i in cols]
            X.append(features)
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='python ccaQE.py')
    parser.add_argument('--training-examples', action="append", dest='training_examples',
                            help='Training examples', required=True)
    parser.add_argument('--training-labels', action="append", dest='training_labels',
                            help='Training labels', required=True)
    parser.add_argument('--unlabelled-examples', action="append", dest='un_examples',
                            help='Unlabelled examples', required=True)
    parser.add_argument('--test', action="append", dest='test',
                            help='Test segments', required=True)
    parser.add_argument('--output', action="append", dest='output_file',
                            help='Prediction output', required=True)
    parser.add_argument('--epsilon', action="append", dest='epsilon',
            help='epsilon parameter for SVR default:0.232 ', default=0.232)
    parser.add_argument('--c', action="append", dest='c',
            help='C parameter for SVR default:41.06 ', default=41.06)
    parser.add_argument('--hidden-layer', action="append", dest='hidden',
            help='size hidden layer default:10 ', default=10)
    results = parser.parse_args()

    main(results.training_examples, results.training_labels, 
            results.un_examples, results.test, results.output_file, 
            results.epsilon, results.c, results.hidden)


