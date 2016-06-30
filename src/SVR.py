import sys
from sklearn.svm import SVR
from sklearn import grid_search
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
from sklearn import preprocessing
import scipy as sp
import pickle
import numpy as np
import re

def main(args):
    (training_file, label_file, test_file, test_label, c, e) = args
    #svm_model = SVC()
    svr = SVR(C=float(c), epsilon=float(e), kernel='rbf')
    #svr = SVR()
    #svr = LogisticRegression(C=1e15)
    X = load_feat(training_file)
    y = [float(line.strip()) for line in open(label_file)]
    
    X = np.asarray(X)
    #X = np.delete(X, 4, 1)
    X = preprocessing.normalize(X, norm='l2')
     
    y = np.asarray(y)
    
    test_X = load_feat(test_file)
    test_X = np.asarray(test_X)
    #test_X = np.delete(test_X, 4, 1)
    test_X[np.isnan(test_X)] = 0
    test_X = preprocessing.normalize(test_X, norm='l2')

    svr.fit(X, y)
    
    pred = svr.predict(test_X)
    #print pred
    #print test_y
    if test_label != 'none':
        test_y = [float(line.strip()) for line in open(test_label)]
        test_y = np.asarray(test_y)
        print 'MAE: ', mean_absolute_error(test_y, pred)
        print 'RMSE: ', sqrt(mean_squared_error(test_y, pred))
        print 'corrpearson: ', sp.stats.pearsonr(test_y, pred)
        print 'r-sqr: ', sp.stats.linregress(test_y, pred)[2] ** 2
        print mquantiles(test_y, prob=[0.10, 0.90])
        print mquantiles(pred, prob=[0.10, 0.90])
    with open(test_file + '.svr.pred', 'w') as output:
        for p in pred:
            print >>output, p
    #svm_model.fit(X, y)
    #pickle.dump(lr, open(model_file, "wb"))
    return

def load_feat(feat_file):
    X = []
    with open(feat_file) as feat:
        for line in feat:
            cols = re.split('\s+', line.strip())
            #label = int(cols[0])
            features = [float(i) for i in cols]
            X.append(features)
            #y.append(label)
    return X

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print 'usage:python SVR.py <training-features> <training-label> <test-features> <test-labels> <C> <epsilon>'
        sys.exit(1)
    else:
        main(sys.argv[1:])

