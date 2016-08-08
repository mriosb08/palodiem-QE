import sys
import argparse 
from collections import defaultdict
import itertools
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
from scipy import stats
import scipy as sp
import re
import numpy as np
from sklearn.svm import SVR



u_size = 10000

def main():
    parser = argparse.ArgumentParser(description='python easy_adapt.py')
    parser.add_argument('-t', action="store", dest='training_data',
            help='Training dataset', required=True)
    parser.add_argument('-l', action="store", dest='training_label',
            help='Training labels', required=True)
    parser.add_argument('-u', action="store", dest='u_data',
            help='Training un', required=True)
    parser.add_argument('-w', action="store", dest='u_label',
            help='Training un labelled', required=True)
    parser.add_argument('-p', action="store", dest='test_data',
            help='Test dataset', required=True)
    parser.add_argument('-q', action="store", dest='test_label',
            help='Test label', required=True)
    parser.add_argument('-c', action="store", dest='c',
            help='C value', type=float, required=True)
    parser.add_argument('-e', action="store", dest='e',
            help='E value', type=float, required=True)
    
    results = parser.parse_args()
    
    X = load_feat(results.training_data)
    y = load_label(results.training_label)
    X = np.asarray(X)
    y = np.asarray(y)
    visible_size = X.shape[1]
    test_X = load_feat(results.test_data)
    test_X = np.asarray(test_X)

    U = load_feat(results.u_data)
    U = np.asarray(U[:u_size])
    y_u = load_label(results.u_label)
    y_u = np.asarray(y_u[:u_size])
    (train_features, y) = feature_augmentation([X, U], [y, y_u], -1, 2)
    
    c = results.c
    e = results.e
    svr = SVR(C=c, epsilon=e, kernel='rbf') 
    svr.fit(train_features, y)


    print 'test'
    #pred = svr.predict(test_features)     
    
    test_y = load_label(results.test_label)
    test_y = np.asarray(test_y)
    (test_features, test_y) = feature_augmentation([test_X], [[], test_y], 1, 2)
    pred = svr.predict(test_features) 
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    scores = stats.linregress(test_y, pred)
    r_value = scores[2]
    print 'MAE: ', mae
    print 'RMSE: ', sqrt(mse)
    print 'pearson:', sp.stats.pearsonr(test_y, pred)[0]
    print 'true: ', mquantiles(test_y, prob=[0.1,0.9])
    print 'pred: ', mquantiles(pred, prob=[0.1,0.9])
    print 'resid: ', np.mean(test_y - pred)
    print 'r-squared: ', r_value**2  
    

    with open(test_file + '.easysvr.pred', 'w') as output:
        for p in pred:
            print >>output, p
    return

def load_label(label_file):
    y = []
    with open(label_file) as lf:
        for line in lf:
            line = line.strip()
            y.append(float(line))
    return y

def load_feat(feat_file):
    X = []
    with open(feat_file) as feat:
        for line in feat:
            line = line.strip()
            cols = re.split('\s+', line)
            features = [float(i) for i in cols]
            X.append(features)
    return X




def feature_augmentation(X, y, test_id, num_tasks):
    X_aug = []
    y_aug = []
    if num_tasks == -1:
        num_tasks = len(X)
    flag_task = 0
    for id_task, task_line in enumerate(X):
        if test_id != -1:
            id_task = test_id
        for features, score in zip(task_line, y[id_task]):
            aug_features = []
            if test_id == None or test_id == -1:
                aug_features.append(features) #general!!!
            else:
                aug_features.append([0.0] * len(features))
            for i in range(num_tasks):
                #different to current task
                if id_task != i: 
                    copy = [0.0] * len(features)
                    aug_features.append(copy)
                #current task
                else:
                    aug_features.append(features)

            merged = list(itertools.chain.from_iterable(aug_features))
            X_aug.append(merged)
            y_aug.append(score)
    return (X_aug, y_aug)




if __name__ == '__main__':
    main()
