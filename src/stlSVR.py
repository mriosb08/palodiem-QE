import sys
import argparse
from selfTaughtLearning import SparseAutoencoder, feedForwardAutoencoder
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
from scipy import stats
import scipy as sp
import re
import numpy as np
from sklearn.svm import SVR

rho = 0.5
lamda = 0.0005
beta = 100
#hidden_size = 20
max_iterations = 30
u_size = 10000

def main(args):
    (training_file, label_file, u_file, test_file, output_file, e, c, hidden_size) = args
    X = load_feat(training_file)
    y = load_label(label_file)
    X = np.asarray(X)
    #X = np.delete(X, 4, 1) 
    y = np.asarray(y)
    visible_size = X.shape[1]
    test_X = load_feat(test_file)
    test_X = np.asarray(test_X)
    #test_X = np.delete(test_X, 4, 1)

    U = load_feat(u_file)
    U = np.asarray(U[:u_size])
    #U = np.delete(U, 4, 1)
    print 'autoencoder'
    u_encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)
    opt_solution  = sp.optimize.minimize(u_encoder.sparseAutoencoderCost, u_encoder.theta,
            args = (U.T,), method = 'L-BFGS-B',
            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    #opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)


    train_features = feedForwardAutoencoder(opt_theta, hidden_size, visible_size, X.T)
    test_features  = feedForwardAutoencoder(opt_theta, hidden_size, visible_size, test_X.T)
    print train_features.T.shape
    print test_features.T.shape
    print 'model build'
    svr = SVR(C=c, epsilon=e, kernel='rbf') 
    svr.fit(train_features.T, y)


    print 'test'
    #TODO delete nan features for future
    test_features[np.isnan(test_features)] = 0
    pred = svr.predict(test_features.T)     
    
    with open(output_file, 'w') as output:
        for p in pred:
            print >>output, p
    #
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
            #label = int(cols[0])
            features = [float(i) for i in cols]
            X.append(features)
            #y.append(label)
    return X

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='python stlSVR.py')
    parser.add_argument('--training-examples', dest='training_examples',
                            help='Training examples', required=True)
    parser.add_argument('--training-labels', dest='training_labels',
                            help='Training labels', required=True)
    parser.add_argument('--unlabelled-examples', dest='un_examples',
                            help='Unlabelled examples', required=True)
    parser.add_argument('--test', dest='test',
                            help='Test segments', required=True)
    parser.add_argument('--output', dest='output_file',
                            help='Prediction output', required=True)
    parser.add_argument('--epsilon', dest='epsilon',
            help='epsilon parameter for SVR default:0.232 ', type=float, default=0.232)
    parser.add_argument('--c', dest='c',
            help='C parameter for SVR default:41.06 ', type=float, default=41.06)
    parser.add_argument('--hidden-layer', dest='hidden',
            help='size hidden layer default:10 ', type=int, default=10)
    results = parser.parse_args()

    main([results.training_examples, results.training_labels, 
            results.un_examples, results.test, results.output_file, 
            results.epsilon, results.c, results.hidden])

