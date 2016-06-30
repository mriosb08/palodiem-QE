import GPy
import sys
import numpy as np
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
import scipy as sp
import re
from GPy import kern
from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch


def main(args):
    (training_file, label_file, test_file, test_label, unlabel, n_dim, output) = args
    X = load_feat(training_file)
    y = load_label(label_file)
    
    X = np.asarray(X)
    y = np.asarray(y)

    U = load_feat(unlabel)
    U = np.asarray(U[:10000])

    new_dim = int(n_dim)
    k = kern.RBF(new_dim, ARD=True) 
    #m = GPy.models.GPRegression(X, y)
    print 'reduction model'
    #m_red = BayesianGPLVMMiniBatch(X, new_dim, init="random", num_inducing=50,
    #                               kernel=k, missing_data=True)
    m_U =  GPy.models.SparseGPLVM(U, new_dim, kernel=k, num_inducing=50)
    #m_red.Ytrue = U
    print 'reduction optimize'
    m_U.optimize(optimizer='bfgs', max_iters=50)
    
    u_params = m_U.parameters

    m_O = GPy.models.SparseGPLVM(X, new_dim, kernel=k, num_inducing=50)
    m_O.parameters = u_params
    m_O.update_model()
    X = m_O.X.values

    #print dir(m_O.X)
    #print m_O.X.values
    print 'gp model'
    m = GPy.models.SparseGPRegression(X, y, num_inducing=50)
    print 'gp optimize'
    m.optimize(optimizer='bfgs', max_iters=50)
    
    test_X = load_feat(test_file)
    test_X = np.asarray(test_X)
    test_y = load_label(test_label)
    test_y = np.asarray(test_y)
    
    #test_latent = m_red.predict(test_X)[0]
    #print test_latent.shape
    #sys.exit(1)
    
    #m_redTest = BayesianGPLVMMiniBatch(test_X, new_dim, init="random", num_inducing=50,
    #                                    kernel=k, missing_data=True)
    
    m_test = GPy.models.SparseGPLVM(test_X, new_dim, num_inducing=50)
    m_test.parameters = u_params 
    m_test.update_model()
    test_latent = m_test.X.values
    #print test_latent.shape
    #sys.exit(1)
    pred = m.predict(test_latent)[0]
   
    #TODO test_X to latent space!!!

    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    print 'MAE: ', mae
    print 'RMSE: ', sqrt(mse)
    print 'pearson:', sp.stats.pearsonr(test_y, pred)[0]
    print 'true: ', mquantiles(test_y, prob=[0.1,0.9])
    print 'pred: ', mquantiles(pred, prob=[0.1,0.9])
    print 'resid: ', np.mean(test_y - pred)
    print 'r-sqr: ', sp.stats.linregress(test_y[:,0], pred[:,0])[2] ** 2 
     
    #with open(output, 'w') as output:
    #    for p in pred:
    #        print >>output, p[0]
    
    return

def load_label(label_file):
    y = []
    with open(label_file) as lf:
        for line in lf:
            line = line.strip()
            y.append([float(line)])
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
    if len(sys.argv) != 8:
        print 'usage:python GPy.3.py <training-features> <training-label> <test-features> <test-label> <u-file> <hid-dim> <output>'
        sys.exit(1)
    else:
        main(sys.argv[1:])

