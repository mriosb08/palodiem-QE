import sys
from dA import dA
import theano
import theano.tensor as T
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
import scipy as sp
import re
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.svm import SVR


rho = 0.5
lamda = 0.0005
beta = 100
max_iterations = 100
u_size = 10000
batch_size=20

def main(args):
    (training_file, label_file, u_file1, u_file2, test_file, test_label, output, c, e, hid_size1, hid_size2) = args
    X = load_feat(training_file)
    y = load_label(label_file)
    X = theano.shared(np.asarray(X, dtype=theano.config.floatX))
    y = np.asarray(y)
    visible_size = X.get_value().shape[1]

    test_X = load_feat(test_file)
    test_y = load_label(test_label)
    test_X = theano.shared(np.asarray(test_X, dtype=theano.config.floatX))
    test_y = np.asarray(test_y)

    U1 = load_feat(u_file1)
    U1 = theano.shared(np.asarray(U1[:u_size], dtype=theano.config.floatX))
    U2 = load_feat(u_file2)
    U2 = theano.shared(np.asarray(U2[:u_size], dtype=theano.config.floatX))
   
    (u_da1, train_da1) = sAE(U1, visible_size, int(hid_size1))
    
    rho = 0.05
    lamda = 0.00005
    beta = 300

    (u_da2, train_da2) = sAE(U2, visible_size, int(hid_size2))

    train_features = u_da1.reconstruct(X)
    train_features = u_da2.reconstruct(train_features)
    test_features = u_da2.reconstruct(test_X)
   

    #print dir(train_features)
    #print type(train_features.eval())
    #print train_features.eval()
    #train_features = np.asarray(train_features.eval()) 
    #test_features = np.asarray(test_features.eval())
    
    #kernel = GPy.kern.RBF()
    #m = GPy.models.GPRegression(X, y)
    #n = '1000'
    print 'model build'
    
    c = float(c)
    e = float(e)
    svr = SVR(C=float(c), epsilon=float(e), kernel='rbf')
    svr.fit(train_features.eval(), y)
    
    print 'test'
    #TODO delete nan features for future
    #test_features[np.isnan(test_features)] = 0
    pred = svr.predict(test_features.eval())
        
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    print 'MAE: ', mae
    print 'RMSE: ', sqrt(mse)
    print 'pearson:', sp.stats.pearsonr(test_y, pred)[0]
    print 'resid mean:', np.mean(test_y - pred)
    print 'true: ', mquantiles(test_y, prob=[0.1,0.9])
    print 'pred: ', mquantiles(pred, prob=[0.1,0.9])
    
    #with open(output, 'w') as output:
    #    for p in pred:
    #        print >>output, p[0]
    return

def sAE(U, visible_size, hid_size):
    print 'autoencoder'
    ul = T.dmatrix('ul')
    index = T.lscalar()
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    n_train_batches = U.get_value().shape[0]
    u_da = dA(numpy_rng=rng, theano_rng=theano_rng, input=ul, n_visible=visible_size, n_hidden=hid_size)
    #print u_da.n_visible
    #print u_da.n_hidden
    cost, updates = u_da.get_cost_updates(
        corruption_level=0.5,
        learning_rate=0.000001
    )
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            ul: U[index * batch_size: (index + 1) * batch_size]
        }
    )

    #start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(max_iterations):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
     
    return (u_da, train_da)


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
    if len(sys.argv) != 12:
        print 'usage:python stlQETheano.py <training-features> <training-label> <un-file1> <un-file2> <test-features> <test-label> <output> <c> <e> <hid_size1> <hid_size2>'
        sys.exit(1)
    else:
        main(sys.argv[1:])

