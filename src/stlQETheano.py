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
import GPy
from theano.tensor.shared_randomstreams import RandomStreams

max_iterations = 100
u_size = 10000
batch_size = 1000

def main(args):
    (training_file, label_file, u_file, test_file, test_label, output, n, hid_size) = args
    X = load_feat(training_file)
    y = load_label(label_file)
    X = theano.shared(np.asarray(X, dtype=theano.config.floatX))
    y = np.asarray(y)
    visible_size = X.get_value().shape[1]

    test_X = load_feat(test_file)
    test_y = load_label(test_label)
    test_X = theano.shared(np.asarray(test_X, dtype=theano.config.floatX))
    test_y = np.asarray(test_y)

    U = load_feat(u_file)
    U = theano.shared(np.asarray(U[:u_size], dtype=theano.config.floatX))
    print 'autoencoder'
    ul = T.dmatrix('ul')
    index = T.lscalar()
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    n_train_batches = U.get_value().shape[0]
    print n_train_batches
    u_da = dA(numpy_rng=rng, theano_rng=theano_rng, input=ul, n_visible=visible_size, n_hidden=int(hid_size))
    #print u_da.n_visible
    #print u_da.n_hidden
    cost, updates = u_da.get_cost_updates(
        corruption_level=1.0,
        learning_rate=0.00001
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
            c_tmp = train_da(batch_index)
            c.append(c_tmp) 
        #print 'Training epoch %d, cost ' % epoch, np.mean(c)
    #end_time = timeit.default_timer()

    #training_time = (end_time - start_time)
    

    train_features = u_da.get_hidden_values(X)
    test_features = u_da.get_hidden_values(test_X)
    print train_features.eval().shape

    #print dir(train_features)
    #print type(train_features.eval())
    #print train_features.eval()
    #train_features = np.asarray(train_features.eval()) 
    #test_features = np.asarray(test_features.eval())
    
    #kernel = GPy.kern.RBF()
    #m = GPy.models.GPRegression(X, y)
    #n = '1000'
    print train_features.eval()
    print 'model build'
    kernel = GPy.kern.RBF(input_dim=int(hid_size), variance=1., lengthscale=1.)
    m = GPy.models.SparseGPRegression(train_features.eval(), y, kernel=kernel, num_inducing=int(n))
    print 'training'
    m.optimize(optimizer='bfgs', max_iters=50, messages=True)
    
    print 'test'
    pred = m.predict(test_features.eval())[0]
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    print 'MAE: ', mae
    print 'RMSE: ', sqrt(mse)
    print 'pearson:', sp.stats.pearsonr(test_y, pred)[0]
    print 'resid mean:', np.mean(test_y - pred)
    print 'true: ', mquantiles(test_y, prob=[0.1,0.9])
    print 'pred: ', mquantiles(pred, prob=[0.1,0.9])
    
    with open(output, 'w') as output:
        for p in pred:
            print >>output, p[0]
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
    if len(sys.argv) != 9:
        print 'usage:python stlQETheano.py <training-features> <training-label> <un-file>  <test-features> <test-label> <output> <num-inducing> <hid_size>'
        sys.exit(1)
    else:
        main(sys.argv[1:])

