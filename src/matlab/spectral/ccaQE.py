import sys
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
import re
from sklearn.metrics import accuracy_score
import numpy as np
def main(args):
    (training_file, label_file, test_file, test_label, u_file) = args
    X_training = load_feat(training_file)
    n = len(X_training)
    U = load_feat(u_file)
    y_training = [int(line.strip()) for line in open(label_file)]
   
    U = np.asarray(U)
    X_training = np.asarray(X_training)
    #X = preprocessing.normalize(X, norm='l2')
    y_training = np.asarray(y_training)
    
    X_test = load_feat(test_file)
    y_test = [int(line.strip()) for line in open(test_label)]
    X_test = np.asarray(X_test)
    #test_X = preprocessing.normalize(test_X, norm='l2')
    y_test = np.asarray(y_test)

    
    cca = CCA(n_components=100)
    (X_cca, U_cca) = cca.fit_transform(X_training, U[:n])
    X_test_cca = cca.predict(X_test)
    
    svr = SVC()
    svr.fit(X_cca, y_training)    
    pred = svr.predict(X_test_cca)
    
    print pred
    print test_y
    print accuracy_score(y_test, pred)
    with open(test_file + '.cca.2.pred', 'w') as output:
        for p in pred:
            print >>output, p
    #svm_model.fit(X, y)
    #pickle.dump(lr, open(model_file, "wb"))
    return


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
    if len(sys.argv) != 6:
        print 'usage:python ccaQEpy <training-features> <training-label> <test-features> <test-labels> <U-file>'
        sys.exit(1)
    else:
        main(sys.argv[1:])
