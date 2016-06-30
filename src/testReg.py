import sys
from math import sqrt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats.mstats import mquantiles
import scipy as sp
import numpy as np
def main(args):
    (ftrue, fpred) = args
    y_true = [float(line.strip()) for line in open(ftrue)]
    y_pred = [float(line.strip()) for line in open(fpred)]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print 'MAE: ', mae
    print 'RMSE: ', sqrt(mse)
    print 'pearson:', sp.stats.pearsonr(y_true, y_pred)[0]
    print 'spearman', sp.stats.spearmanr(y_true, y_pred)[0]
    print 'r-squared', sp.stats.linregress(y_true, y_pred)[0]
    print 'true: ', mquantiles(y_true, prob=[0.1,0.9])
    print 'pred: ', mquantiles(y_pred, prob=[0.1,0.9])
    print 'resid_mean: ', np.mean(y_true - y_pred)

    return

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage:python testReg.py <y-true> <y-predicted>'
        sys.exit(1)
    else:
        main(sys.argv[1:])

