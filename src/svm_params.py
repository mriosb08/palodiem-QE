import sys
import numpy as np
import re

def main(args):
    training_file = args[0]
    df = load_file(training_file)
    mean = np.mean(df)
    stdev = np.std(df)
    n = df.size
    c = max(abs(mean + (3 * stdev)), abs(mean + (3 * stdev)))
    eps = (3 * stdev) * np.sqrt(np.log(n) / n)
    print 'C: ', c
    print 'eps: ', eps
    return

def load_file(training_file):
    df = []
    with open(training_file) as tf:
        for line in tf:
            line = line.strip()
            cols = re.split('\s+', line)
            df.append([float(col)  for col in cols])
    return np.asarray(df)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage: svm_params.py <training-set>'
        sys.exit(1)
    else:
        main(sys.argv[1:])
