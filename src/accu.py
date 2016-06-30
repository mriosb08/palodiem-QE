import sys
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import scipy

true = [int(line.strip()) for line in open(sys.argv[1])]
pred = [int(line.strip()) for line in open(sys.argv[2])]

print classification_report(true, pred)
print accuracy_score(true, pred)


