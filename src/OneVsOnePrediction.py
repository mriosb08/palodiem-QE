import sys
from collections import Counter
from collections import defaultdict

def main(args):
    (names, predf1, predf2, predf3) = args
    (n1, n2, n3) = names.split(',')
    pred1 = [float(line.strip()) for line in open(predf1)]
    pred2 = [float(line.strip()) for line in open(predf2)]
    pred3 = [float(line.strip()) for line in open(predf3)]
    
    for i1, i2, i3 in zip(pred1, pred2, pred3):
        i1 = decision(n1, i1)
        i2 = decision(n2, i2)
        i3 = decision(n3, i3)
        most_common,num_most_common = Counter([i1, i2, i3]).most_common(1)[0]
        print most_common
        

    return

def decision(name, pred):
    (a, b) = name.split('-')
    if pred > 0.0:
        return int(a)
    else:
        return int(b)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage:python OneVsOnePrediction.py <system-names> <prediction-file1> <prediction-file2> <prediction-file3>'
        sys.exit(1)
    else:
        main(sys.argv[1:])


