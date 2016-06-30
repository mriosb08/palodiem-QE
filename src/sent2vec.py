import sys
import numpy as np

def main(args):
    (vec_file, sent_file) = args
    (vec, size) = load_vec(vec_file)
    with open(sent_file) as sf:
        for line in sf:
            line = line.strip()
            cols = line.split(' ')
            sum = np.zeros(size)
            #print cols
                    
            for col in cols:
                if col in vec:
                    sum += vec[col]
                    #print vec[col]
            out = [str(i) for i in sum.tolist()]
            print ' '.join(out)
    return

def load_vec(vec_file):
    vec = dict()
    size = 0
    with open(vec_file) as vf:
        for line in vf:
            line = line.strip()
            cols = line.split(' ')
            #print cols
            vec[cols[0]] = np.asarray(cols[1:], dtype=np.float32)
            #print cols[0]
            #print np.asarray(cols[1:], dtype=np.float32)
            size = len(cols[1:])

    return (vec, size)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage:python sent2vec.py <vec_file> <sent_file>'
        sys.exit(1)
    else:
        main(sys.argv[1:])
