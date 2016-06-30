import sys
import re
def main(args):
    if len(args) != 2:
        with open(args[0]) as ff:
            for line in ff:
                line = line.strip()
                features = ['%s:%s'%(i+1,f) for i,f in enumerate(re.split("\s+", line))]
                if features:
                    print '0 %s'%(' '.join(features))
    elif len(args) == 2:
        ff = [line.strip() for line in open(args[0])]
        gf = [line.strip() for line in open(args[1])]
        for gold, features in zip(gf, ff):
            features = ['%s:%s'%(i+1,f) for i,f in enumerate(re.split("\s+", features))]
            if features:
                print '%s %s'%(gold, ' '.join(features))

    return

if __name__ == '__main__':
    if len(sys.argv) > 3:
        print 'usage:python feat2svm.py <feature-file> [gold-file]'
        sys.exit(1)
    else:
        main(sys.argv[1:])
