import sys
import itertools
import numpy as np
import re
from collections import defaultdict
def main(args):
    (l_file, u_file, u_type, output_file) = args
    class_dict = defaultdict(list)
    with open(l_file) as lf:
        for line in lf:
            line = line.strip()
            cols = line.split(' ')
            #print cols
            class_dict[int(cols[0])].append(cols[1:])
    num_class = len(class_dict.keys())
    print sorted(class_dict.keys())
    print num_class
    comb = list(itertools.combinations(sorted(class_dict.keys()), 2))
    print comb
    u = [re.split('\s+', line.strip()) for line in open(u_file)]
    print u_type
    if u_type == 'all':
        for a,b in comb:
            final = add_class(class_dict[a], '+1') + add_class(class_dict[b], '-1') + add_class(u, '')
            print len(final)
            prefix = '%s.%s'%(a,b)
            label_f = open(output_file + '.label.%s'%prefix, 'w')
            feat_f = open(output_file + '.feat.%s'%prefix, 'w')
            for f in final:
                cols = re.split('\s+', f)
                print >>label_f, cols[0]
                print >>feat_f, ' '.join(cols[1:])
                #print f
    elif u_type == 'cos':
        #for instance in u:
        #    features_u = extract_features(instance[1:])
        #    features_a = extract_features(class_dict[aclass_dict[a])
        #TODO add unk given instance to combination (1,2) based on cognate info
        #like 1,2 
        pass



    return

def extract_features(instance):
    return [re.sub(r'[0-9]+:', '', i) for i in instance]

def add_class(instances, c):
    final = []
    for instance in instances:
        if c:
            final.append('%s %s'%(c, ' '.join(instance)))
        else:
            final.append(' '.join(instance))
    return final

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'usage:python OneVsOneFeature.py <labeled-file> <unlabeled-file> <type-unlabel> <output-prefix>'
        sys.exit(1)
    else:
        main(sys.argv[1:])
