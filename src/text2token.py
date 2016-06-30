# -*- coding: utf-8 -*-
import sys
words = dict()

for line in open(sys.argv[1]):
        line = line.strip()
        for w in line.split(' '):
            words[w] = 1

for w in sorted(words, key=lambda key: (-words[key], key)):
    print w
