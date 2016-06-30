SENT1='autodesk.test.en-cz.en'
MODEL1='modelA.en-cz.autodesk.100'

SENT2='autodesk.test.en-cz.cz'
MODEL2='modelB.en-cz.autodesk.100'

OUTPUT='sent.test.autodesk.en-cz.vec'

/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < $SENT1 | /data/mrios/workspace/sw/cdec-2014-10-12/corpus/lowercase.pl > $SENT1.lc
python text2token.py $SENT1.lc > $SENT1.vocab
/data/mrios/workspace/sw/bicvm/bin/extract-vectors --input $SENT1.vocab -m $MODEL1 >  $SENT1.out
python /data/mrios/workspace/palodiem-qe/sent2vec.py $SENT1.out $SENT1.lc > $SENT1.vec

/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < $SENT2 | /data/mrios/workspace/sw/cdec-2014-10-12/corpus/lowercase.pl > $SENT2.lc
python text2token.py $SENT2.lc > $SENT2.vocab
/data/mrios/workspace/sw/bicvm/bin/extract-vectors --input $SENT2.vocab -m $MODEL2 >  $SENT2.out
python /data/mrios/workspace/palodiem-qe/sent2vec.py $SENT2.out $SENT2.lc > $SENT2.vec
paste -d ' ' $SENT1.vec $SENT2.vec > $OUTPUT


