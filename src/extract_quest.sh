INPUT="autodesk"
PAIR="en-cz"
SRC="en"
TRG="cz"
PE="pe"

/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < autodesk.$PAIR.clean.$SRC > training_quest/autodesk.$PAIR.$SRC
/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < autodesk.$PAIR.clean.$TRG > training_quest/autodesk.$PAIR.$TRG
cp /data/mrios/workspace/mosesexp/autodesk/$PAIR/model/lex.1.e2f training_quest/autodesk.$PAIR.e2f

/data/mrios/workspace/sw/mymoses/bin/lmplz -o 3 < training_quest/autodesk.$PAIR.$SRC > training_quest/autodesk.$PAIR.$SRC.lm
/data/mrios/workspace/sw/mymoses/bin/lmplz -o 3 < training_quest/autodesk.$PAIR.$TRG > training_quest/autodesk.$PAIR.$TRG.lm

/data/mrios/workspace/sw/srilm/bin/i686-m64/ngram-count -text training_quest/autodesk.$PAIR.$SRC -write training_quest/autodesk.$PAIR.$SRC.ngram
/data/mrios/workspace/sw/srilm/bin/i686-m64/ngram-count -text training_quest/autodesk.$PAIR.$TRG -write training_quest/autodesk.$PAIR.$TRG.ngram

/data/mrios/workspace/sw/mymoses/scripts/recaser/train-truecaser.perl --model training_quest/autodesk.$PAIR.true.$SRC --corpus training_quest/autodesk.$PAIR.$SRC
/data/mrios/workspace/sw/mymoses/scripts/recaser/train-truecaser.perl --model training_quest/autodesk.$PAIR.true.$TRG --corpus training_quest/autodesk.$PAIR.$TRG

java -classpath /data/mrios/workspace/quest/quest/build/classes:/data/mrios/workspace/quest/quest/lib/ shef.mt.util.NGramSorter training_quest/autodesk.$PAIR.$SRC.ngram training_quest/autodesk.$PAIR.$SRC.ngram.sort
java -classpath /data/mrios/workspace/quest/quest/build/classes:/data/mrios/workspace/quest/quest/lib/ shef.mt.util.NGramSorter training_quest/autodesk.$PAIR.$TRG.ngram training_quest/autodesk.$PAIR.$TRG.ngram.sort
