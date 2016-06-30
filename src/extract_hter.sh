INPUT="autodesk.test.en-cz"
TRG="cz"
PE="pe"

/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < $INPUT.$PE | ./text2ter.pl > $INPUT.$PE.tok
/data/mrios/workspace/sw/cdec-2014-10-12/corpus/tokenize-anything.sh < $INPUT.$TRG | ./text2ter.pl > $INPUT.$TRG.tok
java -jar /data/mrios/workspace/sw/tercom-0.7.25/tercom.7.25.jar -N -o ter -r $INPUT.$PE.tok -h $INPUT.$TRG.tok -n $INPUT.tercom
cut -f 4 -d ' ' $INPUT.tercom.ter | sed '/^$/d' > $INPUT.hter
