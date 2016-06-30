TRAIN=""
TRAIN_LABEL=""
ULABEL=""
TEST=""
TEST_LABEL=""
python /data/mrios/workspace/palodiem-qe/feat2svm.py training/$TRAIN training/$TRAIN_LABEL > training/$TRAIN.svm
python /data/mrios/workspace/palodiem-qe/feat2svm.py training/$ULABEL > training/$ULABEL.svm
python /data/mrios/workspace/palodiem-qe/feat2svm.py test/$TEST > test/$TEST.1.2.svm
python /data/mrios/workspace/palodiem-qe/feat2svm.py test/$TEST > test/$TEST.1.3.svm
python /data/mrios/workspace/palodiem-qe/feat2svm.py test/$TEST > test/$TEST.2.3.svm
python /data/mrios/workspace/palodiem-qe/OneVsOneFeatures.py training/$TRAIN.svm training/$ULABEL.svm all training/$ULABEL.features.svm.part
/data/mrios/workspace/sw/svmlin-v1.0/svmlin -A 2 training/$ULABEL.features.svm.part.feat.1.2 training/$ULABEL.features.svm.part.label.1.2
/data/mrios/workspace/sw/svmlin-v1.0/svmlin -A 2 training/$ULABEL.features.svm.part.feat.1.3 training/$ULABEL.features.svm.part.label.1.3
/data/mrios/workspace/sw/svmlin-v1.0/svmlin -A 2 training/$ULABEL.features.svm.part.feat.2.3 training/$ULABEL.features.svm.part.label.2.3

/data/mrios/workspace/sw/svmlin-v1.0/svmlin -f $ULABEL.features.svm.part.feat.1.2.weights test/$TEST.1.2.svm
/data/mrios/workspace/sw/svmlin-v1.0/svmlin -f $ULABEL.features.svm.part.feat.1.3.weights test/$TEST.1.3.svm
/data/mrios/workspace/sw/svmlin-v1.0/svmlin -f $ULABEL.features.svm.part.feat.2.3.weights test/$TEST.2.3.svm

python /data/mrios/workspace/palodiem-qe/OneVsOnePrediction.py 1-2,1-3,2-3 $TEST.1.2.svm.outputs $TEST.1.3.svm.outputs $TEST.2.3.svm.outputs > test.svm.pred
python /data/mrios/workspace/palodiem-qe/accu.py test/$TEST_LABEL test.svm.pred
