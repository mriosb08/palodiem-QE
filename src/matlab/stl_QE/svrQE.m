addpath /data/mrios/workspace/sw/libsvm-3.21/matlab/

clearvars;


%% ======================================================================
%  STEP 1: Load data 
trainingFile = '/data/mrios/workspace/qeexp/en-es-ro/training/task1-1_en-es_training.features.smote';
trainingLabel = '/data/mrios/workspace/qeexp/en-es-ro/training/en-es_score.train.smote';
testFile = '/data/mrios/workspace/qeexp/en-es-ro/test/zoo.en-es-ro.1_as.en.tok_to_zoo.en-es-ro.1_as.ro.tok.out';
testLabel = '/data/mrios/workspace/qeexp/en-es-ro/test/zoo.en-es-ro.1_as.label';
thetaFile = '/data/mrios/workspace/qeexp/en-es-ro/training/zoo.en-es-domain.theta.mat';


trainData = load(trainingFile);
testData = load(testFile);

%labels
trainLabels = load(trainingLabel);
testLabels = load(testLabel);

fprintf('# examples in training set: %d\n', size(trainData));
fprintf('# examples in test set: %d\n', size(testData));

trainData = zscore(trainData);
testData = zscore(testData);

%TODO [trn_data, tst_data, jn2] = scaleSVM(trn_data, tst_data, trn_data, 0, 1);

optparam = '-s  3  -t 2 -c  20  -g 64 -p 1';
%trainLabels = str2num(trainLabels)
%testLabels = str2num(testLabels)

model = svmtrain(trainLabels, trainData, optparam);
[pred, Acc] = svmpredict(testLabels, testData, model);


