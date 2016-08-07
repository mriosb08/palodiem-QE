addpath /data/mrios/workspace/matlab/UFLDL-tutorial/common/
addpath /data/mrios/workspace/matlab/UFLDL-tutorial/common/fminlbfgs
clearvars

uFile = '/data/mrios/workspace/data/autodesk/sent.training.autodesk.en-pt.vec';
trainingFile = '/data/mrios/workspace/data/autodesk/sent.training.autodesk.en-es.vec';
trainingLabel = '/data/mrios/workspace/data/autodesk/autodesk.training.en-es.hter';
testFile = '/data/mrios/workspace/data/autodesk/sent.test.autodesk.en-pt.vec';
testLabel = '/data/mrios/workspace/data/autodesk/autodesk.test.en-pt.hter';
output = '/data/mrios/workspace/data/autodesk/sent.autodesk.en-pt.vec';


unlabeledData = load(uFile);
%unlabeledData = unlabeledData.'; %transpose the instances are columns!!!
trainData = load(trainingFile);
%trainData = trainData.';
testData = load(testFile);
%testData = testData.';

%labels
trainLabels = load(trainingLabel);
%trainLabels = trainLabels.';
testLabels = load(testLabel);
%testLabels = testLabels.';

sizeTrain = size(trainData, 1);
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 1));
fprintf('# examples in training set: %d\n', size(trainData, 1));
fprintf('# examples in test set: %d\n', size(testData, 1));

%opTrain = unlabeledData2(:, 1:614);
%opLabel = ones(1, 614) * 1;
%trainData = horzcat(trainData, opTrain);
%trainLabels = horzcat(trainLabels, opLabel);

unlabeledData = unlabeledData(1:sizeTrain, :);


%
unlabeledData = zscore(unlabeledData); %scaling with z-score
trainData = zscore(trainData);
testData = zscore(testData);


[A, B] = canoncorr(trainData, unlabeledData);

size(A)
size(B)
size(testData)
size(trainData)
testFeatures = testData * B;
trainFeatures = trainData * B;

size(testFeatures)
size(trainFeatures)
size(A)
size(B)
hiddenSize = size(B, 2)

dlmwrite(strcat(output, '.training.cca'), trainFeatures, 'delimiter', '\t');
dlmwrite(strcat(output, '.test.cca'), testFeatures, 'delimiter', '\t');

