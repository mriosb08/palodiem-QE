addpath /data/mrios/workspace/matlab/UFLDL-tutorial/common/
addpath /data/mrios/workspace/matlab/UFLDL-tutorial/common/fminlbfgs
clearvars

inputSize  = 217; 
numLabels  = 3;

uFile = '/data/mrios/workspace/qeexp/bicvm/sent.training.opus.es-ro.vec.combo';
trainingFile = '/data/mrios/workspace/qeexp/bicvm/sent.training.wmt.en-es.vec.combo.smote';
trainingLabel = '/data/mrios/workspace/qeexp/bicvm/wmt.training.en-es.label.smote';
testFile = '/data/mrios/workspace/qeexp/bicvm/sent.test.zoo.es-ro.vec.combo';
testLabel = '/data/mrios/workspace/qeexp/bicvm/zoo.test.es-ro.label';

unlabeledData = load(uFile);
unlabeledData = unlabeledData.'; %transpose the instances are columns!!!
trainData = load(trainingFile);
trainData = trainData.';
testData = load(testFile);
testData = testData.';

%labels
trainLabels = load(trainingLabel);
trainLabels = trainLabels.';
testLabels = load(testLabel);
testLabels = testLabels.';

sizeTrain = size(trainData, 2);
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in training set: %d\n', size(trainData, 2));
fprintf('# examples in test set: %d\n', size(testData, 2));

%opTrain = unlabeledData2(:, 1:614);
%opLabel = ones(1, 614) * 1;
%trainData = horzcat(trainData, opTrain);
%trainLabels = horzcat(trainLabels, opLabel);

unlabeledData = unlabeledData(:, 1:sizeTrain);


%
unlabeledData = normr(unlabeledData); %scaling with z-score
trainData = normr(trainData);
testData = normr(testData);


[A, B] = canoncorr(trainData.', unlabeledData.');

size(A)
size(B)
testFeatures = testData.' * A;
trainFeatures = trainData.' * A;

size(trainFeatures)
testFeatures = testFeatures.';
trainFeatures = trainFeatures.';
size(testFeatures)
size(trainFeatures)
size(A)
size(B)
hiddenSize = size(A, 2)

dlmwrite(strcat(trainingFile, '.cca'), trainFeatures.', 'delimiter', '\t');
dlmwrite(strcat(testFile, '.cca'), testFeatures.', 'delimiter', '\t');

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

lambda = 1e-10;
numClasses = numel(unique(trainLabels));
%trainLabels = trainLabels.';
%testLabels = testLabels.';

softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);


%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);



%% -----------------------------------------------------

% Classification Score
pred

fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

R = corrcoef(pred, testLabels)
RHO = corr(pred.', testLabels.')

dlmwrite(strcat(testFile, '.pred.cca'), pred.', 'delimiter', '\t');
