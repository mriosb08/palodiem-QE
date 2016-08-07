addpath ../common/
addpath ../common/fminlbfgs
clearvars;

inputSize  = 17; 
numLabels  = 3;
hiddenSize = 9;
sparsityParam = 1.0653e-05; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 1.0000e-10;       % weight decay parameter       
beta = 15;            % weight of sparsity penalty term   
maxIter = 20;

%% ======================================================================
%  STEP 1: Load data 
uFile = '/data/mrios/workspace/qeexp/en-es-pt/training/zoo.training.10k.2015.en-pt.en.tok_to_zoo.training.10k.2015.en-pt.en.trans.tok.out';
trainingFile = '/data/mrios/workspace/qeexp/en-es-pt/training/task1-1_en-es_training.features';
trainingLabel = '/data/mrios/workspace/qeexp/en-es-pt/training/en-es_score.train';
testFile = '/data/mrios/workspace/qeexp/en-es-pt/test/en-pt.en.tsv.tok_to_en-pt.pt.tsv.tok.out';
testLabel = '/data/mrios/workspace/qeexp/en-es-pt/test/en-pt.score.tsv';
thetaFile = '/data/mrios/workspace/qeexp/en-es-pt/training/zoo.en-es-domain.theta.mat';

%unlabeledData = load(uFile);
%unlabeledData = unlabeledData.'; %transpose the instances are columns!!!
trainData = load(trainingFile);
trainData = trainData.';
testData = load(testFile);
testData = testData.';

%labels
trainLabels = load(trainingLabel);
trainLabels = trainLabels.';
testLabels = load(testLabel);
testLabels = testLabels.';

%fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in training set: %d\n', size(trainData, 2));
fprintf('# examples in test set: %d\n', size(testData, 2));

%unlabeledData = abs(unlabeledData);
%simple scaling
%unlabeledData = (unlabeledData - min(min(unlabeledData)))./(max(max(unlabeledData))-min(min(unlabeledData)))
%trainData = (trainData - min(min(trainData)))./(max(max(trainData))-min(min(trainData)))
%testData = (testData - min(min(testData)))./(max(max(testData))-min(min(testData)))
%unlabeledData = zscore(unlabeledData); %scaling with z-score
trainData = zscore(trainData);
testData = zscore(testData);

theta = initializeParameters(hiddenSize, inputSize);


%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

opttheta = theta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 20;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';



softmaxModel = struct;  

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

lambda = 1e-10;
numClasses = numel(unique(trainLabels));
%trainLabels = trainLabels.';
%testLabels = testLabels.';

softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            trainData, trainLabels, options);


%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testData);


%% -----------------------------------------------------

% Classification Score
pred

fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
dlmwrite(strcat(testFile, '.pred.base'), pred.', 'delimiter', '\t');
