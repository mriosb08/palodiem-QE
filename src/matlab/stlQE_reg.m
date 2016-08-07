clearvars
addpath common/
addpath common/fminlbfgs



inputSize  = 217; 
numLabels  = 3;
hiddenSize = 100;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 0.003;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data 
uFile = '/data/mrios/workspace/data/binQE/wmt12.training.en-es.combo';
trainingFile = '/data/mrios/workspace/data/autodesk/sent.training.autodesk.en-es.pe.combo';
trainingLabel = '/data/mrios/workspace/data/autodesk/autodesk.training.en-es.caped.hter';
testFile = '/data/mrios/workspace/data/binQE/wmt12.test.en-es.combo';
testLabel = '/data/mrios/workspace/data/binQE/wmt12.test.en-es.hter';
%thetaFile = '/data/mrios/workspace/qeexp/en-es-pt/training/zoo.en-es-domain.theta.mat';

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

fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in training set: %d\n', size(trainData, 2));
fprintf('# examples in test set: %d\n', size(testData, 2));

%unlabeledData = unlabeledData(:, 1:10000);
%unlabeledData = abs(unlabeledData);
%simple scaling
%unlabeledData = (unlabeledData - min(min(unlabeledData)))./(max(max(unlabeledData))-min(min(unlabeledData)))
%trainData = (trainData - min(min(trainData)))./(max(max(trainData))-min(min(trainData)))
%testData = (testData - min(min(testData)))./(max(max(testData))-min(min(testData)))
unlabeledData = zscore(unlabeledData); %scaling with z-score
trainData = zscore(trainData);
testData = zscore(testData);

theta = initializeParameters(hiddenSize, inputSize);


%nitializeParameters(hiddenSize, inputSize);% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

%opttheta = theta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';


%fminlbfgs
[opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                  inputSize, hiddenSize, ...
                                  lambda, sparsityParam, ...
                                  beta, unlabeledData), ...
                             theta, options);

%% -----------------------------------------------------
                          
% Visualize weights
%W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
%display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);



testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);



%net = fitnet(100);
%net.trainFcn = 'trainbr';
%net.trainFcn = 'trainscg';
%[net,tr] = train(net,trainFeatures, trainLabels);

%pred = net(testFeatures);

mdlStd = LinearModel.fit(trainFeatures.',trainLabels);
pred = predict(mdlStd,testFeatures.');

errperf(testLabels,pred.','mae')
errperf(testLabels,pred.','rmse')
corrcoef(testLabels,pred.')
