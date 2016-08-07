addpath ../common/
addpath ../common/fminlbfgs
clearvars

inputSize  = 17; 
numLabels  = 3;
hiddenSize = 5;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-5;       % weight decay parameter       
beta = 30;            % weight of sparsity penalty term   
maxIter = 5;

%% ======================================================================
%  STEP 1: Load data 
%uFile = '/data/mrios/workspace/qeexp/en-ru-uk/training/zoo.training.es-ro.es.tok_to_zoo.training.es-ro.ro.tok.out';
trainingFile = '/data/mrios/workspace/qeexp/en-ru-uk/training/ru-mteval.qe.en-ru.full.en.tok_to_ru-mteval.qe.en-ru.full.ru.tok.out';
trainingLabel = '/data/mrios/workspace/qeexp/en-ru-uk/training/ru-mteval.qe.en-ru.label';
%testFile = '/data/mrios/workspace/qeexp/en-ru-uk/test/zoo.test.es-ro.es.tok_to_zoo.test.es-ro.ro.tok.out';
%testLabel = '/data/mrios/workspace/qeexp/en-ru-uk/test/zoo.es-ro.ed_as.label';
%thetaFile = '/data/mrios/workspace/qeexp/en-ru-uk/training/zoo.es-pt.theta.mat';

%unlabeledData = load(uFile);
%unlabeledData = unlabeledData.'; %transpose the instances are columns!!!
trainData = load(trainingFile);
trainData = trainData.';
%testData = load(testFile);
%testData = testData.';

%labels
trainLabels = load(trainingLabel);
trainLabels = trainLabels.';
%testLabels = load(testLabel);
%testLabels = testLabels.';

%fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in training set: %d\n', size(trainData, 2));
%fprintf('# examples in test set: %d\n', size(testData, 2));

%unlabeledData = unlabeledData(:, 1:10000);
%unlabeledData = abs(unlabeledData);
%simple scaling
%unlabeledData = (unlabeledData - min(min(unlabeledData)))./(max(max(unlabeledData))-min(min(unlabeledData)))
%trainData = (trainData - min(min(trainData)))./(max(max(trainData))-min(min(trainData)))
%testData = (testData - min(min(testData)))./(max(max(testData))-min(min(testData)))
%unlabeledData = zscore(unlabeledData); %scaling with z-score
trainData = zscore(trainData);
%testData = zscore(testData);


[trainInd,valInd,testInd] = dividerand(size(trainData, 2),0.9,0.1,0.0);
%[trainData,devData,testT] = divideind(trainData,trainInd,valInd,testInd);
devData = trainData(:, valInd.');
devLabels = trainLabels(:, valInd.');
trainData = trainData(:, trainInd.');
trainLabels = trainLabels(:, trainInd.');
fprintf('# examples in training set: %d\n', size(trainData, 2));
fprintf('# examples in dev set: %d\n', size(devData, 2));

lambda = 1e-10;
numClasses = numel(unique(trainLabels));

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 20;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';



softmaxModel = struct;

softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                                 trainData, trainLabels, options);

[pred] = softmaxPredict(softmaxModel, devData);
pred
fprintf('Test Accuracy split 90-10: %f%%\n', 100*mean(pred(:) == devLabels(:)));



%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  instances. 

%  Randomly initialize the parameters
%theta = initializeParameters(hiddenSize, inputSize);

%learn hyperparems from validation
%for i = 1:maxIter
    %fprintf('value of a: %d\n', a);
%    [score, h, l, sp, b, t] = fparam(inputSize, ...
%                                    unlabeledData, theta, ... 
%                                    trainData, trainLabels, ... 
%                                    devData, devLabels);
    %fprintf('value  iter %d: %f\n', i, pred);
%    vecParam(i).pred = score;
%    vecParam(i).hiddenSize = h;
%    vecParam(i).lambda = l;
%    vecParam(i).sparsityParam = sp;
%    vecParam(i).beta = b;
%    vecParam(i).theta = t;
%end

%[max_score, index_score] = max([vecParam.pred]);
%fprintf('max i:%d %f\n', index_score,max_score);
%vecParam(index_score);
%theta =  vecParam(index_score).theta;
%hiddenSize = vecParam(index_score).hiddenSize;
%lambda = vecParam(index_score).lambda;
%sparsityParam = vecParam(index_score).sparsityParam;
%beta = vecParam(index_score).beta;


%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

%opttheta = theta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 20;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

%fminlbfgs
%[opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
%                                  inputSize, hiddenSize, ...
%                                  lambda, sparsityParam, ...
%                                  beta, unlabeledData), ...
%                             theta, options);

%% -----------------------------------------------------
                          
% Visualize weights
%W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
%display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  

%trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
%                                       trainData);



%testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
%                                       testData);

cv = cvpartition(trainLabels, 'kfold',10); 
score = 0.0;
for i = 1:10
    tvec = test(cv,i);
    trvec = training(cv,i);
    trainL = trainLabels(:, trvec.');
    trainF = trainData(:, trvec.');
    testL = trainLabels(:, tvec.');
    testF = trainData(:, tvec.');
    softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                                trainF, trainL, options);
    [predc] = softmaxPredict(softmaxModel, testF);
    %fprintf('Test Accuracy %d: %f%%\n', i, 100*mean(predc(:) == testL(:)));
    score = score + (100*mean(predc(:) == testL(:)));
end

score = score / 10.0
%fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

