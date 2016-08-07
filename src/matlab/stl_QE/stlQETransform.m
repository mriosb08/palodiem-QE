addpath ../common/
addpath ../common/fminlbfgs
clearvars

inputSize  = 200; 
numLabels  = 3;
hiddenSize = 20;
sparsityParam = 0.5; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 0.0005;       % weight decay parameter       
beta = 100;            % weight of sparsity penalty term   
maxIter = 30;

%% ======================================================================
%  STEP 1: Load data 
uFile = '/data/mrios/workspace/data/autodesk/sent.training.autodesk.en-pt.vec';
trainingFile = '/data/mrios/workspace/data/autodesk/sent.training.autodesk.en-es.vec';
trainingLabel = '/data/mrios/workspace/data/autodesk/autodesk.training.en-es.hter';
testFile = '/data/mrios/workspace/data/autodesk/sent.test.autodesk.en-pt.vec';
testLabel = '/data/mrios/workspace/data/autodesk/autodesk.test.en-pt.hter';
output = '/data/mrios/workspace/data/autodesk/sent.autodesk.en-pt.vec';


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

unlabeledData = unlabeledData(:, 1:10000);
unlabeledData = zscore(unlabeledData); %scaling with z-score
trainData = zscore(trainData);
testData = zscore(testData);


%[trainInd,valInd,testInd] = dividerand(size(trainData, 2),0.9,0.1,0.0);
%devData = trainData(:, valInd.');
%devLabels = trainLabels(:, valInd.');
%trainData = trainData(:, trainInd.');
%trainLabels = trainLabels(:, trainInd.');
%fprintf('# examples in training set: %d\n', size(trainData, 2));
%fprintf('# examples in dev set: %d\n', size(devData, 2));



%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  instances. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%learn hyperparems from validation
%for i = 1:maxIter
    %fprintf('value of a: %d\n', a);
%    [score, h, l, sp, b, t] = fparam(inputSize, ...
%                                    unlabeledData, theta, ... 
%                                    trainData, trainLabels, ... 
%                                    devData, devLabels);
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
                          
%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);



testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);



trainFeatures = trainFeatures.';
testFeatures = testFeatures.';
trainLabels = trainLabels.';
testLabels = testLabels.';
dlmwrite(strcat(output, '.training.1.stl'), trainFeatures, 'delimiter', '\t');
dlmwrite(strcat(output, '.training.hter.1.stl'), trainLabels, 'delimiter', '\t');
dlmwrite(strcat(output, '.test.1.stl'), testFeatures, 'delimiter', '\t');
dlmwrite(strcat(output, '.test.hter.1.stl'), testLabels, 'delimiter', '\t');

