addpath ../common/
addpath ../common/fminlbfgs
clearvars;
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 217;
numClasses = 3;
hiddenSizeL1 = 196;    % Layer 1 Hidden Size
hiddenSizeL2 = 180;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 1e-10;         % weight decay parameter       
beta = 30;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data %

%trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
%trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

%trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
uFile1 = '/data/mrios/workspace/qeexp/bicvm/sent.training.opus.en-es.vec.combo';
uFile2 = '/data/mrios/workspace/qeexp/bicvm/sent.training.opus.es-ro.vec.combo';
trainingFile = '/data/mrios/workspace/qeexp/bicvm/sent.training.wmt.en-es.vec.combo.2.smote';
trainingLabel = '/data/mrios/workspace/qeexp/bicvm/wmt.training.en-es.label.2.smote';
testFile = '/data/mrios/workspace/qeexp/bicvm/sent.test.zoo.es-ro.vec.combo';
testLabel = '/data/mrios/workspace/qeexp/bicvm/zoo.test.es-ro.label';
uName='es-ro';



unlabeledData1 = load(uFile1);
unlabeledData1 = unlabeledData1.'; %transpose the instances are columns!!!
unlabeledData2 = load(uFile2);
unlabeledData2 = unlabeledData2.'; %transpose the instances are columns!!!
%unlabeledData3 = load(uFile3);
%unlabeledData3 = unlabeledData3.';


trainData = load(trainingFile);
size(trainData)
trainData = trainData.';
testData = load(testFile);
testData = testData.';

%labels
trainLabels = load(trainingLabel);
trainLabels = trainLabels.';
testLabels = load(testLabel);
testLabels = testLabels.';

fprintf('# examples in unlabeled set: %d\n', size(unlabeledData1, 2));
fprintf('# examples in training set: %d\n', size(trainData, 2));
fprintf('# examples in test set: %d\n', size(testData, 2));

%size(unlabeledData2)
fprintf('# examples in unlabeled set2: %d\n', size(unlabeledData2, 2));
%simple scaling
%unlabeledData = (unlabeledData - min(min(unlabeledData)))./(max(max(unlabeledData))-min(min(unlabeledData)))
%trainData = (trainData - min(min(trainData)))./(max(max(trainData))-min(min(trainData)))
%testData = (testData - min(min(testData)))./(max(max(testData))-min(min(testData)))
unlabeledData1 = zscore(unlabeledData1); %scaling with z-score 
% TODO use other normalization PCA whitening!!!
unlabeledData2 = zscore(unlabeledData2);
trainData = zscore(trainData);
testData = zscore(testData);


%% STEP 2: Train the first sparse autoencoder


%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';

 [sae1OptTheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData1), ...
                              sae1Theta, options);


 % save('layer1.mat',  'sae1OptTheta');

 % load('layer1.mat');

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

sae1Features;
size(sae1Features)
size(trainData)
%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, inputSize);

%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"

 [sae2OptTheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData2), ...
                              sae2Theta, options);

 % save('layer2.mat',  'sae2OptTheta');
 % load('layer2.mat')'

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
size(sae2OptTheta);
sae1Features;
size(sae2Theta);
size(sae1Theta);
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                       hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);



%sae2Features;
options.MaxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% save('layer3.mat', 'saeSoftmaxOptTheta');
% load('layer3.mat');

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model


% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

pred
acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
dlmwrite(strcat(testFile, '.pred.beforestacked.stl'), pred.', 'delimiter', '\t');
errperf(testLabels,pred,'mae')
errperf(testLabels,pred,'rmse')




% save('layer4.mat', 'stackedAEOptTheta');


% -------------------------------------------------------------------------


%%======================================================================
%% STEP 6: Test 
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
%testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
%testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

[stackedAEOptTheta, cost] = fminlbfgs( @(p) stackedAECost(p, ...
                                   inputSize, hiddenSizeL2, ...
                                   numClasses, netconfig, ...
                                   lambda, trainData, trainLabels), ...
                              stackedAETheta, options);




[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);



pred

dlmwrite(strcat(testFile, '.pred.stacked.stl'), pred.', 'delimiter', '\t');

[predNum, predLabel] = hist(pred, unique(pred))

[testNum, testLabel] = hist(testLabels, unique(testLabels))
errperf(testLabels,pred,'mae')
errperf(testLabels,pred,'rmse')

dlmwrite(strcat(trainingFile, '.', uName, '.2.stl'), trainData, 'delimiter', '\t');
dlmwrite(strcat(trainingLabel, '.', uName, '.2.stl'), trainLabels, 'delimiter', '\t');
%dlmwrite(strcat(testFile, '.pred.3.stl'), pred.', 'delimiter', '\t');
dlmwrite(strcat(testFile, '.2.stl'), testData, 'delimiter', '\t');

