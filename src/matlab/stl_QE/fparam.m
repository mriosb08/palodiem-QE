function [pred, hiddenSize, lambda, sparsityParam,beta, theta] = fparam(inputSize, unlabeledData, theta, trainData, trainLabels, testData, testLabels)
hiddenSize = randi([1 inputSize],1,1);
lambda = 0+1e-5*rand(1,1);%1e-5
sparsityParam = 0+0.1*rand(1,1);
beta =  randi([0 60],1,1);
theta = initializeParameters(hiddenSize, inputSize);
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.MaxIter = 20;	  % Maximum number of iterations of L-BFGS to run 
options.Display = 'iter';
options.GradObj = 'on';
%fminlbfgs
[opttheta, cost] = fminlbfgs( @(p) sparseAutoencoderCost(p, ...
                                  inputSize, hiddenSize, ...
                                  lambda, sparsityParam, ...
                                  beta, unlabeledData), ...
                             theta, options);
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);
testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);
softmaxModel = struct;  
lambda = 1e-10;
numClasses = numel(unique(trainLabels));
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);
[pred] = softmaxPredict(softmaxModel, testFeatures);
pred = (100*mean(pred(:) == testLabels(:)));
end

