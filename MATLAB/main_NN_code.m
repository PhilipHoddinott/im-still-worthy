close all; clear all;

%% setup
addpath('Util/');
dataFolder = 'Data/';
user_options.inputSize = 784;
user_options.hiddenSize = 69;
user_options.outputSize = 10;
user_options.learnRate = 0.05;
user_options.batchSize = 128; % 
user_options.numEpochs = 40;
user_options.seed = 69; % 

%% load in data

[xFit, yFit, xValidation, yValidation, xTest, yTest] = prepData_for_NN(dataFolder, user_options.seed);


%% train the NN
% get the model
model = init_NN(user_options);
% misc vars
numberOfFitRows = size(xFit,1);

% make a history struct
history.epoch = (1:user_options.numEpochs)';
history.fitLoss = zeros(user_options.numEpochs);
history.validLoss = zeros(user_options.numEpochs);
history.fitAcc = zeros(user_options.numEpochs);
history.validAcc = zeros(user_options.numEpochs);
history.elapsedSec = zeros(user_options.numEpochs);

trainTimer = tic;

for i_epoch = 1:user_options.numEpochs

    % shuffle some rows
    shuffledRows = randperm(numberOfFitRows);

    for iBatch = 1:user_options.batchSize:numberOfFitRows
        batchEnd = min(iBatch + user_options.batchSize -1, numberOfFitRows);

        batchRows = shuffledRows(iBatch:batchEnd);

        XBatch = xFit(batchRows,:);
        yBatch = yFit(batchRows);

        b_cache = NN_forward(model,XBatch);

        gradiants = NN_backward(model, b_cache, yBatch);

        model = update_model(model, gradiants, user_options);
    end
    
    % eval the fits
    [fitLoss, fitAcc] = getFit(model,xFit,yFit);

    history.fitLoss(i_epoch) = fitLoss;
    history.fitAcc(i_epoch) = fitAcc;

    % eval the validations

    validProbs = NN_forward(model,xValidation);

    history.validLoss(i_epoch) = NN_cross_entropy(validProbs.prob, yValidation);
    [~, validClassIdx] = max(validProbs.prob, [] , 2);
    validPred = validClassIdx - 1;

    history.validAcc(i_epoch) = mean(validPred == yValidation);

    % print each loop output

    history.elapsedSec(i_epoch) = toc(trainTimer);

    fprintf('Epoch %d / %d | ',i_epoch, user_options.numEpochs);
    fprintf('Fit loss %.4f | validation loss %.4f | ',history.fitLoss(i_epoch), history.validLoss(i_epoch));
    fprintf('Fit %.2f | validation %.2f\n ',history.fitAcc(i_epoch), history.validAcc(i_epoch));
end
    
    

  
