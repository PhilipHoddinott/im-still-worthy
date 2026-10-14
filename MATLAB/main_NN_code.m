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
numberOfFitRows = size(XFit,1);

% make a history struct
history.epoch = (1:user_options.numEpochs)';
history.fitLoss = zeros(user_options.numEpochs);
history.fitAcc = zeros(user_options.numEpochs);
history.validAcc = zeros(user_options.numEpochs);
history.elapsedSec = zeros(user_options.numEpochs);

trainTimer = tic;

for i_epoch = 1:user_options.numEpochs

    % shuffle some rows
    shuffledRows = randperm(user_options.numEpochs);

    for iBatch = 1:user_options.batchSize:numberOfFitRows
        batchEnd = min(iBatch + user_options.batchSize -1, numberOfFitRows);

        batchRows = shuffledRows(iBatch:batchEnd);

        XBatch = Xfit(batchRows,:);
        yBatch = yFit(batchRows);

        b_cache = NN_forward(model,XBatch);

        gradiants = NN_backward(model, b_cache, yBatch);

        model = update_model(model, gradiants, user_options);
    end

