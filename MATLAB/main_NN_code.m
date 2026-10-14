close all; clear all;

%% setup
addpath('Util/');
dataFolder = 'Data/';
user_options.inputSize = 784;
user_options.hiddenSize = 256;
user_options.outputSize = 10;
user_options.learnRate = [0.05, 0.02, 0.01]; % 0.05;
user_options.batchSize = 128; % 
user_options.numEpochs = 60;
user_options.seed = 69; % 
user_options.momentum = .9;

%% load in data

[xFit, yFit, xValidation, yValidation, xTest, yTest] = prepData_for_NN(dataFolder, user_options.seed);


%% train the NN
% get the model
[model, velocity] = init_NN(user_options);
% misc vars
numberOfFitRows = size(xFit,1);

% make a history struct
history.epoch = (1:user_options.numEpochs)';
history.fitLoss = zeros(user_options.numEpochs,1);
history.validLoss = zeros(user_options.numEpochs,1);
history.fitAcc = zeros(user_options.numEpochs,1);
history.validAcc = zeros(user_options.numEpochs,1);
history.elapsedSec = zeros(user_options.numEpochs,1);

trainTimer = tic;
i_lr = 1;

lr = user_options.learnRate;
lrr = lr;
sz_lr = size(lr,2); 


for i_epoch = 1:user_options.numEpochs

    % shuffle some rows
    shuffledRows = randperm(numberOfFitRows);
    
    % handle changing lr 
    lr = user_options.learnRate;
    if sz_lr >1 
        ii = i_epoch ./ user_options.numEpochs;
        if ii > (i_lr / sz_lr)
            i_lr = i_lr + 1;
        end
        lr = lrr(i_lr);
    else
        lr = lrr;
    end

    i_batch_out_count = 0;
    for iBatch = 1:user_options.batchSize:numberOfFitRows
        batchEnd = min(iBatch + user_options.batchSize -1, numberOfFitRows);

        batchRows = shuffledRows(iBatch:batchEnd);

        XBatch = xFit(batchRows,:);
        yBatch = yFit(batchRows);

        b_cache = NN_forward(model,XBatch);

        gradiants = NN_backward(model, b_cache, yBatch);
        
        % model = update_model(model, gradiants, lr);
        [model, velocity] = update_model(model, gradiants, lr, user_options.momentum, velocity);

        if i_epoch == 1
            if mod(i_batch_out_count, 10) == 0
                [~, fitAcc] = getFit(model,xFit,yFit);
                fprintf('Epoch %d / %d | batch: %d / %d | in batch Fit %.4f \n ',i_epoch, user_options.numEpochs, iBatch, numberOfFitRows,  fitAcc);
            end
            i_batch_out_count = i_batch_out_count+ 1;
        end
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

    fprintf('Epoch %d / %d | lr: %.2f | ',i_epoch, user_options.numEpochs, lr);
    fprintf('Fit loss %.4f | validation loss %.4f | ',history.fitLoss(i_epoch), history.validLoss(i_epoch));
    fprintf('Fit %.4f | validation %.4f\n',history.fitAcc(i_epoch), history.validAcc(i_epoch));
end
    
    

  
