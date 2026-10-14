function [xFit, yFit, xValidation, yValidation, xTest, yTest] = prepData_for_NN(dataFolder, seed)  

if nargin < 1
    dataFolder = 'Data';
end
trainFile = [dataFolder, '/mnist_training.mat'];
testFile = [dataFolder, '/mnist_test.mat'];


load(trainFile, 'labels','pixel_values');
validFraction = 10/60; % for mNIST 

[fitIdx, validIdx] = make_strat_split(labels, validFraction, seed);

xFit = pixel_values(fitIdx,:); 
yFit = labels(fitIdx);

xValidation = pixel_values(validIdx,:); 
yValidation = labels(validIdx);


%% get traning data 
load(testFile, 'labels','pixel_values');

xTest = pixel_values; 
yTest = labels;
end