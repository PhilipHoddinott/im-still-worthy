function [fitIdx, validIdx] = make_strat_split(labels, validFraction, seed)
% make the stupud split 
% no clue why i found this harder than back prop
labels = labels(:);
% set rng
rng(seed);

labelCounts = countMNISTlabels(labels);

nRows = numel(labels);
numberofValidRows = round(validFraction * nRows);


exactValidationCounts = validFraction*labelCounts;

validationCounts = floor(exactValidationCounts);

rowsLeft = numberofValidRows - sum(validationCounts);

[~, allocationorder] = sort(exactValidationCounts - validationCounts, 'descend');

classesToAdd = allocationorder(1:rowsLeft);

validationCounts(classesToAdd) = validationCounts(classesToAdd) + 1;

fitIndices = [];
validationIndices = [];
for digit = 0:9

    classRows = find(labels == digit);    
    classRows = classRows(randperm(numel(classRows)));    
    classValidationCount = validationCounts(digit + 1);
    
    validationIndices = [validationIndices; classRows(1:classValidationCount)];    
    fitIndices = [fitIndices; classRows(classValidationCount + 1:end)];    
end
fitIdx = fitIndices(randperm(numel(fitIndices)));
validIdx = validationIndices(randperm(numel(validationIndices)));
end