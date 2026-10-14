function loss = NN_cross_entropy(prob, labels)
    % is this loss?
    numberOfRows = numel(labels);
    sz_prob = size(prob);
    tempArr = (1:numberOfRows)';
    labelP1 = double(labels) + 1;
    
    correctIndex = sub2ind(sz_prob, tempArr, labelP1); % people wwill make fun of MATLAB for this line with the index startign at 1
    
    correcetProb = prob(correctIndex);
    correcetProb_eps = max(correcetProb,1e-8);
    loss = (-1) * mean(log(correcetProb_eps));
end