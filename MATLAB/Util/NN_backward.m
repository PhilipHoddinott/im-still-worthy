function gradiants = NN_backward(model, b_cache, labels)

    numberOfRows = numel(labels);
    sz_prob = size(b_cache.prob);
    tempArr = (1:numberOfRows)';
    labelP1 = double(labels) + 1;
    
    correctIndex = sub2ind(sz_prob, tempArr, labelP1); % people wwill make fun of MATLAB for this line with the index startign at 1
    
    outGradiant = b_cache.prob;
    outGradiant(correctIndex) = outGradiant(correctIndex) -1;
    outGradiant = outGradiant/numberOfRows;
    
    % 2nd layer
    gradiants.W2 = b_cache.hiddenValues' * outGradiant;
    gradiants.b2 = sum(outGradiant,1);
    
    hiddenValGrad = outGradiant * model.W2';
    hiddenSocreGrad = hiddenValGrad .*(b_cache.hiddenScore > 0);
    gradiants.W1 = b_cache.X' * hiddenSocreGrad;
    gradiants.b1 = sum(hiddenSocreGrad,1);
end
