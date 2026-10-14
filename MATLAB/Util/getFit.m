function [fitLoss, fitAcc] = getFit(model,XFit,yFit)
    % getting fit without my glp1
    fitProb = NN_forward(model,XFit);

    fitLoss = NN_cross_entropy(fitProb.prob,yFit);

    [~, fitClassIdx] = max(fitProb.prob, [], 2);
    fitPredict = fitClassIdx - 1;

    fitAcc = mean(fitPredict == yFit);

end
