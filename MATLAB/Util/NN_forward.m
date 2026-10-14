function b_cache = NN_forward(model,XBatch)

    hiddenScore = XBatch * model.W1 + mode.b1;
    hiddenVal = max(0, hiddenScore);

    scores = hiddenVal*model.W2 + model.b2;

    prob = softMax(scores);

    b_cache.XBatch = XBatch;
    b_cache.hiddenScore = hiddenScore;
    b_cache.hiddenVal = hiddenVal;
    b_cache.scores = scores;
    b_cache.prob = prob;
end