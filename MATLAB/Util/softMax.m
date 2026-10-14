function probs = softMax(scores)

    shiftedScores = scores - max(scores, [] , 2);
    expos = exp(shiftedScores);

    probs = expos ./ sum(expos, 2);
end