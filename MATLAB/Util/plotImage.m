function [] = plotImage(model, xFit, imageRow)
if nargin < 3
    imageRow = 1;
end
pixelRow = xFit(imageRow,:);
imageData = reshape(pixelRow, 28, 28);

xArr = 1:1:28;
yArr = 1:1:28;

figure;
hold on;
surf(xArr, yArr, imageData)

b_cache = NN_forward(model,pixelRow);
[confidence, classIdx] = max(b_cache.prob, [], 2);
predictedDigit = classIdx - 1;

title(sprintf('Predicted digit: %d (Confidence %.1f%%)', predictedDigit, 100 * confidence));
end