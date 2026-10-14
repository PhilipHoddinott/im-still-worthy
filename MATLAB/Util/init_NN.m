function [model, velocity] = init_NN(opts)
% "I work with models" "oh you're a photographer" "something like that"
inputSize = opts.inputSize;
hiddenSize = opts.hiddenSize;
outputSize = opts.outputSize;
rng(opts.seed)

W1 = randn(inputSize, hiddenSize, 'single') * sqrt(2/inputSize);

W2 = randn(hiddenSize, outputSize, 'single') * sqrt(2/hiddenSize);

b1 = zeros(1,hiddenSize, 'single');
b2 = zeros(1,outputSize, 'single');

model.W1 = W1;
model.b1 = b1;
model.W2 = W2;
model.b2 = b2;

model.inputSize = inputSize;
model.hiddenSize = hiddenSize;
model.outputSize = outputSize;

velocity.W1 = zeros(size(W1));
velocity.b1 = zeros(size(b1));
velocity.W2 = zeros(size(W2));
velocity.b2 = zeros(size(b2));

end