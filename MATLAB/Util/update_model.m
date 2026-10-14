function [model, velocity] = update_model(model, gradiants, lr, momentum, velocity)
% leetttttts update this model
    
%{
    model.W1 = model.W1 - lr*gradiants.W1;
    model.b1 = model.b1 - lr*gradiants.b1;
    model.W2 = model.W2 - lr*gradiants.W2;
    model.b2 = model.b2 - lr*gradiants.b2;
%}
    velocity.W1 = momentum*velocity.W1 - lr*gradiants.W1;
    velocity.b1 = momentum*velocity.b1 - lr*gradiants.b1;
    velocity.W2 = momentum*velocity.W2 - lr*gradiants.W2;
    velocity.b2 = momentum*velocity.b2 - lr*gradiants.b2;

    model.W1 = model.W1 + velocity.W1;
    model.b1 = model.b1 + velocity.b1;
    model.W2 = model.W2 + velocity.W2;
    model.b2 = model.b2 + velocity.b2;

end