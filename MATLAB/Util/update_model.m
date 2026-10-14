function model = update_model(model, gradiants, opts)
% leetttttts update this model
    lr = opts.learnRate;

    model.W1 = model.W1 - lr*gradiants.W1;
    model.b1 = model.b1 - lr*gradiants.b1;
    model.W2 = model.W2 - lr*gradiants.W2;
    model.b2 = model.b2 - lr*gradiants.b2;

end