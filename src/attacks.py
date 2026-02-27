import torch.nn as nn

def fgsm_attack(model, X, y, epsilon):
    model.zero_grad()
    X_adv = X.clone().requires_grad_()
    y_adv = model.forward(X_adv).squeeze()
    criterion = nn.MSELoss()
    loss = criterion(y_adv, y)
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    return X_adv.detach()
