import torch
import torch.nn as nn
from train import ranking_mse_loss, criterion

def test(model, test_loader, args):
    criterion = nn.MSELoss()

    model.eval()
    test_loss = []
    with torch.no_grad():
        epoch_val_loss = []
        for batch_idx, (x_test, y_test) in enumerate(test_loader):
            out = model(x_test)
            pred_return_ratio = out 

            loss = ranking_mse_loss(criterion, pred_return_ratio, y_test, args.alpha)

            test_loss.append(loss.item())
    return test_loss