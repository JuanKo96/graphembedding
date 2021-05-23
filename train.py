import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger

logger = logger.opt(colors=True)

def ranking_mse_loss(mse_criterion, pred_rr, true_rr, alpha):
    assert true_rr.size() == pred_rr.size()

    mse_loss = mse_criterion(pred_rr, true_rr)
    
    batch_size, N = true_rr.size()  
    pred_rr_repeated = pred_rr.repeat_interleave(repeats=N, dim=1)
    pred_rr_repeated = pred_rr_repeated.reshape((batch_size, N, N))
    pred_pw_diff = pred_rr_repeated - pred_rr_repeated.transpose(1,2)

    true_rr_repeated = true_rr.repeat_interleave(repeats=N, dim=1)
    true_rr_repeated = true_rr_repeated.reshape((batch_size, N, N))
    true_pw_diff = true_rr_repeated - true_rr_repeated.transpose(1,2)

    ranking_loss = torch.mean(nn.ReLU(pred_pw_diff.mul(true_pw_diff)))

    loss = mse_loss*alpha + ranking_loss*(1.0-alpha)
    return loss
    
def train(model, train_loader, val_loader, args):
    path = f'checkpoints/LSTM_{args.seq_embed_size}_seq_embed_size_{args.window_size}_window'
    # Loss function & optimizer. Should this be written in main.py or train.py? 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    patience = 0
    min_loss = np.inf
    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = []
        for batch_idx, (x_train, y_train) in enumerate(train_loader):    
                 
            # y_train = true return ratio. changed the code from previous just list of ranking indices 
            out = model(x_train)
            out = out.reshape((batch_size, N))
            
            pred_return_ratio = out # !! Edit this later maybe to try out = price prediction => calculate pred RR with pred price => MSE with RR
             # difference between model pred return ratio to the true return ratio
            
            # Now calculate ranking loss
            loss = ranking_mse_loss(criterion, pred_return_ratio, y_train, args.alpha)
            
            epoch_loss.append(loss.item())

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            optimizer.step()

        train_loss.append(np.mean(epoch_loss))
        epoch_loss = []      
        torch.cuda.empty_cache() ## 캐시 비워주기 자주 해줘야함
        
        # Validation
        model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            for batch_idx, (x_val, y_val) in enumerate(val_loader):
                out = model(x_val)

                loss = ranking_mse_loss(criterion, pred_return_ratio, y_train, args.alpha)

                epoch_val_loss.append(loss.item())
                val_loss.append(np.mean(epoch_val_loss))

        if epoch % 100 == 0: 
            logger.info('Epoch: {}, Train Loss: {:.4e}, Valid Loss: {:.4e}'.format(epoch, train_loss[-1], val_loss[-1]))

        # Update minimum loss
        if min_loss > train_loss[-1]:
            patience = 0
            min_loss = train_loss[-1]
            torch.save(model.state_dict(), path)
        else:
            patience += 1

        # Early stop when patience become larger than 10
        if patience > 10:
            break
            
        torch.cuda.empty_cache() ## 캐시 비워주기 자주 해줘야함
        