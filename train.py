import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger

logger = logger.opt(colors=True)

def train(model, train_loader, val_loader, args):
    path = f'checkpoints/LSTM_{args.seq_embed_size}_seq_embed_size_{args.window_size}_window'
    # Loss function & optimizer. Should this be written in main.py or train.py? 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    patience = 0
    min_loss = np.inf
    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = []
        for batch_idx, (x_train, y_train) in enumerate(train_loader):            
            out = model(x_train)
            print(out)
            print(y_train)
            print(type(out))
            print(type(y_train))
            loss = criterion(out, y_train)
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

                loss = criterion(out, y_val)

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
        