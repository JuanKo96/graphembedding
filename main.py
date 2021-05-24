import argparse

import torch
import pickle
from loguru import logger

from train import train
from dataloader import get_dataloader
from graph.wiki import get_encodings
from rankingmodel.models import TemporalSAGE

logger = logger.opt(colors=True)
NASDAQ_PICKLE_URL = 'data/nasdaq_test_price.pickle'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_embed_size', default=64)
    parser.add_argument('--k_hops', default=3)
    parser.add_argument('--hop_layers', default=3)
    parser.add_argument('--lstm_layers', default=3)
    parser.add_argument('--fn_layers', default=3)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--weight_decay', default=1e-8)
    parser.add_argument('--alpha', default=0.6)

    parser.add_argument('--epochs', default=50)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--window_size', default=30)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    # maintaining the path stuff here allows the log output to have this information as well
    # which helps later on when looking at a log and feeling like retrieving the correspoinding model
    args.model_id = f'LSTM_{args.seq_embed_size}_seq_{args.window_size}_window'
    args.path = f'checkpoints/{args.model_id}.pt'
    args.logger_path = f'logs/{args.model_id}.log'
    args.session_data_path = f'sessiondata/{args.model_id}.pkl'
    logger.add(args.logger_path)
    
    

    if torch.cuda.is_available():
        args.device = "cuda:0"

    # Load Data
    logger.info("Loading data...")
    nasdaq_data = pickle.load(open(NASDAQ_PICKLE_URL, "rb"))
    model_tickers = nasdaq_data.columns.tolist()

    # Load dataloaders & graphs
    inputs, dataloaders = get_dataloader(
        data=nasdaq_data, 
        batch_size=args.batch_size, 
        window_size=args.window_size,
        device=args.device
    )
    N, n_features = inputs
    train_loader, test_loader, val_loader = dataloaders
    
    encoding, _ = get_encodings('20180105', 'NASDAQ', model_tickers)
    relational_encoding = torch.FloatTensor(encoding).to(args.device)

    # Initialize Model
    model = TemporalSAGE(
        input_shape=[args.batch_size, args.window_size, N, n_features],
        seq_embed_size=args.seq_embed_size, 
        relational_encoding=relational_encoding, 
        k_hops=args.k_hops, 
        hop_layers=args.hop_layers, 
        lstm_layers=args.lstm_layers,
        fn_layers=args.fn_layers,
        device=args.device
    )
    logger.info(f"\n{model}")    
    
    logger.info(f"Starting training with args:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    train_loss, val_loss = train(model=model, train_loader=train_loader, val_loader=val_loader, args=args)
    test_loss = test(model=model, test_loader=test_loader, args=args)

    session_info = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "args": args
    }
    f = open(args.session_data_path, "rb")
    pickle.dump(session_info, f)
    f.close()

if __name__ == "__main__":
    main()