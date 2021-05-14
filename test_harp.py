import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})


from graph.wiki import load_relation_data
from harp.harp import harp

def test_harp(market):
    '''Generates graph embeddings for specified market using HARP.
    Parameters:
        market (str): "NYSE" or "NASDAQ"
    Return:s
        embedding (numpy.ndarray) HARP graph embedding of the market.
    '''
    encoding, binary_encoding = load_relation_data('20180105', market.upper())

    options = {
        "embedding_model": "node2vec",
        "walk_length": 20
    }

    embedding = harp(binary_encoding, options)

    # assert encoding.shape == (1737, 1737, 33)
    # assert binary_encoding.shape == (1737, 1737)

    # if options["embedding_model"].lower() in ['deepwalk', 'node2vec']:
    #     assert embedding.shape == (1737, 128)
    # elif options["embedding_model"].lower() == "line":
    #     assert embedding.shape == (1737, 64)
    return embedding

def get_stock_listings(market):
    '''Returns the list of stocks in order in the market. By ordered, we mean
    if ret_arr[x] == "AAPL", then the embedding for AAPL would be 
    embedding[x].

    Parameters:
        market (str): NYSE or NASDAQ?
    Returns:
        stock_arr (list(str)): list of stocks in order.
    '''
    stock_arr = []
    f = open(f"data/{market.upper()}_tickers.csv", "r").read()
    for line in f.split("\n")[:-1]:
        stock_arr.append(line)
    return stock_arr

def index_of_stocks(stock_list, market):
    '''Given a list of stocks, returns the index of each stock in 
    our db. Purpose of this would be to find the embedding for that stock.

    Parameters:
        stock_list (list(str)): list of stock tickers. e.g. ["AAPL", "FB"]
    Returns:
        success_list (list(str)): list of stocks that could be matched to DB
        indices (list(int)): index of stock in our DB.
            EXAMPLE: embedding of stock == success_list[X] is embedding[indices[X]].
    '''
    stock_list.sort() # sort alphabetically
    success_list = []
    indices = []

    cur_search_idx = 0
    market_list = open(f"data/{market.upper()}_tickers.csv", "r").read().split("\n")
    for market_index, market_stock in enumerate(market_list):
        if market_stock == stock_list[cur_search_idx]:
            success_list.append(market_stock)
            indices.append(market_index)
            cur_search_idx += 1

            if cur_search_idx == len(stock_list):
                break
        
    return (success_list, indices)

def generate_tsne_harp(stock_listings, market, png_file_name):
    '''List of stocks in the market => t-SNE output.png graphing their embeddings.
    Parameters:
        stock_listings (list(str)): list of stocks to embed & make t-SNE plotsof
        market (str): the market these stocks come from.
        png_file_name (str): where to output the graph in png form. 
    Returns:
        void. produces tsne.png.
    '''

    # produce HARP embeddings
    embeddings = test_harp(market.upper())

    # get index of stocks to get embeddings from
    stock_listings, indices = index_of_stocks(stock_listings, market.upper())
    new_embedding = []
    for idx in indices:
        one_embed = embeddings[idx]
        new_embedding.append(one_embed)

    # embedding matrix of just our desired stocks
    embeddings = np.array(new_embedding)

    # run TSNE
    tsne = TSNE()
    X_embedded = tsne.fit_transform(embeddings)
    palette = sns.color_palette("bright", len(stock_listings))
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=stock_listings, legend='full', palette=palette)
    sns_plot.figure.savefig(png_file_name)

def test_tsne_harp():
    stock_listings = [
        "AAPL",
        "FB",
        "TMUX",
        "AMZN",
        "BIDU",
        "GOOG",
        "GOOGL",
        "MSFT",
        "NVDA",
        "PCAR",
        "PEP"
    ]
    generate_tsne_harp(stock_listings, "NASDAQ")

def test_tsne_harp_all():
    all_stocks = get_stock_listings("NYSE")
    generate_tsne_harp(all_stocks, "NYSE", "harp/tsne_all_nyse_harp_node2vec.png")
    all_stocks = get_stock_listings("NASDAQ")
    generate_tsne_harp(all_stocks, "NASDAQ", "harp/tsne_all_nasdaq_harp_nod2vec.png")
    
if __name__ == "__main__":
    # test_harp()
    test_tsne_harp_all()