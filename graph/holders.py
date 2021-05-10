import os
import numpy as np
import pandas as pd
import pickle
import yfinance as yf

from loguru import logger
from typing import List

NASDAQ_TICKER_DIR = "data/NASDAQ_tickers.csv"
NYSE_TICKER_DIR = "data/NYSE_tickers.csv"

def get_ticker_inst_df(market, ticker_list: List[str]):
    b_df_url = f"data/inst_holdings/{market}_ticker_to_holders.pkl" 
    holders_mst = []
    b_df = dict()

    # Step 1. Load or download [ticker]: (holder_name, percent) dictionary
    if os.path.isfile(b_df_url):
        logger.info(f"Found existing ticker-to-holdings db for '{market}' at: {b_df_url}")
        f = open(b_df_url, "rb")
        b_df = pickle.load(f)
        f.close()
    else:
        logger.info("Could not find existing ticker-to-holder db for '{market}'.\n Crawling...")
        for ticker in ticker_list:
            out = get_inst_holders(ticker)
            if not out:
                b_df[ticker] = None
            else:
                holders, percents = out
                b_df[ticker] = list(zip(holders, percents))
        f = open(b_df_url, "wb")
        pickle.dump(b_df, f)
        f.close()        

    # Step 2: Create holders_mst & save holders indexing file.
    for key, value in b_df.items():
        h_p = b_df[key]
        if h_p:
            h_p = list(h_p)
            b_df[key] = h_p
            for holder, percent in h_p:
                existing_holder_idx = -1
                for holder_idx, existing_holder in enumerate(holders_mst):
                    if existing_holder[0] == holder:
                        existing_holder_idx = holder_idx

                if existing_holder_idx == -1:
                    # this holder is not yet in holders_mst
                    holders_mst.append([holder, 1])
                else:
                    # existing holder in holder_mst
                    holders_mst[existing_holder_idx][1] += 1
    
    # Filtering holders:
    # Remove holders that hold only one stock in the exchange
    # i.e. we want holders that will make pairs of stocks by sharing the same holder.
    filtered_holders_mst = []
    for mst_holder in holders_mst:
        if mst_holder[1] == 1:
            # this holder only holds one stock
            filtered_holders_mst.append(mst_holder[0])
    holders_mst = filtered_holders_mst

    # Saving holders index file
    holders_index_file = "\n".join(holders_mst)
    f = open(f"data/inst_holdings/{market}_holders_index.txt", "w+")
    f.write(holders_index_file)
    f.close()
    logger.info(f"Saved holder index file for {market}.\nGenerating statistics...")

    # Encode holder information for each stock in usable arrays instea of dictionary strngs
    for ticker, h_p in b_df.items():
        encoding = [0] * len(holders_mst)
        if h_p:
            for holder, percent in h_p:
                try:
                    # might fail since we filtered holders that only hold one stock
                    # the "holder" variable might be the holder that only has one stock, causing .index() to fail
                    idx = holders_mst.index(holder)
                    encoding[idx] = percent
                except:
                    pass
        b_df[ticker] = encoding
    
    # Generate Statistics
    logger.info(f"Number of unique holders: {len(holders_mst)}")
    no_holders = []
    for key, value in b_df.items():
        binary_encoding = b_df[key]
        if list(set(binary_encoding)) == [0]:
            # it only has 0 values - no holders
            no_holders.append(key)
    logger.info(f"Number of tickers without holders: {len(no_holders)}")
    # logger.info(f"Tickers without holders: {no_holders}")

    return b_df

def get_inst_holders(ticker: str):
    tick = yf.Ticker(ticker)
    df = tick.institutional_holders

    try:
        holders = df["Holder"]
        percents = df["% Out"]
    except:
        return None
    return (holders.tolist(), percents.tolist())

def generate_relation_matrix(market):
    ticker_dir = f"data/{market.upper()}_tickers.csv"
    tickers = open(ticker_dir, "r").read().split("\n")
    b_df = get_ticker_inst_df(market.upper(), tickers)

    holders_list = open(f"data/inst_holdings/{market.upper()}_holders_index.txt").read().split("\n")

    full_matrix = np.zeros((len(tickers), len(tickers), len(holders_list)))
    logger.info(f"Dimensions of full matrix: {full_matrix.shape}")

    for holder_idx, holder in enumerate(holders_list):
        pairs = []
        matches = []
        for ticker_idx, ticker in enumerate(tickers):
            holding_percent = b_df[ticker][holder_idx]
            if holding_percent is not 0:
                matches.append((ticker_idx, holding_percent))

        for match_idx, match in enumerate(matches):
            if match_idx == len(matches) - 1:
                break
            remaining = matches[match_idx+1:]
            for one_remaining in remaining:
                one_pair = (match[0], one_remaining[0], match[1]*one_remaining[1])
                # one_pair = (match[0], one_remaining[0], 1)
                pairs.append(one_pair)

        for first_ticker_idx, second_ticker_idx, cor_percent in pairs:
            full_matrix[first_ticker_idx][second_ticker_idx] += cor_percent

    np.save(f"data/inst_holdings/{market.upper()}_inst_relation.npy", full_matrix)
    logger.info(f"Saved full relation matrix at: data/inst_holdings/{market.upper()}_inst_relation.npy")
    return full_matrix

def load_relation_npy(market):
    '''
        If .npy file found in data/inst_holdings/{market}_inst_relation.npy
        then simply load.

        If not, run optimized generation of the matrix, downloading data if need be.
    '''
    local_data_url = f"data/inst_holdings/{market}_inst_relation.npy"
    if os.path.isfile(local_data_url):
        logger.info(f"Found pre-existing relation matrix for {market}.\nLoading...")
        relation = np.load(local_data_url)
        logger.info(f"Loading complete. relation.shape={relation.shape}")
        return relation
    else:
        logger.info(f"Could not find pre-existing relation matrix for {market}.\nGenerating new...")
        relation = generate_relation_matrix(market)
        return relation

