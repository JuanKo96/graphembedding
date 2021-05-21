import os
import re
import gzip
import json
import shutil
from pathlib import Path
import requests

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger

WIKI_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz"
WIKI_PARENT_SITE = "https://dumps.wikimedia.org/wikidatawiki/entities/";

def wiki_latest_date():
    
    '''Extracts latest date available in wikidata database.
    Parameters:
    Returns:
        latest_date (str): "20210507" (example output)

        If we are downloading the latest wiki data, find out which date it is up to.
        Example ouput from html:
            ...
            <a href="20210503/">20210503/</a>
            <a href="20210505/">20210505/</a>
            <a href="20210507/">20210507/</a>
            <a href="dcatap.rdf">dcatap.rdf</a>
            ...
        
        Want to extract "20210507" here.
    '''
    html = requests.get(WIKI_PARENT_SITE).text
    soup = BeautifulSoup(html)
    links = list(soup.find_all("a"))

    last_idx = -1
    for idx, link in links[1:]:
        match = re.search(link.text, "^[0-9]{8}/$")
        if match is None:
            # this is the "dcatap" thing
            last_idx = idx - 1
            break
    
    date_value = links[last_idx].replace("/", "") # "20210322/" -> "20210322"
    return date_value
    

def download(url: str, fname: str, file_size_offline=None):
    resume_header = ({'Range': 'bytes=%d-' % file_size_offline} if file_size_offline else None)
    resp = requests.get(url, stream=True, headers=resume_header)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def load_wiki():    
    CACHE_WIKI = "cache/latest-all.pkl"
    CACHE_GZIP = "cache/latest-all.json.gzip"
    CACHE_JSON = "cache/latest-all.json"

    if os.path.isfile(CACHE_WIKI):
        logger.info(f"Found pickled cache at {CACHE_WIKI}")
        pkl_f = open(CACHE_WIKI, 'rb')
        json_read = pickle.load(pkl_f)
        pkl_f.close()
        return json_read
    else:   
        logger.info(f"Could not find pickled cache at {CACHE_WIKI}")

        if not os.path.isdir("cache/"):
            os.mkdir("cache/")
        
        if not os.path.isfile(CACHE_JSON) and not os.path.isfile(CACHE_GZIP):
            logger.info(f"Could not find gzip WIKI file... downloading most recent at {WIKI_URL}")
            #download(WIKI_URL, CACHE_GZIP)
            logger.info("Most recent WIKI data download complete.")
        
        if os.path.isfile(CACHE_GZIP) and not os.path.isfile(CACHE_JSON):  
            # is it a complete download or just partially downloaded (interrupted in the middle?)
            r = requests.head(WIKI_URL)
            # Get filesize of online data
            file_size_online = int(r.headers.get('content-length', 0))
            file_size_offline = Path(CACHE_GZIP).stat().st_size
            if file_size_online != file_size_offline:
                # incomplte download, resume gzip download.
                logger.info(f"Found incomplete download... (sought {file_size_online} bytes)\nRestarting download at {file_size_offline} bytes")
                #download(WIKI_URL, CACHE_GZIP, file_size_offline)



            logger.info(f"Could not find json WIKI file... unzipping at {CACHE_GZIP}")
            with gzip.open(CACHE_GZIP, 'rb') as f_in:
                with open(CACHE_JSON, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info("Unzip complete.")

        # Read JSON file
        json_f = open(CACHE_JSON,) 
        json_read = json.load(json_f)
        json_f.close()

        # Pickle JSON file
        pkl_f = open(CACHE_WIKI, 'ab')
        pickle.dump(json_read, pkl_f)
        pkl_f.close()

        logger.info(f"Pickled json saved at {CACHE_WIKI}")

        return json_read
        
def load_relation_npy(date, market):
    if os.path.isdir(f"data/wiki/{date}"):
        market_npy_url = f"data/wiki/{date}/{market.upper()}_wiki_relation.npy"
        if os.path.isfile(market_npy_url):
            wiki_relation_np = np.load(market_npy_url)
            return wiki_relation_np
    logger.error(f"Could not find desired pre-saved wikidata relation file for date: {date}, market: {market}.\nConsider downloading the most recent version.")
    return None

def load_relation_data(date, market):
    if os.path.isdir(f"data/wiki/{date}"):
        market_npy_url = f"data/wiki/{date}/{market.upper()}_wiki_relation.npy"
            
        relation_encoding = np.load(market_npy_url)
        rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
        mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                            np.sum(relation_encoding, axis=2))
        mask = np.where(mask_flags, np.zeros(rel_shape, dtype=int), np.ones(rel_shape, dtype=int))
        return relation_encoding, mask
    return None

def get_encodings(date, market, model_tickers):
    
    # Get encoding and binary_encoding
    encoding, binary_encoding = load_relation_data(date, market)

    tickers_csv_url = f"data/{market.upper()}_tickers.csv"
    
    # Get tickers from csv as a list
    universe_tickers = pd.read_csv(tickers_csv_url, header=None)
    universe_tickers = universe_tickers.iloc[:,0].tolist()

    ticker_test = [i for i in model_tickers if i not in universe_tickers]

    # Check whether every ticker is in the market
    assert len(ticker_test) == 0, f"{ticker_test} not in {market}"

    # Make an index list for given tickers
    idx_list = []

    for ticker in model_tickers:
        idx_list.append(universe_tickers.index(ticker))
        
    # Make a new relation encoding with the given tickers only
    for i in range(encoding.shape[2]):
        temp_rel_encoding = encoding[:,:,i]
        
        temp_rel_encoding = temp_rel_encoding[np.ix_(idx_list,idx_list)]
        
        if i == 0:
            new_rel_encoding = temp_rel_encoding
        else:
            new_rel_encoding = np.dstack([new_rel_encoding, temp_rel_encoding])     
            
    new_binary_encoding = binary_encoding[np.ix_(idx_list,idx_list)]
    
    return new_rel_encoding, new_binary_encoding