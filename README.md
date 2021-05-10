# GraphEmbedding

## Loading Graphs

### Wiki

- [x] Loading adjacency matrix for 20180105, both NASDAQ and NYSE
- [ ] Downloading wikidata -> Generating an adjacency matrix for most recent dump.

The second part's been tough because the dumps are 90GB+ with 5+ hours to even download and I haven't even been able to unzip the file yet.

#### Code
Based on: [Temporal Relational Ranking for Stock Prediction](https://arxiv.org/abs/1809.09441)

The following code allows you to download a numpy array of the following shape: `(len(tickers), len(tickers), len(wiki_relationship_types))`.

The paper states: "there are 42 and 32 types of company relations occurring between stock pairs in NASDAQ and NYSE, respectively." But loading NYSE data will give you a numpy array of size `(1737, 1737, 33)`. Not really sure where the extra relation came from, especially since **they don't provide an indexing file of sorts to tell us which index of the 3rd dimension of the array corresponds to which relation**. They do provide a list of relations, 58 of them, but it's meant for both markets.


```
def load_relation_npy(date: str, market: str)
```

Example Use:
```
from graph.wiki import load_relation_npy

relation = load_relation_npy('20180105', 'NYSE')
print(relation.shape) # (1737, 1737, 33)
```

### Institutional Holdings
Inspired by WIND (from [this paper](https://dl.acm.org/doi/abs/10.1145/3269206.3269269) - I actually can't access this paper anymore; I don't know how but I do have the original paper saved locally), but not using their data.

Essentially used Yahoo Finance to find institutional holdings data of companies in our markets. For instance a certain company might be held by the following companies:

```
Capital Advisors Inc/ok
Greenlight Capital, Inc.
Johnson & Johnson Innovation - JJDC, Inc.
Altium Capital Management, LP
Ohio-Public Employees Retirement System (PERS)
South Dakota Investment Council
...
```

and the adjacency matrix my code returns basically shows pairs of stocks that share the same institutional holder. The value in the corresponding spot in the array is a multiple of that holder's percent share in stock A and the percent share in stock B.

For example, say Blackrock holds 6% of Google and 5% of Facebook. Also suppose that Blackrock is the 2nd institutional holder in our list of 619 holders. Then:

|          | Google    | Facebook  |
|----------|-----------|-----------|
| Google   |           | [0, 0.06*0.05, 0, 0, ...] |
| Facebook | [0, 0.06*0.05, 0, 0, ...] |           |


#### Code

```
from graph.holders import load_relation_npy

holder_relations = load_relation_npy("NYSE")
print(holder_relations.shape) # (1738, 1738, 619)
```

#### Details
 - This code will save an .npy file of the above matrix at data/inst_holdings/, and while the data that are used to generate the matrix (e.g. data/inst_holdings/{market}_ticker_to_holders.pkl) is saved as part of the repo, the .npy file is not because it's massive. It's faster to just generate it on the fly based on the ticker-to-holders dictionary in the repo. Making the ticker-to-holders dictionary *does* take a long time because it requires yfinance queries, so thats why this one is part of the repo.
 - data/inst_holdings/{market}_holders_index.txt is a list of institutional holders for each market. You saw above that NYSE had 619 (as the 3rd dimension of the numpy array). The index.txt file has 619 lines, each line corresponding to the index of the numpy array.
