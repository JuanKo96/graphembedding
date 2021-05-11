import os
import sys

from graph.wiki import load_relation_data
from harp.harp import harp

def test_harp():
    encoding, binary_encoding = load_relation_data('20180105', 'NYSE')

    options = {
        "embedding_model": "line",
        "sfdp_path": "bin/sfdp_linux",
        "number_walks": 40,
        "walk_length": 10,
        "representation_size": 128,
        "window_size": 10
    }

    embedding = harp(binary_encoding, options)

    assert encoding.shape == (1737, 1737, 33)
    assert binary_encoding.shape == (1737, 1737)

    if options["embedding_model"].lower() in ['deepwalk', 'node2vec']:
        assert embedding.shape == (1737, 128)
    elif options["embedding_model"].lower() == "line":
        assert embedding.shape == (1737, 64)
    
if __name__ == "__main__":
    test_harp()