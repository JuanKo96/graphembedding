'''
    HARP Implementation based on paper: "HARP: Hierarchical Representation 
    Learning for Networks" (https://arxiv.org/pdf/1706.07845.pdf)       

    Implementation Details:
'''

import os
import glob
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

import numpy
import networkx as nx
from loguru import logger

from harp.magicgraph import from_numpy, WeightedDiGraph, WeightedNode
import harp.graph_coarsening as graph_coarsening


def harp(G_input: numpy.ndarray, options=None) -> numpy.ndarray:
    '''Uses HARP to generate an embedding matrix for G_input, the adjacency numpy matrix.

    Parameters:
        G_input 
        options (Optional[dict]): {
            embedding_model: ["deepwalk", "node2vec", "line"],
            sfdp_path,
            number_walks,
            walk_length
            representation_size,
            window_size
        }
    Returns:
        embeddings (numpy.ndarray): 
            Generated embedding matrix of dimension: (G_input.shape[0], 128)

    '''
    G_input = from_numpy(G_input)

    G = graph_coarsening.DoubleWeightedDiGraph(G_input)
    logger.info('Number of nodes: {}'.format(G.number_of_nodes()))
    logger.info('Number of edges: {}'.format(G.number_of_edges()))

    parser = ArgumentParser('harp',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
                            
    parser.add_argument('--sfdp-path', default='./bin/sfdp_osx',
                        help='Path to the SFDP binary file which produces graph coarsening results.')
    parser.add_argument('--model', default='deepwalk',
                        help='Embedding model to use. Could be deepwalk, line or node2vec.')
    parser.add_argument('--number-walks', default=40, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--output', required=False,
                        help='Output representation file')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--walk-length', default=10, type=int,
                        help='Length of the random walk started at each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of the Skip-gram model.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of parallel processes.')
    args = parser.parse_args()

    if options:
        for key, value in options.items():
            setattr(args, key, value)

    if args.embedding_model == 'deepwalk':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(
            G,
            scale=-1,
            iter_count=1,
            sfdp_path=args.sfdp_path,
            num_paths=args.number_walks,
            path_length=args.walk_length,
            representation_size=args.representation_size,
            window_size=args.window_size,
            lr_scheme='default',
            alpha=0.025,
            min_alpha=0.001,
            sg=1,
            hs=1,
            coarsening_scheme=20, 
            sample=0.1
        )
    elif args.embedding_model == 'node2vec':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(
            G,
            scale=-1,
            iter_count=1,
            sfdp_path=args.sfdp_path,
            num_paths=args.number_walks,
            path_length=args.walk_length,
            representation_size=args.representation_size,
            window_size=args.window_size,
            lr_scheme='default',
            alpha=0.025, 
            min_alpha=0.001,
            sg=1,
            hs=0,
            coarsening_scheme=2, 
            sample=0.1
        )
    elif args.embedding_model == 'line':
        embeddings = graph_coarsening.skipgram_coarsening_disconnected(
            G,
            scale=1, 
            iter_count=50,
            sfdp_path=args.sfdp_path,
            representation_size=64,
            window_size=1,
            lr_scheme='default',
            alpha=0.025, 
            min_alpha=0.001,
            sg=1,
            hs=0,
            sample=0.001
        )
    else:
        logger.error("Only supported embedding models are one of: ['deepwalk', 'node2vec', 'line']")
        return None

    # This whole process generates external "walk" files to track the random walks.
    # Code below cleans up our directory resulting from the above process.
    wild = os.getcwd() + "/default.walks.*"
    matches = glob.glob(wild)
    for file_match in matches:
        os.remove(file_match)

    return embeddings

