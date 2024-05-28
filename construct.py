from scipy.sparse import csr_matrix
from params import args
import numpy as np
import pickle
import copy
import os

def construct_graphs(seqs, num_items, distance, prefix):
    user = list()
    r, c, d = list(), list(), list()
    for i, seq in enumerate(seqs):
        print(f"Processing {i}/{len(seqs)} (>﹏<)    ", end='\r')
        for dist in range(1, distance + 1):
            if dist >= len(seq): break;
            #head->tail
            r += copy.deepcopy(seq[:-dist])
            c += list(map(lambda x: x + num_items, copy.deepcopy(seq[+dist:])))
            r += list(map(lambda x: x + num_items, copy.deepcopy(seq[+dist:])))
            c += copy.deepcopy(seq[:-dist])

    d = np.ones_like(r)
    print(len(r))
    iigraph = csr_matrix((d, (r, c)), shape=(num_items*2, num_items*2))
    print('Constructed i-i graph, density=%.6f' % (len(d) / ((num_items*2+2) ** 2)))
    print(iigraph)
    print(iigraph.shape)
    with open(prefix + 'trn', 'wb') as fs:
        pickle.dump(iigraph, fs)

if __name__ == '__main__':

    # dataset = input('Choose a dataset: ')
    dataset = args.data
    prefix = './datasets/' + dataset + '/'

    # distance  = int(input('Max distance of edge: '))
    distance = 3

    with open(prefix + 'seq', 'rb') as fs:
        seqs = pickle.load(fs)

    if dataset == ('books'):
        num_items = 54755
    elif dataset == ('toys'):
        num_items = 11924
    elif dataset == ('retailrocket'):
        num_items = 43885
    elif dataset == ('Beauty'):
        num_items = 12102

    print(seqs[0])
    print(seqs[0][+1:])
    print(seqs[0][:-1])
    construct_graphs(seqs, num_items, distance, prefix)
