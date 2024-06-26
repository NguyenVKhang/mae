Please switch to the master branch on git to see my custom code, data, and results.
# BiGRec
The BiGRec model utilizes PyTorch, you can read the details of the model at [BigRec](https://www.overleaf.com/read/ydkdtxsmmwkc#694d14)

![image](https://github.com/NguyenVKhang/mae/assets/100186039/4ecebc10-819a-4c98-8fac-9bef143560cc)

### 1. Introduction

The BiGRec model is a bipartite graph-based recommender system that leverages contrastive learning and transformers for enhanced recommendation performance.

### 2. Environment



We suggest the following environment for running MAERec:

```
python==3.8.13
pytorch==1.12.1
numpy==1.18.1
```

### 3. How to run

Please first unzip the desired dataset in the dataset folder, and then run

- Amazon Books: `python main.py --data books`
- Amazon Toys: `python main.py --data toys`
- Retailrocket: `python main.py --data retailrocket`

More explanation of model hyper-parameters can be found [here](./params.py).

### 4. Running on customized datasets

The dataset should be structured into four files for MAERec:

- `seq` a list of user sequences for training
- `tst` a list of user sequences for testing
- `neg` the negative samples for each test sequence
- `trn` the i-i graph, which can be generated using the following code:

```Python
def construct_graphs(num_items=54756, distance=3, path='./books/'):
    with open(path + 'seq', 'rb') as fs:
        seq = pickle.load(fs)
    user = list()
    r, c, d = list(), list(), list()
    for i, seq in enumerate(seqs):
        print(f"Processing {i}/{len(seqs)}          ", end='\r')
        for dist in range(1, distance + 1):
            if dist >= len(seq): break;
            r += copy.deepcopy(seq[+dist:])
            c += copy.deepcopy(seq[:-dist])
            r += copy.deepcopy(seq[:-dist])
            c += copy.deepcopy(seq[+dist:])
    d = np.ones_like(r)
    iigraph = csr_matrix((d, (r, c)), shape=(num_items, num_items))
    print('Constructed i-i graph, density=%.6f' % (len(d) / (num_items ** 2)))
    with open(prefix + 'trn', 'wb') as fs:
        pickle.dump(iigraph, fs)
```

