# LSNA
## Introduction
This is the source code for *Unsupervised Large-Scale Social Network Alignment via Cross Network Embedding*. 

## Execute
* *initial_data_processing.py*: Transforming the raw test data into a suitable format for code to run.
* *partitioning.py*: Partition and alignment of large-scale networks.
* *main.py* : The main function to run the code.

### Example execute command
```
python initial_data_processing.py -n 5000 -g 5 -r 0.05 -d lastfm
```
```
python partitioning.py -d FT
```
```
python main.py -d flickr-flickr -n 5000
```

## Dependencies
* [Python](https://www.python.org/)
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [networkx](http://networkx.github.io/)
* [grakel](https://ysig.github.io/GraKeL/0.1a8/index.html)

## Data
### Raw datasets
We used two public social network datasets in the paper: 

* *Facebook-Twitter*
* *Weibo-Douban*

You can download the complete datasets from [here](http://apex.sjtu.edu.cn/datasets/8). 

### Test data

In order to test the code conveniently, we put small-scale test data in *-data/graph_edge*.
