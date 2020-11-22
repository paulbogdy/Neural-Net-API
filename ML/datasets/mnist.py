import idx2numpy as id2
import gzip


def read_mnist():
    return [id2.convert_from_file(gzip.open('ML/datasets/Mnist/train-images-idx3-ubyte.gz')) / 256,
            id2.convert_from_file(gzip.open('ML/datasets/Mnist/train-labels-idx1-ubyte.gz')),
            id2.convert_from_file(gzip.open('ML/datasets/Mnist/t10k-images-idx3-ubyte.gz')) / 256,
            id2.convert_from_file(gzip.open('ML/datasets/Mnist/t10k-labels-idx1-ubyte.gz'))]
