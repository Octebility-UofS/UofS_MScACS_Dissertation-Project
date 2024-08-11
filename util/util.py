from collections.abc import Iterable
from functools import reduce
from operator import mul
import pickle
from typing import Any, TypeVar, Union, overload

import matplotlib.pyplot as plt
import numpy as np

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
@overload
def rec_frozenset(mapping: dict[K, V]) -> frozenset[tuple[K, V]]: ...
@overload
def rec_frozenset(mapping: T) -> T: ...

def rec_frozenset(mapping: Union[dict[K, V],T]) -> Union[frozenset[tuple[K, V]],T]:
    if isinstance(mapping, dict):
        return frozenset( (k, rec_frozenset(v)) for k,v in mapping.items() )
    elif isinstance(mapping, list):
        return tuple( rec_frozenset(e) for e in mapping )
    else:
        return mapping
    
def nary_sequences(*sequences: Iterable[list[Any]]) -> list[Iterable[Any]]:
    seq_lens = [ len(seq) for seq in sequences ]
    curr_ixs = [ 0 for _ in sequences ]
    gen = []
    while curr_ixs[0] < seq_lens[0]:
        gen.append([ seq[curr_ixs[ix]] for ix, seq in enumerate(sequences) ])
        curr_ixs[-1] += 1
        for ix in reversed(range(len(curr_ixs))):
            if curr_ixs[ix] >= seq_lens[ix]:
                if ix == 0:
                    break
                else:
                    curr_ixs[ix-1] += 1
                    curr_ixs[ix] = 0
    return gen


def file_write(file_name: str, content: str, append=False):
    with open(file_name, 'a' if append else 'w') as f:
        f.write(content)

def pickle_dump(file_name: str, content):
    with open(file_name, 'wb') as f:
        pickle.dump(content, f)


class LinePlot:
    def __init__(self, x_label, y_label, **kwargs):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def add(self, x_data, y_data, **kwargs):
        self.ax.plot(x_data, y_data, **kwargs)

    def save(self, save_path: str):
        self.ax.legend()
        self.fig.savefig(save_path)
        plt.close(self.fig)
        

class HeatMatrix:
    def __init__(self, data_matrix, xlabels, ylabels, **kwargs):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        self.ax.matshow(data_matrix, cmap=plt.cm.Blues)
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                c = np.round(data_matrix[i, j], 2)
                self.ax.text(i, j, str(c), va='center', ha='center')
        self.ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
        self.ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)

    def save(self, save_path: str):
        self.fig.savefig(save_path)
        plt.close(self.fig)