from collections.abc import Iterable
from functools import reduce
from operator import mul
from typing import Any, TypeVar, Union, overload

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