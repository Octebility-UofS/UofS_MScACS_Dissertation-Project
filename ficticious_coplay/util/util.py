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