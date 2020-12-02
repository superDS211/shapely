from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import attr
from operator import mul
from functools import reduce


@attr.s(auto_attribs=True, eq=False)
class Shape:
    value: Any = attr.ib(repr=False)
    maxlen: int = 3
    _tensors: Dict[int, Tuple[List[int], int]] = attr.ib(factory=dict, init=False)
    _parsed: Any = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._parsed = self._parse(self.value, maxlen=self.maxlen)

    def _get_torch_size_as_list(self, arg) -> List[int]:
        return [-1] + list(arg[1:]) if len(arg) > 3 else list(arg)

    def _get_tensor_size(self, sp: List[int]) -> int:
        return abs(reduce(mul, sp))

    @property
    def size(self) -> int:
        return sum(sz for _, sz in self._tensors.values())

    @property
    def tensor_sizes(self) -> Dict[int, int]:
        return {k: sz for k, (_, sz) in self._tensors.items()}

    @property
    def tensor_shape(self) -> List[int]:
        assert len(self._tensors) == 1, "Cannot get tensor_shape for multi-tensor shape"
        return list(self._tensors.values())[0][0]

    def dump(self):
        return self._parsed

    def __str__(self):
        return repr(self.dump())

    def _parse(self, arg, maxlen) -> Any:
        t = type(arg)
        if t is dict:
            return {k: self._parse(v, maxlen) for k, v in arg.items()}
        elif t is tuple or (t is list and maxlen >= len(arg)):
            return t(self._parse(v, maxlen) for v in arg)
        elif t is list and maxlen <= len(arg):
            return (f"L({len(arg)})", [self._parse(v, maxlen) for v in arg[:maxlen]])
        elif hasattr(arg, "shape"):
            # Note: if maxlen limits how deep we recurse into `value`,
            # we will not capture all tensors
            s = arg.shape
            if len(s):
                sz_lst = self._get_torch_size_as_list(s)
                # key by id() so that we can later uniquify
                self._tensors[id(arg)] = (sz_lst, self._get_tensor_size(sz_lst))
                return sz_lst
            else:
                # tensor of one item
                return arg
        elif arg.__class__.__module__ == "builtins":
            return arg
        else:
            return classname(arg)


def classname(arg) -> str:
    return f"{arg.__class__.__module__}.{arg.__class__.__qualname__}"


def shape(arg, maxlen=3) -> Shape:
    return Shape(arg, maxlen=maxlen)
