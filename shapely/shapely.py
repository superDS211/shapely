from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from dataclasses import dataclass, field


@dataclass(eq=False, repr=False)
class Shape:
    # This constant exists so that we can override the default behavior from "outside"
    MAXLEN: ClassVar[int] = 6

    value: Any = field(repr=False)
    maxlen: Optional[int] = MAXLEN
    _tensors: Dict[int, Tuple[List[int], int]] = field(default_factory=dict, init=False)
    _parsed: Any = field(init=False)

    def __post_init__(self):
        self._parsed = self._parse(self.value, maxlen=(self.maxlen or self.MAXLEN))

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

    def __repr__(self):
        return repr(self._parsed)

    def _parse(self, arg, maxlen: int) -> Any:
        t = type(arg)
        if t is dict:
            return (
                f"D({len(arg)})",
                {k: self._parse(v, maxlen) for k, v in list(arg.items())[:maxlen]},
            )
        elif t is tuple or (t is list and maxlen >= len(arg)):
            return t(self._parse(v, maxlen) for v in arg)
        elif t is list and maxlen <= len(arg):
            return (f"L({len(arg)})", [self._parse(v, maxlen) for v in arg[:maxlen]])
        elif hasattr(arg, "shape"):
            # Note: if maxlen limits how deep we recurse into `value`,
            # we will not capture all tensors
            s = arg.shape
            if len(s):
                sz_lst = list(s)
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


def shape(*arg, maxlen: Optional[int] = None) -> Shape:
    return Shape(arg, maxlen=maxlen)
