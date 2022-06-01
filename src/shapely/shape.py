from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, ClassVar, Optional, Sequence


@dataclass
class Shape:
    parse: Any

    def __repr__(self):
        return repr(self.parse)


@dataclass(repr=False)
class ShortCollectionShape(Shape):
    parse: Sequence[Shape]


@dataclass(repr=False)
class ElidedShape(Shape):
    type: Any
    len: int

    def __repr__(self):
        return f"{shortcode(self.type)}({self.len}) {repr(self.parse)}"


@dataclass(repr=False)
class DictShape(ElidedShape):
    parse: dict[Any, Shape]


@dataclass(repr=False)
class ListShape(ElidedShape):
    parse: list[Shape]


@dataclass(repr=False)
class TensorShape(Shape):
    parse: list[int]
    tensor_id: int

    @property
    def size(self) -> int:
        return abs(reduce(mul, self.parse))


@dataclass(repr=False)
class ClassShape(Shape):
    parse: str


def shortcode(arg) -> str:
    if arg is dict:
        return "D"
    elif arg is list:
        return "L"
    else:
        return str(arg)


@dataclass(eq=False, repr=False)
class ShapeParser:
    # This constant exists so that we can override the default behavior from "outside"
    MAXLEN: ClassVar[int] = 6

    arg: Any = field(repr=False)
    maxlen: Optional[int] = MAXLEN
    # _tensors: Dict[int, Tuple[List[int], int]] = field(default_factory=dict, init=False)
    parsed_shape: Shape = field(init=False)

    def __post_init__(self):
        self.parsed_shape = self._parse(self.arg, maxlen=(self.maxlen or self.MAXLEN))

    def __repr__(self):
        return repr(self.parsed_shape)

    def _parse(self, arg, maxlen: int) -> Shape:
        t = type(arg)
        if t is dict:
            return DictShape(
                {k: self._parse(v, maxlen) for k, v in list(arg.items())[:maxlen]},
                type=t,
                len=len(arg),
            )
        elif t is tuple or (t is list and maxlen >= len(arg)):
            return ShortCollectionShape(t(self._parse(v, maxlen) for v in arg))
        elif t is list and maxlen <= len(arg):
            return ListShape(
                [self._parse(v, maxlen) for v in arg[:maxlen]],
                type=t,
                len=len(arg),
            )
        elif hasattr(arg, "shape"):
            # Note: if maxlen limits how deep we recurse into `arg`,
            # we will not capture all tensors
            s = arg.shape
            if len(s):
                return TensorShape(list(s), tensor_id=id(arg))
            else:
                # tensor of one item
                return Shape(arg)
        elif arg.__class__.__module__ == "builtins":
            return Shape(arg)
        else:
            return ClassShape(classname(arg))


def classname(arg) -> str:
    return f"{arg.__class__.__module__}.{arg.__class__.__qualname__}"


def shape(arg, maxlen: Optional[int] = None) -> ShapeParser:
    return ShapeParser(arg, maxlen=maxlen)

    # def _get_tensor_size(self, sp: List[int]) -> int:
    #     pass

    # @property
    # def size(self) -> int:
    #     return sum(sz for _, sz in self._tensors.values())

    # @property
    # def tensor_sizes(self) -> Dict[int, int]:
    #     return {k: sz for k, (_, sz) in self._tensors.items()}

    # @property
    # def tensor_shape(self) -> List[int]:
    #     assert len(self._tensors) == 1, "Cannot get tensor_shape for multi-tensor shape"
    #     return list(self._tensors.values())[0][0]
