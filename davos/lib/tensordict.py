from collections import OrderedDict
import copy


# @init_overloads
class TensorDict(OrderedDict):
    """Container mainly used for dicts of torch tensors. Extends OrderedDict with pytorch functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def concat(self, other):
        """Concatenates two dicts without copying internal data."""
        return TensorDict(self, **other)

    def copy(self):
        return TensorDict(super(TensorDict, self).copy())

    def __deepcopy__(self, memodict={}):
        return TensorDict(copy.deepcopy(list(self), memodict))

    def __getattr__(self, name):

        # Get an example attribute, to see if it exists and whether it is callable
        attr = getattr(next(iter(self.values())), name, None)
        if attr is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        if callable(attr):
            # Assume all elements are callable and return a function that will call them.
            def apply_attr(*args, **kwargs):
                return TensorDict({n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e for n, e in self.items()})
            return apply_attr
        else:
            # Assume none of the elements are callable, return the attributes directly
            return TensorDict({n: getattr(e, name) if hasattr(e, name) else e for n, e in self.items()})

    def attribute(self, attr: str, *args):
        return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

    def apply(self, fn, *args, **kwargs):
        return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorDict, dict))

    @staticmethod
    def __unary_op(op, self):
        return self.__class__({k: op(v) for k, v in self.items()})

    @staticmethod
    def __binary_op(op, self, other):
        get_other = (lambda k: other[k]) if isinstance(other, dict) else (lambda k: other)
        return self.__class__({k: op(v, get_other(k)) for k, v in self.items()})

    @staticmethod
    def __inplace_binary_op(op, self, other):
        get_other = (lambda k: other[k]) if isinstance(other, dict) else (lambda k: other)
        for k, v in self.items():
            op(self[k], get_other(k))
        return self

    @classmethod
    def _init_overloads(cls):

        import operator

        for name in ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'pow', 'lshift', 'rshift']:
            op = getattr(operator, name)
            setattr(cls, f"__{name}__", lambda a, b, op=op: cls.__binary_op(op, a, b))
            setattr(cls, f"__r{name}__", lambda a, b, op=op: cls.__binary_op(op, a, b))

        for name in ['iadd', 'isub', 'imul', 'imatmul', 'itruediv', 'ifloordiv', 'imod', 'ipow', 'ilshift', 'irshift']:
            setattr(cls, f"__{name}__", lambda a, b, op=getattr(operator, name): cls.__inplace_binary_op(op, a, b))

        for name in ['lt', 'le', 'eq', 'ne', 'gt', 'ge']:
            setattr(cls, f"__{name}__", lambda a, b, op=getattr(operator, name): cls.__binary_op(op, a, b))

        for name in ['pos', 'neg', 'abs', 'invert']:
            setattr(cls, f"__{name}__", lambda a, op=getattr(operator, name): cls.__unary_op(op, a))
