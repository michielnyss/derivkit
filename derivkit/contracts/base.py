import inspect


class BaseContract:
    """
    Base class for all derivative contracts.

    Subclasses implement price(paths) and optionally extra methods
    (e.g. fair_strike for variance swaps).
    """

    def price(self, paths):
        raise NotImplementedError

    def __repr__(self):
        sig = inspect.signature(self.__class__.__init__)
        params = {k: getattr(self, k) for k in sig.parameters if k != 'self'}
        param_str = ', '.join(f'{k}={v!r}' for k, v in params.items())
        return f'{self.__class__.__name__}({param_str})'
