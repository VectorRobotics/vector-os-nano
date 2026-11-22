"""Build a pixel-only model from the declarative user-config tables."""
from importlib import import_module


def build(table, name):
    if name not in table:
        raise ValueError(f'unknown perception model {name!r}; valid: {list(table)}')
    module, factory = table[name].split(':')
    return getattr(import_module(module), factory)()
