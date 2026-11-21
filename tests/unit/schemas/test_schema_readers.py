"""Typed fields have outside consumers, directly or through consumed pure data methods."""
import ast
import dataclasses
import inspect
from pathlib import Path

from vector_os_nano.schemas import robot, world


def test_robot_and_world_fields_have_readers():
    root = Path(__file__).resolve().parents[3]
    readers = set()
    for folder in ('vector_os_nano', 'scripts'):
        for path in (root / folder).rglob('*.py'):
            if 'schemas' not in path.parts:
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Attribute):
                        readers.add(node.attr)
                    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
                          and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                          and isinstance(node.args[1].value, str)):
                        readers.add(node.args[1].value)
    for module in (robot, world):
        methods = {node.name: {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                              and isinstance(n.value, ast.Name) and n.value.id == 'self'}
                   for cls in ast.parse(Path(module.__file__).read_text()).body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef)}
        while True:
            through_methods = set().union(*(attrs for name, attrs in methods.items() if name in readers))
            if through_methods <= readers:
                break
            readers |= through_methods
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and dataclasses.is_dataclass(cls):
                for field in dataclasses.fields(cls):
                    assert field.name in readers, f'{cls.__name__}.{field.name} has no reader outside schemas/'
