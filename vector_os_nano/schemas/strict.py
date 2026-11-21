# SPDX-License-Identifier: Apache-2.0
"""Safe YAML with duplicate keys rejected before they can hide invalid values."""
from pathlib import Path
from typing import Any


def yaml_loader(path: Path, error: type[ValueError]) -> Any:
    import yaml

    class Strict(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            seen: set[str] = set()
            for key_node, _value in node.value:
                key = str(getattr(key_node, "value", ""))
                if key in seen:
                    raise error(f"{path}: duplicate key {key} (line {key_node.start_mark.line + 1}): "
                                "a key appears once")
                seen.add(key)
            return super().construct_mapping(node, deep)

    return Strict
