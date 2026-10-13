"""The layers of the package import only downwards: data <- physics <- learning (<- service)."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "temsim"
ORDER = ["data", "physics", "learning", "service"]


def imported_layers(path):
    package = path.relative_to(ROOT.parent).with_suffix("").parts[:-1]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".") for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package[: len(package) - node.level + 1])
                names = [base + (node.module.split(".") if node.module else [])]
            else:
                names = [(node.module or "").split(".")]
        for parts in names:
            if parts[0] == "temsim" and len(parts) > 1 and parts[1] in ORDER:
                found.add(parts[1])
            if parts[0] in ("temsim_gui", "gui_qt", "PySide6"):
                found.add("gui")
    return found


def test_import_direction():
    for layer in ORDER:
        for path in (ROOT / layer).rglob("*.py"):
            found = imported_layers(path)
            assert "gui" not in found, path
            for other in found:
                assert ORDER.index(other) <= ORDER.index(layer), f"{path} imports {other}"


def test_data_layer_is_numpy_only():
    for path in (ROOT / "data").rglob("*.py"):
        text = path.read_text()
        assert "import torch" not in text and "from torch" not in text, path


def _all_imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


def test_client_side_modules_are_light():
    forbidden = ("torch", "..physics", "..learning", ".session", ".server", ".jobs", ".assemble")
    for name in ("protocol.py", "client.py"):
        imports = _all_imports(ROOT / "service" / name)
        bad = [i for i in imports if any(i == f or i.startswith(f + ".") for f in forbidden)]
        assert not bad, f"{name} imports {bad}"


def test_interface_talks_to_the_core_only_through_the_protocol():
    allowed = ("temsim.data", "temsim.service.client", "temsim.service.protocol")
    for path in (ROOT.parent / "temsim_gui").rglob("*.py"):
        for name in _all_imports(path):
            assert name.split(".")[0] != "torch", f"{path} imports torch"
            if name.startswith("temsim"):
                assert any(name == a or name.startswith(a + ".") for a in allowed), f"{path} imports {name}"
