# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Source/installer checks runnable without torch, Triton, or a GPU."""

import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/flag_gems/runtime/backend/_hygon/experimental"
DOCS = ROOT / "docs/hygon-optimization"


def executable_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def load_installer():
    spec = importlib.util.spec_from_file_location(
        "hygon_inventory_test_loader", PACKAGE / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HygonInventoryTests(unittest.TestCase):
    def test_history_integrity_and_parse(self):
        manifest = json.loads((DOCS / "history/MANIFEST.json").read_text())
        self.assertEqual(manifest["source_count"], len(manifest["sources"]))
        self.assertEqual(len(manifest["operator_registry"]), 15)
        for item in manifest["sources"]:
            data = (ROOT / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            ast.parse(data.decode("utf-8-sig"))
            self.assertTrue(item["origins"])

    def test_selected_source_hashes_and_executable_equivalence(self):
        selected = json.loads((DOCS / "SELECTED_SOURCES.json").read_text())["sources"]
        history = json.loads((DOCS / "history/MANIFEST.json").read_text())["sources"]
        original_asts = {
            executable_ast((ROOT / item["path"]).read_text(encoding="utf-8-sig"))
            for item in history
        }
        pinned = json.loads((PACKAGE / "SOURCES.json").read_text())
        self.assertEqual(len(selected), 13)
        for item in selected:
            path = ROOT / item["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
            )
            self.assertEqual(
                pinned[path.relative_to(PACKAGE).as_posix()], item["sha256"]
            )
            self.assertIn(executable_ast(path.read_text()), original_asts)
            self.assertEqual(
                executable_ast(path.read_text()),
                executable_ast(
                    (ROOT / item["archived_source"]).read_text(encoding="utf-8-sig")
                ),
            )

    def test_import_does_not_import_device_modules(self):
        before = set(sys.modules)
        module = load_installer()
        self.assertIsNone(module._STATE)
        added = set(sys.modules) - before
        self.assertFalse({"torch", "triton", "flag_gems", "torch_fl"} & added)

    def test_request_validation(self):
        module = load_installer()
        fg = types.SimpleNamespace(
            vendor_name="hygon",
            current_work_registrar=None,
            _FULL_CONFIG=(),
            FULL_CONFIG_BY_FUNC={},
        )
        torch = types.SimpleNamespace(
            _C=types.SimpleNamespace(_get_privateuse1_backend_name=lambda: "flagos")
        )
        self.assertEqual(
            module._check_request(["matrix", "sort"], False, fg, torch),
            ("sort", "matrix"),
        )
        for groups in ["matrix", [], ["matrix", "matrix"], ["invalid"]]:
            with self.assertRaises(ValueError):
                module._check_request(groups, False, fg, torch)
        with self.assertRaises(ValueError):
            module._check_request(["under90"], False, fg, torch)
        fg.vendor_name = "nvidia"
        with self.assertRaises(RuntimeError):
            module._check_request(["matrix"], False, fg, torch)
        fg.vendor_name = "hygon"
        fg.current_work_registrar = object()
        with self.assertRaises(RuntimeError):
            module._check_request(["matrix"], False, fg, torch)
        fg.current_work_registrar = None
        torch._C._get_privateuse1_backend_name = lambda: "privateuseone"
        with self.assertRaises(RuntimeError):
            module._check_request(["index"], False, fg, torch)

    def test_refresh_preserves_schema_metadata_and_container_type(self):
        module = load_installer()
        old, new, other = object(), object(), object()
        metadata = object()
        for entries in [
            (("mm", old, metadata), ("unrelated", other)),
            [("mm", old, metadata)],
        ]:
            updated = module._refresh_entries(entries, {id(old): new})
            self.assertIs(type(updated), type(entries))
            self.assertIs(updated[0][1], new)
            self.assertIs(updated[0][2], metadata)
            self.assertIs(entries[0][1], old)

    def test_integrity_failure_is_detected(self):
        module = load_installer()
        module._verify_sources()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "changed.py").write_text("pass\n")
            (root / "SOURCES.json").write_text(json.dumps({"changed.py": "0" * 64}))
            module.__file__ = str(root / "__init__.py")
            with self.assertRaises(RuntimeError):
                module._verify_sources()

    def test_install_refreshes_both_registrar_views_and_refuses_second_install(self):
        module = load_installer()
        old, new = lambda: None, lambda: None
        fg = types.ModuleType("flag_gems")
        fg.vendor_name = "hygon"
        fg.current_work_registrar = None
        fg.mm = old
        fg._FULL_CONFIG = (("mm", old, "metadata"),)
        fg.FULL_CONFIG_BY_FUNC = {"mm": [("mm", old, "metadata")]}
        torch = types.ModuleType("torch")
        candidate = types.SimpleNamespace(
            install=lambda torch, gems: setattr(gems, "mm", new)
        )
        with mock.patch.dict(sys.modules, {"torch": torch, "flag_gems": fg}):
            with mock.patch.object(
                module.importlib, "import_module", return_value=candidate
            ):
                receipt = module.install(groups=("matrix",))
                self.assertEqual(receipt["status"], "installed")
                self.assertIs(fg._FULL_CONFIG[0][1], new)
                self.assertIs(fg.FULL_CONFIG_BY_FUNC["mm"][0][1], new)
                with self.assertRaises(RuntimeError):
                    module.install(groups=("matrix",))

    def test_failed_partial_install_requires_fresh_process(self):
        module = load_installer()
        fg = types.ModuleType("flag_gems")
        fg.vendor_name = "hygon"
        fg.current_work_registrar = None
        fg._FULL_CONFIG = ()
        fg.FULL_CONFIG_BY_FUNC = {}
        torch = types.ModuleType("torch")

        def fail(torch, gems):
            raise RuntimeError("candidate failure")

        with mock.patch.dict(sys.modules, {"torch": torch, "flag_gems": fg}):
            with mock.patch.object(
                module.importlib,
                "import_module",
                return_value=types.SimpleNamespace(install=fail),
            ):
                with self.assertRaisesRegex(RuntimeError, "candidate failure"):
                    module.install(groups=("matrix",))
                self.assertEqual(module._STATE["status"], "failed")
                with self.assertRaisesRegex(RuntimeError, "fresh process"):
                    module.install(groups=("matrix",))


if __name__ == "__main__":
    unittest.main()
