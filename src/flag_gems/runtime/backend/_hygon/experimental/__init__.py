# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hygon integration of previously qualified experiment sources.

Call install once, before dispatcher registration or model execution. No patch
is applied by importing this module. See docs/hygon-optimization/README.md.
"""

import hashlib
import importlib
import json
import threading
from pathlib import Path

_GROUP_MODULES = {
    "sort": ("sort",),
    "under90": tuple(f"under90.f{i:02d}" for i in range(1, 10)),
    "matrix": ("matrix",),
    "index": ("index",),
}
_LOCK = threading.Lock()
_STATE = None


def _check_request(groups, serialized_eager, flag_gems, torch):
    if isinstance(groups, str):
        raise ValueError("groups must be a sequence, for example ('matrix',)")
    groups = tuple(groups)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("choose nonempty, unique groups")
    unknown = set(groups) - set(_GROUP_MODULES)
    if unknown:
        raise ValueError(f"unknown Hygon groups: {sorted(unknown)}")
    if getattr(flag_gems, "vendor_name", None) != "hygon":
        raise RuntimeError("these candidates require the Hygon backend")
    if getattr(flag_gems, "current_work_registrar", None) is not None:
        raise RuntimeError("install before flag_gems.enable/only_enable/use_gems")
    if "under90" in groups and serialized_eager is not True:
        raise ValueError("randperm requires serialized_eager=True; no graph capture")
    if set(groups) & {"under90", "index"}:
        if torch._C._get_privateuse1_backend_name() != "flagos":
            raise RuntimeError(
                "under90/index require the qualified Torch-FL flagos route"
            )
    if not hasattr(flag_gems, "_FULL_CONFIG") or not hasattr(
        flag_gems, "FULL_CONFIG_BY_FUNC"
    ):
        raise RuntimeError("unsupported FlagGems registrar interface")
    # The original experiment order is significant when groups are combined.
    return tuple(group for group in _GROUP_MODULES if group in groups)


def _refresh_entries(entries, replacements):
    updated = [
        (
            (item[0], replacements.get(id(item[1]), item[1]), *item[2:])
            if len(item) >= 2
            else item
        )
        for item in entries
    ]
    return tuple(updated) if isinstance(entries, tuple) else updated


def _verify_sources():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if (
            root not in path.parents
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Hygon candidate integrity mismatch: {relative}")


def install(*, groups, serialized_eager=False):
    """Patch only this fresh process; return a source-bound installation receipt.

    All requested modules are imported and source-checked before installing.
    A failed installation poisons this process: restart it instead of retrying.
    This entrypoint does not modify configuration files or shared environments.
    """
    global _STATE
    import torch

    import flag_gems

    with _LOCK:
        if _STATE is not None:
            raise RuntimeError(
                "Hygon installation already attempted; use a fresh process"
            )
        ordered = _check_request(groups, serialized_eager, flag_gems, torch)
        _verify_sources()
        modules = [
            importlib.import_module(f"{__name__}.{name}")
            for group in ordered
            for name in _GROUP_MODULES[group]
        ]
        originals = {
            name: value for name, value in vars(flag_gems).items() if callable(value)
        }
        _STATE = {"status": "installing", "groups": ordered}
        try:
            for module in modules:
                module.install(torch, flag_gems)
            replacements = {}
            for name, old in originals.items():
                new = getattr(flag_gems, name, old)
                if new is not old:
                    previous = replacements.get(id(old), new)
                    if previous is not new:
                        raise RuntimeError(
                            f"ambiguous registrar alias replacement: {name}"
                        )
                    replacements[id(old)] = new
            # Sidecars from different dates patch different registrar views.
            # Refresh both views by identity; preserve each schema's metadata.
            flag_gems._FULL_CONFIG = _refresh_entries(
                flag_gems._FULL_CONFIG, replacements
            )
            flag_gems.FULL_CONFIG_BY_FUNC = {
                name: _refresh_entries(entries, replacements)
                for name, entries in flag_gems.FULL_CONFIG_BY_FUNC.items()
            }
            _STATE = {
                "status": "installed",
                "groups": ordered,
                "changed_roots": len(replacements),
            }
        except BaseException:
            _STATE = {"status": "failed", "groups": ordered}
            raise
        return dict(_STATE)
