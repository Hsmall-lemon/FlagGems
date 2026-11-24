# isort: skip_file - preserve the archived provider import order
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import hashlib, importlib.util, sys  # noqa: E401 - retained import order


def install(torch, flag_gems):
    root = Path(__file__).resolve().parent
    for i in range(1, 10):
        p = root / f"f{i:02d}.py"
        name = (
            "_under90_f"
            + str(i)
            + "_"
            + hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        )
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        m.install(torch, flag_gems)


def randperm_debug(*args, **kwargs):
    import flag_gems

    module = sys.modules.get(flag_gems.randperm.__module__)
    hook = getattr(module, "randperm_debug", None)
    if not callable(hook):
        raise NotImplementedError("Active randperm has no same-kernel debug hook")
    return hook(*args, **kwargs)
