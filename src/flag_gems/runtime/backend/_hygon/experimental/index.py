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

"""Process-local CUDA index registration; pinned FG algorithm unchanged.
Keep Torch-FL PrivateUse1 bounds/boxing. The trusted FG config uses vendor
aminmax to avoid the independently demonstrated noncontiguous reduction bug.
No shared installation or Tensor.__getitem__ mutation.
"""

_REGISTRATION = None
_LIBRARY = None


def install(torch, flag_gems):
    global _REGISTRATION, _LIBRARY
    assert _REGISTRATION is None, "index integration must install once"
    from flag_gems.runtime.op_registrar import GeneralOpRegistrar

    _LIBRARY = torch.library.Library("aten", "IMPL")
    flag_gems.only_enable(lib=_LIBRARY, include=["index"], registrar=GeneralOpRegistrar)
    _REGISTRATION = flag_gems.current_work_registrar
    assert _REGISTRATION.reg_key == "CUDA", (
        "Unexpected index dispatch key",
        _REGISTRATION.reg_key,
    )
    assert _REGISTRATION.get_all_keys() == [
        "index.Tensor"
    ], _REGISTRATION.get_all_keys()
    assert _REGISTRATION.get_all_ops() == ["index"], _REGISTRATION.get_all_ops()
    flag_gems._top20_index_registration = dict(
        dispatch_key=_REGISTRATION.reg_key,
        schemas=_REGISTRATION.get_all_keys(),
        provider="flag_gems.index",
        kind="process_local_CUDA_registration_pinned_algorithm_unchanged",
    )
