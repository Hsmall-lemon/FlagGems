"""Profile public FlagGems dtype casts and the DAS boxing reference."""

import argparse
import ctypes
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

from torch_fl import _C  # isort: split

import torch

import flag_gems

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--physical-card", type=int, required=True)
opts = ap.parse_args()
model = opts.name == "model"
calls, warmups, ranges = (20, 20, 25) if model else (1000, 100, 5)
root = Path(__file__).resolve().parent

torch.set_num_threads(4)
hip = ctypes.CDLL("libamdhip64.so")

source = Path(flag_gems.__file__).parents[2]
hygon = source / "src/flag_gems/runtime/backend/_hygon"
annotation = ctypes.CDLL(str(Path(os.environ["DTK_ROOT"]) / "lib/libroctx64.so"))
annotation.roctxRangePushA.argtypes = [ctypes.c_char_p]
hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
bus = ctypes.create_string_buffer(64)
assert hip.hipDeviceGetPCIBusId(bus, 64, 0) == 0
fingerprint = dict(
    node=os.uname().nodename,
    job=os.environ.get("SLURM_JOB_ID"),
    logical_device=0,
    visible_device_count=torch.flagos.device_count(),
    physical_oam=opts.physical_card,
    pci_bus=bus.value.decode(),
    source=str(source),
    source_files={
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            source / "src/flag_gems/__init__.py",
            hygon / "dtype_cast_hip.py",
            hygon / "dtype_cast_hip.cpp",
            *sorted((hygon / "ops").glob("to*.py")),
        ]
    },
    profiler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    baseline="Torch-FL boxing aten::_to_copy on the same flagos input",
    metric="all HIPOPS + HIPCOPY DurationNs per call; host and queue gate excluded",
    output_sampling_budget_bytes=1073741824,
    max_calls_per_roi=1024,
    queue_gate_ctas=0,
)
assert torch.flagos.device_count() == 1
assert fingerprint["pci_bus"], fingerprint
(root / (opts.name + "-fingerprint.json")).write_text(
    json.dumps(fingerprint, indent=2) + "\n"
)

ids = set(map(int, opts.cases.split(",")))
cases = [c for c in json.loads((root / "cases.json").read_text()) if c["row"] in ids]
assert {c["row"] for c in cases} == ids
with (root / (opts.name + "-records.jsonl")).open("w", buffering=1) as out:
    for case in cases:
        ci = case["row"]
        meta, target = case["metadata"][0]
        dtype = getattr(torch, meta["dtype"].split(".")[-1])
        target = getattr(torch, target.split(".")[-1])
        shape, stride, offset = (
            meta["shape"],
            meta["stride"],
            meta.get("storage_offset", 0),
        )
        size = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        seed = case["seed"]
        gen = torch.Generator().manual_seed(seed)
        if dtype == torch.bool:
            storage = torch.randint(0, 2, (size,), generator=gen).bool()
        elif dtype == torch.int64:
            storage = torch.randint(
                -(2**30), 2**30, (size,), generator=gen, dtype=dtype
            )
        else:
            storage = torch.randn(size, dtype=dtype, generator=gen)
        cpu = storage.as_strided(shape, stride, offset)
        x = storage.to("flagos").as_strided(shape, stride, offset)
        native = _C._flagos_to_cuda_view(x)
        assert (
            native.data_ptr() == x.data_ptr()
            and native.stride() == x.stride()
            and native.storage_offset() == offset
            and native.shape == x.shape
            and native.dtype == x.dtype
        )
        ref = cpu.to(target)
        identity = dtype == target
        input_hash = hashlib.sha256(
            storage.view(torch.uint8).numpy().tobytes()
        ).hexdigest()

        def correct(y, inp):
            actual = y.cpu()
            mask = (
                ~torch.isnan(ref)
                if target.is_floating_point
                else torch.ones_like(ref, dtype=torch.bool)
            )
            value = torch.equal(actual[mask], ref[mask])
            if target.is_floating_point:
                value = value and torch.equal(torch.isnan(actual), torch.isnan(ref))
            if target == torch.bfloat16:
                value = value and torch.equal(
                    actual.contiguous().view(torch.int16)[mask],
                    ref.contiguous().view(torch.int16)[mask],
                )
            if target == torch.float32:
                value = value and torch.equal(
                    actual.contiguous().view(torch.int32)[mask],
                    ref.contiguous().view(torch.int32)[mask],
                )
            return dict(
                value_ok=bool(value),
                dtype_ok=y.dtype == target,
                stride_ok=y.stride() == ref.stride(),
                device_ok=y.device == inp.device,
                alias_ok=(
                    (y is inp)
                    if identity
                    else y is not inp and y.data_ptr() != inp.data_ptr()
                ),
                output_stride=list(y.stride()),
                output_device=str(y.device),
            )

        providers = ["das", "new_public"]
        with torch.no_grad():
            for ar in range(3):
                order = providers[ar:] + providers[:ar]
                for provider in order:
                    context = (
                        flag_gems.use_gems(include=["to_copy"])
                        if provider == "new_public"
                        else nullcontext()
                    )
                    inp = x
                    with context:
                        if provider == "das" and not identity:
                            fn = lambda: torch.ops.aten._to_copy.default(
                                x, dtype=target
                            )
                        else:
                            fn = lambda: torch.ops.aten.to.dtype(inp, target)
                        trial = fn()
                        torch.flagos.synchronize()
                        checked = correct(trial, inp)
                        if provider != "das":
                            assert all(
                                checked[k]
                                for k in [
                                    "value_ok",
                                    "dtype_ok",
                                    "stride_ok",
                                    "device_ok",
                                    "alias_ok",
                                ]
                            ), (ci, provider, checked)
                        for _ in range(warmups):
                            fn()
                        torch.flagos.synchronize()
                        for roi in range(ranges):
                            annotation.roctxRangePushA(
                                f"correction:case={ci}:provider={provider}:aa={ar}:roi={roi}".encode()
                            )
                            if model:
                                keep = [fn() for _ in range(calls)]
                                last = keep[-1]
                            else:
                                for _ in range(calls):
                                    last = fn()
                            torch.flagos.synchronize()
                            annotation.roctxRangePop()
                            assert correct(last, inp) == checked
                            out.write(
                                json.dumps(
                                    dict(
                                        case=ci,
                                        provider=provider,
                                        aa=ar,
                                        roi=roi,
                                        calls=calls,
                                        seed=seed,
                                        correctness=checked,
                                        shape=shape,
                                        stride=stride,
                                        offset=offset,
                                        input_dtype=str(dtype),
                                        target_dtype=str(target),
                                        memory_format="None",
                                        noop=identity,
                                        input_storage_sha256=input_hash,
                                        input_pointer_shared=True,
                                        source_snapshot=str(source),
                                        physical_oam=opts.physical_card,
                                        pci_bus=fingerprint["pci_bus"],
                                    )
                                )
                                + "\n"
                            )
                            del last
                            if model:
                                del keep
                    print("DTYPE_COMPARE_BLOCK", ci, provider, ar, checked, flush=True)
print("DTYPE_COMPARE_COMPLETE", flush=True)
