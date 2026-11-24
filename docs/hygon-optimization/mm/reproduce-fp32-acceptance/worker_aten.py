"""Source-bound device-time and strict-numerical MM investigation worker."""

import argparse
import csv
import ctypes
import hashlib
import inspect
import json
import os
import traceback
from contextlib import nullcontext
from pathlib import Path

root = Path(__file__).resolve().parent
config_source = Path(
    "/opt/flagos/python/lib/python3.10/site-packages/torch_fl/configs/backends_hygon_boxing.conf"
)
text = config_source.read_text()
assert "\nmm = flaggems\n" in text
private = root / ("boxing-" + str(os.getpid()) + ".conf")
private.write_text(text.replace("\nmm = flaggems\n", "\nmm = boxing\n"))
os.environ["FLAGOS_BACKEND_CONFIG"] = str(private)
# Torch-FL must initialize its device integration before PyTorch is imported.
# isort: off
import torch_fl  # noqa: E402,F401
import torch  # noqa: E402

# isort: on
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--cases", default="cases.json")
ap.add_argument("--precision", action="store_true")
ap.add_argument("--primary-reference", choices=["cpu32", "math64"], default="cpu32")
ap.add_argument("--variants", default="table,candidate,das")
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--rois", type=int, default=8)
ap.add_argument("--seed", type=int, default=20261109)
ap.add_argument("--table-inputs", action="store_true")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--warm-batches", type=int, default=8)
ap.add_argument("--reconstruction-label", default="table_seeded_if_available")
ap.add_argument("--provider-manifest", default="integration-r451-provenance.json")
ap.add_argument("--acceptance-csv", type=Path)
args = ap.parse_args()
acceptance = {}
if args.acceptance_csv:
    with args.acceptance_csv.open(encoding="utf-8-sig", newline="") as file:
        acceptance = {int(row["case_id"]): row for row in csv.DictReader(file)}


def source_fingerprint():
    files = (
        list(root.glob("*.py"))
        + list(root.glob("*configs.json"))
        + list(root.glob("cases*.json"))
    )
    files += [
        p
        for folder in ("table_public", "candidate_public", "pipeline_public")
        for p in (root / folder).glob("*.py")
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


source_at_start = source_fingerprint()
torch.set_num_threads(4)
torch.set_grad_enabled(False)
assert torch.flagos.device_count() == 1
print("READY", args, torch.__version__, triton.__version__, flush=True)


@triton.jit
def queue_gate(P):
    ix = tl.program_id(0) * 64 + tl.arange(0, 64)
    v = tl.load(P + ix)
    for _ in range(1000000):
        v = tl.inline_asm_elementwise(
            "v_add_u32 $0, 1, $1\ns_nop 1",
            "=v,v",
            [v],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    tl.store(P + ix, v)


gate = torch.zeros((8192,), device="flagos", dtype=torch.int32)
queue_gate[(128,)](gate, num_warps=1)
torch.flagos.synchronize()
roctx = ctypes.CDLL("/public/software/compiler/dtk-26.04/lib/libroctx64.so")
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]


def tensors(case, seed):
    metas = case["metadata"][0]
    generator = torch.Generator().manual_seed(seed)
    groups, keys = {}, []
    for index, meta in enumerate(metas):
        key = meta.get(
            "storage_alias", meta.get("alias_group", "distinct-" + str(index))
        )
        size = (
            meta.get("storage_offset", 0)
            + 1
            + sum((n - 1) * s for n, s in zip(meta["shape"], meta["stride"]))
        )
        keys.append(key)
        if key not in groups:
            groups[key] = {
                "size": size,
                "dtype": getattr(torch, meta["dtype"].split(".")[-1]),
            }
        else:
            groups[key]["size"] = max(size, groups[key]["size"])
    for index, group in enumerate(groups.values()):
        rng = (
            generator
            if args.table_inputs
            else torch.Generator().manual_seed(seed + index)
        )
        backing = torch.randn(group["size"], dtype=group["dtype"], generator=rng)
        group["cpu"] = backing
        group["gpu"] = backing.to("flagos")
        group["sha256"] = hashlib.sha256(backing.numpy().tobytes()).hexdigest()
    result = []
    for meta, key in zip(metas, keys):
        group = groups[key]
        geometry = (meta["shape"], meta["stride"], meta.get("storage_offset", 0))
        result.append(
            (
                group["cpu"].as_strided(*geometry),
                group["gpu"].as_strided(*geometry),
                group["sha256"],
            )
        )
    for index, meta in enumerate(metas):
        result[index][1].requires_grad_(meta.get("requires_grad", False))
    return result


def variant_scope(variant):
    if variant == "gems":
        import flag_gems

        return flag_gems.use_gems(include=["mm"])
    return nullcontext()


def make(variant, a, b, case):
    return lambda: torch.ops.aten.mm.default(a, b)


def check(value, ref, atol, rtol):
    delta = (value.double() - ref.double()).abs()
    bad = ~torch.isfinite(delta) | (delta > atol + rtol * ref.double().abs())
    return {
        "correct": not bool(bad.any()),
        "bad_elements": int(bad.sum()),
        "max_abs_error": float(delta.max()),
        "total": ref.numel(),
        "atol": atol,
        "rtol": rtol,
    }


import flag_gems  # noqa: E402

print(
    "ATEN_SOURCE", flag_gems.__file__, inspect.getsourcefile(flag_gems.mm), flush=True
)
expected_source = json.loads((root / args.provider_manifest).read_text())
provider_root = (
    Path(os.environ["FLAGOS_DEV_SOURCE"]) / "src/flag_gems/runtime/backend/_hygon/ops"
)


def provider_fingerprint():
    return {
        name: hashlib.sha256(
            (provider_root / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in expected_source
    }


assert (
    Path(flag_gems.__file__)
    .resolve()
    .is_relative_to(Path(os.environ["FLAGOS_DEV_SOURCE"]).resolve())
), "FlagGems package is outside frozen source"
assert (
    Path(inspect.getsourcefile(flag_gems.mm)).resolve()
    == (provider_root / "mm.py").resolve()
), "Public mm provider is outside frozen Hygon source"
assert (
    provider_fingerprint() == expected_source
), "Integrated provider differs from reviewable worktree"
(root / (args.name + "-provider-source.json")).write_text(
    json.dumps(expected_source, indent=2)
)

records, validations = [], []
cases = json.loads((root / args.cases).read_text())
requested = [int(v) for v in args.rows.split(",")]
cases = [c for c in cases if c["row"] in requested]
assert len(cases) == len(requested)
variants = args.variants.split(",")
for case in cases:
    row = case["row"]
    print("CASE", row, flush=True)
    for si in range(args.seeds):
        seed = args.seed + si * 101
        (ac, a, ah), (bc, b, bh) = tensors(case, seed)
        if args.table_inputs and si == 0 and case.get("table_backing_sha256"):
            expected = list(case.get("table_backing_sha256", {}).values())
            match = expected == [ah, bh]
            print("TABLE_BACKING_MATCH", row, match, [ah, bh], expected, flush=True)
            if not match:
                print("REBUILT_INPUT_DIAGNOSTIC", row, flush=True)
        dtype = ac.dtype
        ref64 = ac.double() @ bc.double()
        ref32 = ac @ bc if args.precision else None
        atol, rtol = (
            (1e-4, 1.3e-6)
            if dtype == torch.float32
            else (1e-10, 1e-10) if dtype == torch.float64 else (1e-5, 0.016)
        )
        if args.acceptance_csv:
            limits = acceptance[row]
            assert dtype == torch.float32 and limits["reference"] == "CPU FP32"
            assert args.seeds == int(limits["seed_count"])
            atol, rtol = float(limits["atol"]), float(limits["rtol"])
        reference = ref32 if dtype == torch.float32 else ref64
        cpu32_vs64 = check(ref32, ref64, atol, rtol) if args.precision else None
        das_reference = make("das", a, b, case)().cpu() if args.precision else None
        torch.flagos.synchronize()
        functions, errors = {}, {}
        for variant in variants:
            try:
                with variant_scope(variant):
                    fn = make(variant, a, b, case)
                    value = fn().cpu()
                    torch.flagos.synchronize()
                    primary_reference = reference
                    err = check(value, primary_reference, atol, rtol)
                    err["primary_reference"] = (
                        "cpu_fp32" if dtype == torch.float32 else "cpu_fp64"
                    )
                    if dtype == torch.float32:
                        err["original_tolerance"] = check(
                            value, reference, 1e-5, 1.3e-6
                        )
                        err["atol_2e5"] = check(value, reference, 2e-5, 1.3e-6)
                    if args.precision:
                        err["against_das"] = check(value, das_reference, atol, rtol)
                        err["against_cpu_fp32"] = check(value, ref32, atol, rtol)
                        err["against_math_fp64"] = check(value, ref64, atol, rtol)
                        err["cpu_fp32_against_fp64"] = cpu32_vs64
                    errors[variant] = err
                    validations.append(
                        {"row": row, "variant": variant, "seed": seed, **err}
                    )
                    print("NUMERIC", row, variant, seed, json.dumps(err), flush=True)
                    if si > 0:
                        continue
                    for _ in range(args.warm_batches):
                        keep = [fn() for _ in range(args.calls)]
                        torch.flagos.synchronize()
                        del keep
                    functions[variant] = fn
            except Exception as exc:
                traceback.print_exc()
                validations.append(
                    {
                        "row": row,
                        "variant": variant,
                        "seed": seed,
                        "correct": False,
                        "error": repr(exc),
                    }
                )
        (root / (args.name + "-validation.json")).write_text(
            json.dumps(validations, indent=2)
        )
        if si > 0:
            continue
        for block in range(args.blocks):
            for roi in range(args.rois):
                order = list(functions)
                if (block + roi) % 2:
                    order.reverse()
                for variant in order:
                    queue_gate[(128,)](gate, num_warps=1)
                    label = (
                        f"fp64audit:row={row}:variant={variant}:block={block}:roi={roi}"
                    )
                    with variant_scope(variant):
                        roctx.roctxRangePushA(label.encode())
                        keep = [functions[variant]() for _ in range(args.calls)]
                        roctx.roctxRangePop()
                    torch.flagos.synchronize()
                    records.append(
                        {
                            "row": row,
                            "variant": variant,
                            "block": block,
                            "roi": roi,
                            "calls": args.calls,
                            "correctness": errors[variant],
                            "seed": seed,
                            "input_backing_sha256": [ah, bh],
                            "reconstruction_label": args.reconstruction_label,
                        }
                    )
                    del keep
        (root / (args.name + "-records.json")).write_text(json.dumps(records, indent=2))
    print("CASE_DONE", row, flush=True)
hashes = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ("table_public", "candidate_public", "pipeline_public")
    for p in (root / folder).glob("*.py")
}
for name in (
    "worker.py",
    "accuracy_swizzle.py",
    "accuracy_chunked.py",
    "accuracy_streamed.py",
    "cases.json",
    "tune-configs.json",
):
    if not (root / name).exists():
        continue
    hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / (args.name + "-source.json")).write_text(json.dumps(hashes, indent=2))
assert provider_fingerprint() == expected_source
assert (
    source_fingerprint() == source_at_start
), "Source snapshot changed during the run."
(root / (args.name + "-source-at-start.json")).write_text(
    json.dumps(source_at_start, indent=2)
)
assert all(
    any(r["row"] == c["row"] and r["variant"] == v for r in records)
    for c in cases
    for v in variants
), "Some variants failed; see validation log"
(root / (args.name + ".done")).write_text("completed\n")
print("DONE", args.name, flush=True)
