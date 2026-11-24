"""Full common precision gate, then worksheet-defined device sampling."""

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import socket
import uuid
from importlib.metadata import version
from pathlib import Path

from accuracy_gate import check_full, cpu_reference
from workbook_gate import TOLERANCES, require_case
from workbook_inputs import MANIFEST, make_case

torch_fl = importlib.import_module("torch_fl")
torch = importlib.import_module("torch")
fa = importlib.import_module("flash_attn_2_cuda")
flag_gems = importlib.import_module("flag_gems")

parser = argparse.ArgumentParser()
parser.add_argument("--calls", type=int, default=1000)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--warmup", type=int, default=100)
args = parser.parse_args()
assert args.calls > 0 and args.rounds > 0 and args.warmup >= 0
torch.set_num_threads(16)
torch.set_grad_enabled(False)
root = Path(flag_gems.__file__).resolve().parents[2]
sources = [
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.py",
    "src/flag_gems/runtime/backend/_hygon/attention_state_hip.cpp",
    "src/flag_gems/runtime/backend/_hygon/ops/attention_state.py",
    "src/flag_gems/runtime/backend/_hygon/ops/_scaled_dot_product_flash_attention.py",
    "tests/test_flash_attention_hygon_state.py",
]
binding = torch.library.Library("aten", "IMPL", "PrivateUse1")
binding.impl(
    "_scaled_dot_product_flash_attention", flag_gems._scaled_dot_product_flash_attention
)
roctx = ctypes.CDLL("/public/software/compiler/dtk-26.04/lib/libroctx64.so")
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
receipt = {
    "schema": 5,
    "run_id": uuid.uuid4().hex,
    "workbook_sha256": MANIFEST["workbook_sha256"],
    "numerical_tolerances": TOLERANCES,
    "source_sha256": {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
    },
    "device": {
        "node": socket.gethostname(),
        "physical_card": os.environ.get("HIP_VISIBLE_DEVICES"),
    },
    "sampling": vars(args),
    "external_frozen_gate_verified": False,
    "original_model_acceptance_verified": False,
    "cases": {},
}
output_path = Path("workbook-gate.json")


def save():
    output_path.write_text(json.dumps(receipt, indent=2) + "\n")


for case in MANIFEST["cases"]:
    cpu, dev, kwargs, identity = make_case(case, torch)
    reference = cpu_reference(cpu)
    views = [torch_fl._C._flagos_to_cuda_view(x) for x in dev]
    bshd = [x.transpose(1, 2) for x in views]
    functions = {
        "wheel": lambda bshd=bshd, kwargs=kwargs: fa.fwd(
            *bshd,
            None,
            None,
            0.0,
            kwargs["scale"],
            False,
            -1,
            -1,
            0.0,
            False,
            None,
            False,
            None,
            0.0,
        ),
        "aten": lambda views=views, kwargs=kwargs: torch.ops.aten._scaled_dot_product_flash_attention.default(
            *views, **kwargs
        ),
        "repo": lambda dev=dev, kwargs=kwargs: torch.ops.aten._scaled_dot_product_flash_attention.default(
            *dev, **kwargs
        ),
    }
    record = {
        "case": case,
        "input_identity": identity,
        "full_accuracy": {},
        "timing_started": False,
    }
    receipt["cases"][case["id"]] = record
    for arm, fn in functions.items():
        y = fn()
        torch.flagos.synchronize()
        if arm == "wheel":
            out = torch_fl._C._cuda_to_flagos_view(y[0]).transpose(1, 2)
            lse = torch_fl._C._cuda_to_flagos_view(y[5])
        elif arm == "aten":
            out, lse = [torch_fl._C._cuda_to_flagos_view(x) for x in y[:2]]
        else:
            out, lse = y[:2]
        record["full_accuracy"][arm] = check_full(out, lse, reference)
        save()
        del y, out, lse
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "flash_att" in line and line.split()[-1].startswith("/")
        }
    )
    receipt["native"] = {
        "version": version("flash-attn"),
        "module": fa.__file__,
        "loaded_libraries_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libraries
        },
    }
    require_case(receipt, case)
    record["timing_started"] = True
    save()
    del reference
    for round_index in range(args.rounds):
        order = (
            ("wheel", "aten", "repo")
            if round_index % 2 == 0
            else ("repo", "aten", "wheel")
        )
        for arm in order:
            fn = functions[arm]
            for _ in range(args.warmup):
                fn()
            torch.flagos.synchronize()
            label = (
                f"workbook:{receipt['run_id']}:{case['id']}:{arm}:round={round_index}"
            )
            roctx.roctxRangePushA(label.encode())
            keep = [None, None]
            for index in range(args.calls):
                keep[index % 2] = fn()
            torch.flagos.synchronize()
            roctx.roctxRangePop()
            del keep
            print("PROFILE_ROUND", case["id"], arm, round_index, flush=True)
    record["timing_completed"] = True
    save()
    del cpu, dev, views, bshd, functions, fn
print("WORKBOOK_PROFILE_COMPLETE", flush=True)
