"""Independent native/candidate accuracy and repeatability on the target layout."""

import hashlib
import importlib
import json
import os
import socket
from pathlib import Path

from accuracy_gate import CONTRACT, check_full, cpu_reference

torch_fl = importlib.import_module("torch_fl")
torch = importlib.import_module("torch")
fa = importlib.import_module("flash_attn_2_cuda")
flag_gems = importlib.import_module("flag_gems")

torch.set_num_threads(16)
result = {
    "card": os.environ["HIP_VISIBLE_DEVICES"],
    "node": socket.gethostname(),
    "module": fa.__file__,
    "cases": [],
}
assert result["card"] in ("3", "4")
for seed in [20261105, 0, 42]:
    torch.manual_seed(seed)
    backing = torch.randn((8, 2049, 3, 16, 128), dtype=torch.bfloat16)
    store = backing.to("flagos")
    cpu = [
        backing.as_strided(CONTRACT["output_shape"], CONTRACT["stride"], o)
        for o in CONTRACT["offsets"]
    ]
    dev = [
        store.as_strided(CONTRACT["output_shape"], CONTRACT["stride"], o)
        for o in CONTRACT["offsets"]
    ]
    views = [torch_fl._C._flagos_to_cuda_view(x) for x in dev]
    wv = [x.transpose(1, 2) for x in views]
    ref = cpu_reference(cpu)
    case = {
        "seed": seed,
        "input_sha256": hashlib.sha256(
            backing.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "arms": {},
    }
    for name, fn in [
        (
            "native",
            lambda wv=wv: fa.fwd(
                *wv,
                None,
                None,
                0.0,
                CONTRACT["scale"],
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
        ),
        (
            "repo",
            lambda dev=dev: flag_gems._scaled_dot_product_flash_attention(
                *dev, scale=CONTRACT["scale"]
            ),
        ),
    ]:
        runs = []
        first = None
        for repeat in range(3):
            y = fn()
            torch.flagos.synchronize()
            if name == "native":
                out = torch_fl._C._cuda_to_flagos_view(y[0]).transpose(1, 2)
                lse = torch_fl._C._cuda_to_flagos_view(y[5])
            else:
                out, lse = y[:2]
            check = check_full(out, lse, ref)
            assert all(
                v["failed"] == 0 and v["nonfinite"] == 0 for v in check.values()
            ), (seed, name, check)
            actual = (out.cpu(), lse.cpu())
            if first is None:
                first = actual
            equal = all(torch.equal(a, b) for a, b in zip(actual, first))
            assert equal, (seed, name, "not deterministic")
            runs.append({"full_accuracy": check, "bitwise_equal_to_first": equal})
            del y, out, lse, actual
        case["arms"][name] = runs
    result["cases"].append(case)
    Path("stability.json").write_text(json.dumps(result, indent=2))
    print("CASE_PASS", seed, flush=True)
    del store, backing, cpu, dev, views, wv, ref, first
print("COMPLETE", flush=True)
