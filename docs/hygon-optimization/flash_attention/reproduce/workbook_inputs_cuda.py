"""Deterministic inputs retaining each workbook row's dtype, strides and offsets."""

import hashlib
import json
from pathlib import Path

MANIFEST = json.loads(
    Path(__file__).with_name("workbook_cases.json").read_text(encoding="utf8")
)


def make_case(case, torch, seed=None, fp32_generation=False):
    tensors, kwargs = case["specification"]
    seed = case["seed"] if seed is None else seed
    torch.manual_seed(seed)
    dtype = torch.float32 if fp32_generation else torch.bfloat16
    if case["storage"] == "shared_qkv_backing_from_user_review":
        b, h, n, d = tensors[0]["shape"]
        backings = [torch.randn((b, n, 3, h, d), dtype=dtype).to(torch.bfloat16)]
        indices = [0, 0, 0]
    else:
        backings = []
        indices = [0, 1, 2]
        for spec in tensors:
            size = (
                spec["storage_offset"]
                + 1
                + sum(
                    (length - 1) * stride
                    for length, stride in zip(spec["shape"], spec["stride"])
                )
            )
            backings.append(torch.randn((size,), dtype=dtype).to(torch.bfloat16))
    stores = [x.to("cuda") for x in backings]
    cpu = [
        backings[index].as_strided(
            spec["shape"], spec["stride"], spec["storage_offset"]
        )
        for index, spec in zip(indices, tensors)
    ]
    dev = [
        stores[index].as_strided(spec["shape"], spec["stride"], spec["storage_offset"])
        for index, spec in zip(indices, tensors)
    ]
    h = hashlib.sha256()
    for backing in backings:
        h.update(backing.view(torch.uint8).numpy().tobytes())
    identity = {
        "seed": seed,
        "fp32_generation": fp32_generation,
        "backing_sha256": h.hexdigest(),
        "backing_elements": [x.numel() for x in backings],
        "backing_bytes": sum(x.numel() * x.element_size() for x in backings),
        "storage_policy": case["storage"],
    }
    return cpu, dev, kwargs, identity
