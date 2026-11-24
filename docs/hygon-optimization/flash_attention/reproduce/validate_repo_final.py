import json
from pathlib import Path

from probe import accuracy, flag_gems, make_inputs, torch, torch_fl

from flag_gems.runtime.backend._hygon.attention_state_hip import _load_state_kernel

assert _load_state_kernel() is not None


def run(q, k, v):
    return flag_gems._scaled_dot_product_flash_attention(q, k, v, scale=128**-0.5)


inputs, cpu = make_inputs()
candidate = lambda: run(*inputs)
repo = lambda: flag_gems._scaled_dot_product_flash_attention(*inputs, scale=128**-0.5)
views = [torch_fl._C._flagos_to_cuda_view(x) for x in inputs]
native = lambda: torch.ops.aten._scaled_dot_product_flash_attention.default(
    *views, scale=128**-0.5
)
y = candidate()
torch.flagos.synchronize()
result = {"sample": accuracy(*y[:2], cpu)}
q, k, v = [x.float() for x in cpu]
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=128**-0.5)
a = y[0].cpu().float()
torch.testing.assert_close(a, ref, atol=0.002, rtol=0.02)
result["full_output"] = {
    "elements": a.numel(),
    "max_abs": (a - ref).abs().max().item(),
    "rmse": (a - ref).square().mean().sqrt().item(),
}
rl = torch.empty(q.shape[:3], dtype=torch.float32)
for st in range(0, 2049, 128):
    scores = q[:, :, st : st + 128] @ k.transpose(-1, -2) * (128**-0.5)
    rl[:, :, st : st + 128] = torch.logsumexp(scores, -1)
al = y[1].cpu()
torch.testing.assert_close(al, rl, atol=2e-5, rtol=2e-5)
result["full_lse"] = {"elements": al.numel(), "max_abs": (al - rl).abs().max().item()}
del q, k, v, ref, a, rl, al, scores
Path("repo_final_full_accuracy.json").write_text(json.dumps(result, indent=2))
result["guard_checks"] = {}
for label in ["mixed_large", "zero_qk"]:
    base = cpu[0]._base.clone()
    if label == "mixed_large":
        base[0].mul_(8)
    else:
        base[:, :, 0].zero_()
        base[:, :, 1].zero_()
    c = [
        base.as_strided((8, 16, 2049, 128), (12589056, 128, 6144, 1), o)
        for o in (0, 2048, 4096)
    ]
    dev = base.to("flagos")
    x = [dev.as_strided(t.shape, t.stride(), t.storage_offset()) for t in c]
    yy = run(*x)
    torch.flagos.synchronize()
    result["guard_checks"][label] = accuracy(*yy[:2], c)
print("CANDIDATE_FULL_ACCURACY", json.dumps(result), flush=True)

Path("repo_final_validation.json").write_text(json.dumps(result, indent=2))
