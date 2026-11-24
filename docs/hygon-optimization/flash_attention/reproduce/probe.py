import argparse
import ctypes
import importlib
import json
import statistics
import traceback
from pathlib import Path

# Torch-FL and the native extension must register CUDA hooks before FlagGems
# probes devices. Keep this bootstrap order explicit; isort must not reorder it.
torch_fl = importlib.import_module("torch_fl")
torch = importlib.import_module("torch")
native_fa = importlib.import_module("flash_attn_2_cuda")
triton = importlib.import_module("triton")
tl = importlib.import_module("triton.language")
flag_gems = importlib.import_module("flag_gems")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["das", "baseline", "tune"], required=True)
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()

torch.set_num_threads(16)
print("READY", args.mode, "devices", torch.flagos.device_count(), flush=True)
print("extension", [k for k in dir(torch_fl) if "C" in k or "view" in k], flush=True)
print("SCHEMA", torch.ops.aten._flash_attention_forward.default._schema, flush=True)


@triton.jit
def gate(P, ITERS: tl.constexpr):
    for i in range(ITERS):
        x = tl.load(P, volatile=True)
        tl.store(P, x)


class Timer:
    def __init__(self):
        torch.flagos.init()
        self.lib = ctypes.CDLL(
            "/public/software/compiler/dtk-26.04/cuda/cuda-12/lib64/libcudart.so"
        )
        for n in ["EventCreate", "EventRecord", "EventElapsedTime", "EventDestroy"]:
            setattr(self.lib, "hip" + n, getattr(self.lib, "cuda" + n))
        self.lib.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.hipEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.stream = ctypes.c_void_p(torch.flagos.current_stream().cuda_stream)
        self.gate = torch.zeros((), device="flagos", dtype=torch.int32)
        gate[(1,)](self.gate, 2000000, num_warps=1)
        self.events = [(self.event(), self.event()) for _ in range(10)]
        self.roctx = ctypes.CDLL(
            "/public/software/compiler/dtk-26.04/lib/libroctx64.so"
        )
        self.roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
        torch.flagos.synchronize()

    def check(self, rc):
        if rc:
            raise RuntimeError(f"HIP {rc}")

    def event(self):
        x = ctypes.c_void_p()
        self.check(self.lib.hipEventCreate(ctypes.byref(x)))
        return x

    def bench(self, fn, label, rounds=5):
        for _ in range(3):
            fn()
        torch.flagos.synchronize()
        rois = []
        for i in range(rounds):
            self.roctx.roctxRangePushA(f"flash:{label}:roi={i}".encode())
            if not args.profile:
                gate[(1,)](self.gate, 2000000, num_warps=1)
            keep = []
            for a, b in self.events:
                self.check(self.lib.hipEventRecord(a, self.stream))
                keep.append(fn())
                self.check(self.lib.hipEventRecord(b, self.stream))
            torch.flagos.synchronize()
            self.roctx.roctxRangePop()
            samples = []
            for a, b in self.events:
                ms = ctypes.c_float()
                self.check(self.lib.hipEventElapsedTime(ctypes.byref(ms), a, b))
                samples.append(ms.value * 1000)
            rois.append(statistics.median(samples))
        return {"us": statistics.median(rois), "rois_us": rois}


def make_inputs():
    torch.manual_seed(20261105)
    # Single shared QKV storage, matching all captured offsets and strides.
    cpu = torch.randn((8, 2049, 3, 16, 128), dtype=torch.bfloat16)
    storage = cpu.to("flagos")
    shape = (8, 16, 2049, 128)
    stride = (12589056, 128, 6144, 1)
    return [storage.as_strided(shape, stride, off) for off in (0, 2048, 4096)], [
        cpu.as_strided(shape, stride, off) for off in (0, 2048, 4096)
    ]


def accuracy(out, lse, cpu):
    q, k, v = [x.float() for x in cpu]
    # Cover all batches and heads, including the 2049th tail query.
    ix = torch.tensor([0, 1, 31, 32, 63, 64, 127, 128, 1023, 1024, 2047, 2048])
    scores = q[:, :, ix] @ k.transpose(-1, -2) * (128**-0.5)
    ref_lse = torch.logsumexp(scores, dim=-1)
    ref = torch.softmax(scores, dim=-1) @ v
    actual = out.cpu().float()[:, :, ix]
    actual_lse = lse.cpu().float()[:, :, ix]
    torch.testing.assert_close(actual, ref, atol=0.002, rtol=0.02)
    torch.testing.assert_close(actual_lse, ref_lse, atol=2e-5, rtol=2e-5)
    return {
        "ok": True,
        "max_abs": (actual - ref).abs().max().item(),
        "lse_max_abs": (actual_lse - ref_lse).abs().max().item(),
    }


if __name__ == "__main__":
    try:
        inputs, cpu = make_inputs()
        timer = Timer()
        scale = 128**-0.5
        if args.mode == "das":
            extension = torch_fl._C
            views = [extension._flagos_to_cuda_view(x) for x in inputs]
            print(
                "VIEWS",
                [(str(x.device), x.stride(), x.storage_offset()) for x in views],
                flush=True,
            )
            native = torch.ops.aten._scaled_dot_product_flash_attention.default
            fn = lambda: native(*views, scale=scale)
            y = fn()
            torch.flagos.synchronize()
            print(
                "OUTPUTS",
                [
                    (
                        (str(x.device), tuple(x.shape), str(x.dtype))
                        if isinstance(x, torch.Tensor)
                        else x
                    )
                    for x in y
                ],
                flush=True,
            )
            out, lse = [extension._cuda_to_flagos_view(x) for x in y[:2]]
        else:
            fn = lambda: flag_gems._scaled_dot_product_flash_attention(
                *inputs, scale=scale
            )
            y = fn()
            out, lse = y[:2]
        torch.flagos.synchronize()
        check = accuracy(out, lse, cpu)
        result = {"mode": args.mode, "accuracy": check, **timer.bench(fn, args.mode)}
        Path(f"{args.mode}.json").write_text(json.dumps(result, indent=2))
        print("RESULT", json.dumps(result), flush=True)
    except Exception:
        traceback.print_exc()
        raise
