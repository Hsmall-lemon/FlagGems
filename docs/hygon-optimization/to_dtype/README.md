> Acceptance withdrawn (2026-11-09). PR remains draft; see [current review status](REPORT.md). The following is historical documentation.

# Hygon dtype conversion

This change targets the ten exact `aten::to.dtype` inputs assigned to 万 in the spreadsheet. See [REPORT.md](REPORT.md) and [results.csv](results.csv).
The implementation is selected by dtype and tensor layout; it never inspects tensor values or case IDs.

In the supplied DEV image, import torch_fl before FlagGems and put the checkout's src directory first on PYTHONPATH. Use the normal public context:

```python
import torch_fl
import torch
import flag_gems

with flag_gems.use_gems(include=["to_copy"]):
    y = x.to(torch.bfloat16)
```

No shared image or Torch-FL source changes are required. The PrivateUse1 registration applies only while FlagGems is enabled. Other dtype conversions and unsupported layouts retain the generic FlagGems implementation. `to.dtype` with identical dtypes retains PyTorch's no-copy behavior; the low-level `to_copy` API still always returns a new tensor.

The measurement scripts, exact native traces, input reconstruction and launch logs remain under `/public/home/wzx226226/算子优化/`. Set HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES to 0 or 1, unset ROCR_VISIBLE_DEVICES, and use your writable FLAGOS_DEV_CACHE. Measurement records distinguish the public C++ same-operator reference from the paired boxing cast reference. Both exclude host overhead.
