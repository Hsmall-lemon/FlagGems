# Current submission review: mm

Overall submission remains blocked by the original FP32 and FP64 review inputs
(spreadsheet MM rows 2 and 4) not being matched. No further commit, push or PR
update is authorized by this evidence while the overall gate fails.

The user additionally requires the tables "大于5%的模型工况" and "规模扩展工况".
Their file paths and coverage are still being reconciled. The known records below
do not establish complete coverage of those tables. Submission remains blocked
until that reconciliation is complete as well.

This package covers **179 records**, with **173 passes**,
**4 no-device-work records**, and
**2 unresolved original reports** for this operator. Passing device
records have DAS/new >=90% and both A/A guards <=5%.

`cases.csv` maps every record to its input, device duration, actual kernels,
physical card, baseline, and historical results. `status.json` preserves the
scope, acceptance decisions and unresolved reports. `device-observations.json.gz`
contains compact raw ROI observations from all relevant current and historical
datasets, including failed baseline/stability runs; this is not a profiler API
trace. Runs and protocols remain distinct. It is not one uniform final-source run.

Validate the data and recorded submission hold without torch or a GPU:

```sh
python verify_evidence.py
python verify_evidence.py --require-submission-pass  # exits 2 while blocked
```

All 27 additional spreadsheet MM types were independently measured through
public dispatch on physical GPUs 0 and 1, one GPU per process. The final MM
source passed 97 focused tests on each card with no skips. Row 45's initial
DAS A/A failures are retained alongside the fixed two-provider warmup recheck;
the fastest historical DAS block median divided by the slowest FG block median
also exceeds 90%. Four added exact FP64 low-bit inputs have separate performance
and exact numerical checks. Existing 97 MM launch selections and all device
kernel sources remain unchanged; the local correction adds 22 exact keys.

The dtype comparison uses native HIP `Tensor.to` on a CUDA metadata view that
shares the FG tensor's physical storage, pointer, dtype, sizes, strides, offset
and device index. Native and FG device events share the gate's queue. The old
FlagOS boxing baseline with incorrect output strides and the explicit-layout
different-API `copy_` references remain separate. This is not a claim that the
old boxing route is fixed, that all casts use a new kernel, or that new casts
are uniformly faster than old public FG. This verification round makes no dtype
kernel or routing changes. The optional installer's imports are regrouped to
satisfy isort; its inventory and installation tests pass.
