# Extended dtype/copy device verification, 2026-11-10

The [2026-11-11 launch update and complete current verifier](../recheck-20261111/README.md)
supersede the launch choices for six inputs. This archive remains unchanged;
the verifier delegates to the current evidence when the launch overlay is present.

All 59 reconstructed input/control records pass the current checks: 56 launch
device work with DAS/new FG at least 90%, and three identity/empty records
launch no device work. The latest minimum ratio is 96.57%. This verifies the
metadata and control inputs described here, not replay of original model
operands. The combined submission is still held by unresolved MM review
and performance work; PR #3 remains Draft.

The new workbook contributes 10 `aten::to.dtype` and 36 related `aten::_to_copy`
rows across both requested tabs. Thirteen additional controls cover preserved
and contiguous memory formats, transposes, offsets, storage holes, empty and
scalar inputs, and special FP32/BF16 values. The workbook SHA256 is retained
inside the observation archive. The original workbook is unchanged.

The candidate handles dense physical order while preserving output strides.
FP32-to-BOOL explicitly tests nonzero values. Large aligned FP32-to-BF16 casts
use integer round-to-nearest-even conversion and four-value stores. All 106
numerical tests pass. The quad revision affects only cases 940019 and 940020;
the other route and kernel ASTs are preserved, with baseline source retained
for independent verification.

Measurements sum every device kernel and copy duration per call; host and
queue-gate time are excluded. Each process sees one physical GPU. Every
capture has three A/A blocks of five ROIs and requires both providers' A/A
max/min drift to stay at or below 5%. Native HIP `Tensor.to` or `aten._to_copy`
uses a CUDA metadata view sharing the exact input storage, pointer, dtype,
shape, strides, offset and device index with the FlagOS input. Native SQLite
and official JSON exports agree for every measured sample and component.

Earlier 256 MiB output batches exposed bimodal native baseline timings.
Cases 1711, 940014 and 940041 were checked twice with the same symmetric
1 GiB output budget, capped at 1024 calls per ROI, and both runs passed.
The stronger 1024-CTA gate is outside the ROI; GPU clocks were not modified.
All prior failures, unstable captures, source fingerprints and per-ROI data
are retained in the archive. Ratios from different sampling batches should
not be treated as an improvement in the unchanged kernel. Earlier quad
measurements were 96.51% and 95.43%; their independent confirmation also passes.

No claim is made that every new cast is faster than old public FG. Routing
benefits and new kernel benefits remain distinct, and no-work records have
no speedup ratio.

Run the CPU-only audit from the checkout:

```bash
python docs/hygon-optimization/to_dtype/extended-20261110/verify_evidence.py
```

`cases.csv` lists the selected current observation for every input. The gzip
archive retains every capture, including failures and native database hashes.
Native databases remain outside Git in bounded node-local storage.
