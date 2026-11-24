# Worksheet scope and acceptance

The requested operator is `aten::_scaled_dot_product_flash_attention` only.
The workbook is `待优化设备耗时的算子1.xlsx`, SHA256
`d2596052c66d7145af47b35fda45198d1f213b69825aeb91d10112dfd8871597`.
Scanning both requested sheets finds four rows:

| Sheet | Excel row | Case | Q / K / V shape | Dtype | Stride |
| --- | ---: | --- | --- | --- | --- |
| 大于5%的模型工况 | 645 | model_state | `[8,16,2049,128]` | BF16 | `[12589056,128,6144,1]` |
| 规模扩展工况 | 2 | expanded_small | `[2,16,3,128]` | BF16 | `[18432,128,6144,1]` |
| 规模扩展工况 | 3 | expanded_medium | `[2,16,23,128]` | BF16 | `[141312,128,6144,1]` |
| 规模扩展工况 | 4 | expanded_large | `[2,16,362,128]` | BF16 | `[2224128,128,6144,1]` |

Every row has Q/K/V storage offsets `0/2048/4096`, scale
`0.08838834764831843`, no dropout and no causal masking. Full metadata and
expansion seeds are retained in `reproduce/workbook_cases.json`.

The workbook's worksheet dimension metadata incorrectly reports `A1`.
Read-only openpyxl readers must call `reset_dimensions()` before scanning;
the actual requested sheets have 898 and 953 rows respectively.

## Input provenance and limits

The model fixture uses a shared QKV backing based on the user's previous
review description. The workbook model note says the original alias/backing
contract and original operand payload are unavailable. Its seed is an
explicit reconstruction seed, not a recovered original-model seed.

The three expansion fixtures use independent minimum-size backing stores.
Their total storage sizes match the workbook exactly: 208896, 1683456,
and 26677248 bytes. They retain the worksheet seeds. CPU direct BF16 and
FP32-to-BF16 input generation are tested separately.

These are deterministic metadata reconstructions. They do not establish
that the missing original model payload or external frozen accuracy gate
has been replayed. Every new result retains these limitations explicitly.

## Common numerical and device performance contract

All output and LSE elements must pass the same reference comparison for
FlagGems, the native wheel entry, and native ATen. Output tolerances are
`atol=0.002, rtol=0.02`; LSE tolerances are `atol=rtol=2e-5`.
Nonfinite results, missing elements and shape mismatches fail acceptance.
An unqualified native result cannot supply the performance denominator.

The user-selected current baseline is SourceFind Release
`2.8.4+dtk2610.torch2130.2610211904.g0f1a5a`, package ID 13419. It passes
the common full-output/LSE gate for all four timed fixtures on both cards.
The default DEV image retains DAS185; its prior LSE failures are not repaired
or used as a performance denominator. The unauthorized DTK2604 2.8.4
experiment is historical and supplies no current denominator.

Each physical card is evaluated independently and sequentially on a01r4n15.
Final sampling uses 100 warmups and five rounds of 1000 calls per arm.
Device time is the sum of all GPU kernel durations captured by torch.profiler;
helper and recomputation kernels are included, and host overhead is excluded.
Every individual-round ratio, ratio of mean durations, and ratio of median
durations exceeds 0.9 using the faster qualified native entry. Current
receipts and source identities are in `release2610-verification/`.

The four documented reconstructions meet this contract. Missing original
model payload/external frozen-gate replay remains explicitly unverified.
