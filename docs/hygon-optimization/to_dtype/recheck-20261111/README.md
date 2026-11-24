# Dtype/copy device acceptance, 2026-11-11

All 46 workbook inputs and 13 additional controls pass the current device target: 56 inputs perform device work with DAS/new FG at least 90%; three identity/empty inputs launch no work. The minimum across the complete 59-input scope is 96.57%.

The two requested sheets contain 10 `aten::to.dtype` and 36 related `aten::_to_copy` inputs. Workbook SHA256 and exact metadata matches are retained in the archive.

This update preserves physical output strides, handles FP32-to-BOOL using an explicit nonzero comparison, and extends dense BF16/FP32, INT64/FP32 and BOOL/INT32 conversions. The final launch update uses smaller blocks for BF16-to-FP32 inputs between 262144 and 524288 elements, and packed conversion for aligned FP32-to-BF16 inputs between 32768 and 262144 elements. Large packed/quad routes remain unchanged.

Every original dtype input was independently rechecked on a01r3n13 PCI 0000:77:00.0 and a14r3n03 HCU 0/1, PCI 0000:09:00.0 and 0000:36:00.0: 30 baseline comparisons and 106 passing numerical tests on each card. All six inputs affected by the final launch change were then rechecked on all three cards: 18 comparisons, plus another 106 passing numerical tests per card. The other 40 workbook inputs and 13 controls retain their measured kernel/launch routes; exact source restoration and metadata classification are independently verified.

Timing sums every HIPOPS kernel and HIPCOPY duration per call. Host time and the queue gate are excluded. Each operator uses one GPU; simultaneous tests on other cards are separate runs, never a two-GPU operator timing. Each provider has three A/A blocks of five ROIs. Both providers must pass max/min drift <=5%; SQLite and official JSON agree on every ROI and kernel/copy component. Sampling has a symmetric 1 GiB output budget, capped at 1024 calls per ROI.

Seven paths are retained: native HIP public DAS on a shared-storage metadata view, old public FG, old generic FG directly, routing-only generic FG, published PR #3, the current implementation, and boxing public. Invalid old-public/boxing output strides remain visible and are excluded from valid speedup claims. Native HIP public preserves the required layout. We do not claim to repair boxing or old FG itself.

| Input | Card | DAS/new | Old public/new | Old generic/new | Published PR/new | New A/A |
|---|---|---:|---:|---:|---:|---:|
| 1711 | n03-gpu0 | 103.18% | 1.0404 | 1.0347 | 1.0310 | 0.61% |
| 1711 | n03-gpu1 | 103.17% | 1.0381 | 1.0415 | 1.0456 | 2.48% |
| 1711 | n13 | 104.68% | 1.0263 | 1.0339 | 1.0515 | 1.16% |
| 1713 | n03-gpu0 | 99.34% | layout failure | 0.9903 | 1.0077 | 0.10% |
| 1713 | n03-gpu1 | 100.16% | layout failure | 0.9997 | 0.9977 | 2.02% |
| 1713 | n13 | 99.96% | layout failure | 1.0010 | 0.9963 | 1.25% |
| 1718 | n03-gpu0 | 99.81% | 0.9967 | 0.9963 | 0.9967 | 0.22% |
| 1718 | n03-gpu1 | 100.09% | 0.9948 | 1.0026 | 0.9935 | 1.73% |
| 1718 | n13 | 99.96% | 0.9972 | 1.0013 | 1.0017 | 0.92% |
| 1719 | n03-gpu0 | 100.77% | 1.0076 | 1.3672 | 0.9979 | 1.82% |
| 1719 | n03-gpu1 | 101.89% | 1.0138 | 1.3702 | 1.0118 | 1.62% |
| 1719 | n13 | 100.17% | 1.0117 | 1.3500 | 1.0026 | 1.49% |
| 940018 | n03-gpu0 | 100.52% | 1.0054 | 1.0166 | 1.0058 | 0.56% |
| 940018 | n03-gpu1 | 99.72% | 0.9949 | 0.9993 | 0.9938 | 1.83% |
| 940018 | n13 | 100.10% | 0.9904 | 0.9953 | 0.9932 | 0.45% |
| 940034 | n03-gpu0 | 100.49% | 1.0010 | 1.3907 | 1.0218 | 1.48% |
| 940034 | n03-gpu1 | 100.11% | 1.0100 | 1.3765 | 1.0204 | 1.30% |
| 940034 | n13 | 100.42% | 1.0067 | 1.3757 | 1.0180 | 0.09% |

Ratios below 1 remain visible: reaching the DAS target does not imply that every cast is faster than every old path. In particular, the FP32-to-BF16 37120-element input remains about 0.3%-0.5% slower than old public in this round; no speedup is claimed for it. The earlier 4.17% regression on the 168200-element input is absent in all three final captures, which are about 0.8%-1.4% faster than old public. The INT64 input retains an old-generic and routing-only comparison so routing benefit is not described as a new-kernel-only gain.

These tests reconstruct Excel metadata and additional controls; original model operands were not recovered. Historical reviewer findings, earlier failures, unstable configuration experiments and all earlier archives remain preserved. Configuration experiments with failed A/A checks are not accepted as performance results.

Run the complete CPU-only audit from this checkout:

```bash
python docs/hygon-optimization/to_dtype/recheck-20261111/verify_evidence.py
```

`device-observations.json.gz` retains all 48 full seven-path captures and the native database hashes. `cases.csv` maps every current input to its selected evidence. Native databases are verified and backed up outside Git; profiler scratch and caches stay under node `/tmp` with explicit size limits.
