# Four worksheet reconstructions pass the authorized Release target

SourceFind Release `2.8.4+dtk2610.torch2130.2610211904.g0f1a5a` is the
user-selected comparison version. Physical cards 3 and 4 separately pass
all four cases, five rounds of 1000 calls each. The lowest individual-round
native/FlagGems ratio is 98.38%; host time is excluded.

All timed extension/ATen/FlagGems output and LSE elements pass the unchanged
CPU precision gate and three bitwise stable repeats. Each card also passes
33 repository tests and 34 full-tensor fixtures repeated three times.

The scope is deterministic worksheet reconstruction. The original model
payload and external frozen gate have not been replayed. Additional native
boundary failures are preserved in the report and provide no timing baseline.
See `REPORT.md`, `acceptance-status.json` and `release2610-verification/`.
