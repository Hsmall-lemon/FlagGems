#!/usr/bin/env bash
set -euo pipefail

base=${MM_RUN_ROOT:-${TMPDIR:-/tmp}/flaggems-mm-repro}
upload=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$base"
run=$(mktemp -d "$base/integrated-fast.XXXXXX")
echo "$run"
source=$(realpath -- "${1:?Pass the FlagGems checkout}")
python3 - "$source" "$run" "$upload" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

source, run, upload = map(pathlib.Path, sys.argv[1:])
ops = source / "src/flag_gems/runtime/backend/_hygon/ops"
manifest = (upload / "provider.json").read_text()
for name, digest in json.loads(manifest).items():
    payload = (ops / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(payload).hexdigest() == digest, name
(run / "provider.json").write_text(manifest)
with (upload / "fp32-acceptance-frozen.csv").open(encoding="utf-8-sig", newline="") as file:
    cases = list(csv.DictReader(file))
rows = [int(case["case_id"]) for case in cases]
assert rows and len(set(rows)) == len(rows)
metadata = json.loads((upload / "cases.json").read_text())
assert sorted(rows) == sorted(case["row"] for case in metadata)
assert {float(case["rtol"]) for case in cases} == {1.3e-6}
for card in range(8):
    assigned = rows[card::8]
    batches = [assigned[i:i + 5] for i in range(0, len(assigned), 5)]
    (run / f"gpu{card}.txt").write_text("".join(",".join(map(str, b)) + "\n" for b in batches))
PY

export DEV=${DEV:-/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization}
export FLAGOS_DEV_CACHE=${FLAGOS_DEV_CACHE:-$base/cache}
dtk=/public/software/compiler/dtk-26.04
run_card() (
    card=$1
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export ROCR_VISIBLE_DEVICES=$card OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    ulimit -c 0
    ulimit -f 614400
    batch=0
    status=0
    while IFS= read -r rows; do
        d=$run/gpu$card-batch$batch
        mkdir -p "$d/profiles"
        cp "$upload"/{worker_aten.py,analyze.py,cases.json,fp32-acceptance-frozen.csv} "$d/"
        cp "$run/provider.json" "$d/"
        printf 'START %s %s %s\n' "$card" "$batch" "$rows"
        if (
            cd "$d"
            bash "$DEV/bin/run" --source "$source" -- "$dtk/bin/hipprof" \
                --hip-trace --hiptx-trace --output-type 0 -o "$d/profiles/full" \
                python -u worker_aten.py --rows "$rows" --name full --cases cases.json \
                --precision --table-inputs --seeds 5 --blocks 3 --rois 4 --warm-batches 2 \
                --calls 5 --variants das,gems --provider-manifest provider.json \
                --acceptance-csv fp32-acceptance-frozen.csv &&
            test -f full.done && python3 analyze.py --name full
        ) >"$d/run.log" 2>&1; then
            rm -- "$d/profiles/full.db"
            printf 'DONE %s %s\n' "$card" "$batch"
        else
            printf 'FAILED %s %s\n' "$card" "$batch"
            status=1
        fi
        batch=$((batch+1))
    done <"$run/gpu$card.txt"
    exit "$status"
)

pids=()
for card in {0..7}; do
    run_card "$card" >"$run/gpu$card.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
grep -hE '^(DONE|FAILED)' "$run"/gpu*.log
if [[ "$status" == 0 ]]; then
    python3 "$upload/summarize_frozen.py" "$run"
fi
exit "$status"
