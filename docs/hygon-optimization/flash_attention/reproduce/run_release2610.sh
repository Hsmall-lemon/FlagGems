#!/usr/bin/env bash
# Run on the allocated a01r4n15 compute node, once for each card sequentially.
set -euo pipefail
repo=$(realpath "$1")
candidate=$(realpath "$2")
card=$3
output=$4
[[ $(hostname) == a01r4n15 && ($card == 3 || $card == 4) ]]
mkdir "$output"
output=$(realpath "$output")
reproduce="$repo/docs/hygon-optimization/flash_attention/reproduce"
run_env() {
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$card" CUDA_VISIBLE_DEVICES="$card" \
        FLAGOS_DEV_CACHE="$HOME/.cache/flagos-dev/20261103-flash-node15-gpu$card" \
        bash "$candidate/apptainer-run" --source "$repo" -- \
        bash "$candidate/native-entry.sh" "$@"
}
cd "$output"
run_env "$reproduce/profile_release2610.py" --environment-root "$candidate" > run.log 2>&1
python3 -c 'import json; r=json.load(open("benchmark.json")); assert r.get("complete") and r.get("meets_90_percent")'
mkdir regression
cd regression
run_env "$reproduce/validate_release2610.py" --environment-root "$candidate" > run.log 2>&1
