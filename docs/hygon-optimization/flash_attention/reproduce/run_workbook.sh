#!/usr/bin/env bash
# Archived alternative DAS 2.8.4 experiment; does not establish requested DAS185 acceptance.
# Run on the allocated compute node; repeat sequentially for cards 3 and 4.
set -euo pipefail
repo=$(realpath "$1")
package=$(realpath "$2")
card=$3
output=$4
[[ $(hostname) == a01r4n15 && ($card == 3 || $card == 4) ]]
mkdir "$output"
output=$(realpath "$output")
dev=${FLAGOS_DEV_ROOT:-/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization}
reproduce="$repo/docs/hygon-optimization/flash_attention/reproduce"
run_env() {
    env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$card" CUDA_VISIBLE_DEVICES="$card" \
        FLAGOS_DEV_CACHE="${FLAGOS_DEV_CACHE:-$HOME/.cache/flagos-dev/20261103-flash-node15-gpu$card}" \
        bash "$dev/bin/run" --source "$repo" -- bash -c '
            set -eu
            package=$1; shift
            export PYTHONPATH="$package:$PYTHONPATH"
            export LD_LIBRARY_PATH="$package/flash_attn/lib:$LD_LIBRARY_PATH"
            export FA_SO_PATH="$package/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
            exec "$@"
        ' bash "$package" "$@"
}
cd "$output"
mkdir edges stability
(cd edges; run_env python -u "$reproduce/validate_edge_cases.py" > run.log 2>&1)
(cd "$repo"; run_env python -u -c '
import torch_fl
import torch
import flash_attn_2_cuda
torch.set_num_threads(16)
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_flash_attention_hygon_state.py"]))
' > "$output/pytest.log" 2>&1)
run_env /public/software/compiler/dtk-26.04/bin/hipprof \
    --hip-trace --hiptx-trace --output-type 0 -o profile \
    python -u "$reproduce/profile_workbook.py" > run.log 2>&1
run_env python "$reproduce/archive_workbook_trace.py" . device-trace.json.gz >> run.log 2>&1
run_env python "$reproduce/analyze_workbook_profile.py" device-trace.json.gz >> run.log 2>&1
python3 -c 'import json; assert json.load(open("workbook-summary.json"))["all_reconstructed_cases_meet_90"]'
for part in profile*.json profile.db; do
    [[ ! -f $part ]] || gzip -1 "$part"
done
(cd stability; run_env python -u "$reproduce/validate_workbook_stability.py" > run.log 2>&1)
echo "WORKBOOK_CARD_${card}_COMPLETE"
