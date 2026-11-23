#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
    echo "Usage: bash run.sh CHECKOUT OUTPUT_DIR CARD model|scale" >&2
    exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -e -- "$1")
card=$3
suite=$4
case "$suite" in
    model) rows=${MODEL_ROWS:-$(seq -s, 771 780)} ;;
    scale) rows=${SCALE_ROWS:-$(seq -s, 1056 1080)} ;;
    *) echo "Unknown suite $suite" >&2; exit 2 ;;
esac
export DEV=${DEV:-/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization}
export DTK_ROOT=${DTK_ROOT:-/public/software/compiler/dtk-26.04}
umask 077
mkdir -- "$2"
cd -- "$2"
export FLAGOS_DEV_CACHE=${FLAGOS_DEV_CACHE:-$PWD/cache}
mkdir -p profiles "$FLAGOS_DEV_CACHE"
export FLAGOS_DEV_CACHE=$(realpath -e -- "$FLAGOS_DEV_CACHE")
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES=$card
cp -- "$scripts"/{cases.json,worker.py,analyze.py} .

bash "$DEV/bin/run" --profile boxing --source "$source" -- \
    env DTK_ROOT="$DTK_ROOT" "$DTK_ROOT/bin/hipprof" --hip-trace --hiptx-trace --output-type 0 \
    -o "$PWD/profiles/$suite" python -u worker.py \
    --name "$suite" --cases "$rows" --physical-card "$card" > "$suite.log" 2>&1
grep -qx DTYPE_COMPARE_COMPLETE "$suite.log"
python3 analyze.py "$PWD" "$suite" "$suite-records.jsonl"
rm -f -- "profiles/$suite.db" "profiles/$suite.json" \
    "profiles/$suite.hipkernel.csv" "profiles/$suite.hiptrace.csv" \
    "profiles/${suite}_"[0-9]*.json
