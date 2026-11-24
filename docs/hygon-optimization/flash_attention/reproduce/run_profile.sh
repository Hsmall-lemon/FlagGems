#!/usr/bin/env bash
# Usage: run_profile.sh CHECKOUT UNPACKED_NATIVE_PACKAGE CARD NEW_OUTPUT_DIR
set -euo pipefail
source_dir=$(realpath "$1")
native_package=$(realpath "$2")
card="$3"
output_dir="$4"
case "$card" in 3|4) ;; *) echo 'Only authorized physical cards 3 and 4 are allowed' >&2; exit 2 ;; esac
test -f "$native_package/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
test -f "$native_package/flash_attn/lib/libflash_attention.so"
mkdir "$output_dir"
cd "$output_dir"
export FLAGOS_DEV_CACHE="/public/home/wzx226226/.cache/flagos-dev/20261103-flash-node15-gpu${card}"
dev=/work1/share/scnethpc2601/flagos-dev/20261103-hygon-optimization
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES="$card" CUDA_VISIBLE_DEVICES="$card" \
    bash "$dev/bin/run" --source "$source_dir" -- bash -c '
        set -eu
        package="$1"
        script="$2"
        export PYTHONPATH="$package:$PYTHONPATH"
        export LD_LIBRARY_PATH="$package/flash_attn/lib:$LD_LIBRARY_PATH"
        export FA_SO_PATH="$package/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
        exec /public/software/compiler/dtk-26.04/bin/hipprof \
            --hip-trace --hiptx-trace --output-type 0 -o profile \
            python -u "$script" --mode tune --profile
    ' bash "$native_package" "$source_dir/docs/hygon-optimization/flash_attention/reproduce/bench_repo_final.py"
