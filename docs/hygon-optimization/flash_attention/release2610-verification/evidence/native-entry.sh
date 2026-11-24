set -euo pipefail
ulimit -c 0
C=/public/home/wzx226226/flash_attention_node15_20261109/native_candidates/13419
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export FA_SO_PATH="$C/venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
export ROCM_PATH="$C/dtk-26.10"
export HIP_PATH="$ROCM_PATH/hip"
export PATH="$C/venv/bin:$PATH"
hsa_lib="$ROCM_PATH/.hyhal/lib"
if [ "${FLASH2610_USE_HOST_HSA:-1}" = 1 ]; then hsa_lib=/opt/hyhal/lib; fi
export LD_LIBRARY_PATH="$C/python/lib:$C/venv/lib/python3.12/site-packages/torch/lib:$C/venv/lib/python3.12/site-packages/torch.libs:$C/venv/lib/python3.12/site-packages/flash_attn/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$hsa_lib:$ROCM_PATH/hip/lib:$ROCM_PATH/hsa/lib:$ROCM_PATH/dcc/lib:$ROCM_PATH/dcc/comgr/lib:$ROCM_PATH/opencl/lib:/public/software/mpi/openmpi-dtk_2604/5.0.3/lib:/public/software/base/ucx-dtk_2604/1.17.0/lib:/public/software/base/libevent-dtk_2604/2.1.12/lib:/public/software/base/hwloc-dtk_2604/2.12.2/lib:/public/software/base/pmix-dtk_2604/5.0.8/lib:/opt/hyhal/lib:/usr/local/hyhal/lib"
export TRITON_CACHE_DIR="$C/cache/gpu${HIP_VISIBLE_DEVICES}/triton"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
exec "$C/venv/bin/python" -u "$@"
