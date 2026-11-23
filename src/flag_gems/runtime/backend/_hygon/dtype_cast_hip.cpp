// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t round_bf16(uint32_t bits) {
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return 0x7fc0u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

template <bool Widen>
__device__ __forceinline__ void cast_four(const void* source, void* destination,
                                         unsigned index) {
    if constexpr (Widen) {
        const uint2 packed = static_cast<const uint2*>(source)[index];
        static_cast<uint4*>(destination)[index] = make_uint4(
            packed.x << 16, packed.x & 0xffff0000u,
            packed.y << 16, packed.y & 0xffff0000u);
    } else {
        const uint4 words = static_cast<const uint4*>(source)[index];
        static_cast<uint2*>(destination)[index] = make_uint2(
            round_bf16(words.x) | (round_bf16(words.y) << 16),
            round_bf16(words.z) | (round_bf16(words.w) << 16));
    }
}

template <bool Widen, int Vectors>
__global__ void cast_vectors(const void* source, void* destination) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
        const unsigned index = lane + vector * gridDim.x * blockDim.x;
        cast_four<Widen>(source, destination, index);
    }
}

template <bool Widen, int Vectors>
__global__ void cast_with_tail(const void* source, void* destination, unsigned count) {
    const unsigned first = blockIdx.x * blockDim.x * Vectors + threadIdx.x;
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
        const unsigned index = first + vector * blockDim.x;
        if (index < count / 4) {
            cast_four<Widen>(source, destination, index);
        } else if (index == count / 4) {
            for (unsigned i = index * 4; i < count; ++i) {
                if constexpr (Widen)
                    static_cast<uint32_t*>(destination)[i] =
                        uint32_t(static_cast<const uint16_t*>(source)[i]) << 16;
                else
                    static_cast<uint16_t*>(destination)[i] =
                        round_bf16(static_cast<const uint32_t*>(source)[i]);
            }
        }
    }
}

template <bool Widen>
void launch(const void* source, void* output, unsigned count,
            int threads, int vectors, hipStream_t stream) {
    const int blocks = count / (4 * threads * vectors);
    if (vectors == 1)
        hipLaunchKernelGGL((cast_vectors<Widen, 1>), blocks, threads, 0, stream, source, output);
    else if (vectors == 2)
        hipLaunchKernelGGL((cast_vectors<Widen, 2>), blocks, threads, 0, stream, source, output);
    else
        hipLaunchKernelGGL((cast_vectors<Widen, 4>), blocks, threads, 0, stream, source, output);
}

template <bool Widen>
void launch_tail(const void* source, void* output, unsigned count,
                 int threads, int vectors, hipStream_t stream) {
    const unsigned width = threads * vectors * 4;
    const unsigned blocks = (count + width - 1) / width;
    if (vectors == 1)
        hipLaunchKernelGGL((cast_with_tail<Widen, 1>), blocks, threads, 0, stream, source, output, count);
    else if (vectors == 2)
        hipLaunchKernelGGL((cast_with_tail<Widen, 2>), blocks, threads, 0, stream, source, output, count);
    else
        hipLaunchKernelGGL((cast_with_tail<Widen, 4>), blocks, threads, 0, stream, source, output, count);
}

extern "C" int launch_dtype_cast(const void* source, void* output, unsigned count,
                                 int widen, int threads, int vectors, hipStream_t stream) {
    if (count % (4 * threads * vectors) == 0) {
        if (widen)
            launch<true>(source, output, count, threads, vectors, stream);
        else
            launch<false>(source, output, count, threads, vectors, stream);
    } else if (widen) {
        launch_tail<true>(source, output, count, threads, vectors, stream);
    } else {
        launch_tail<false>(source, output, count, threads, vectors, stream);
    }
    return static_cast<int>(hipGetLastError());
}
