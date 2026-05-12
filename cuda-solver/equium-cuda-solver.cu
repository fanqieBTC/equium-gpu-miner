// Equium CUDA Equihash (96,5) solver MVP.
//
// This CUDA backend keeps the whole Wagner pipeline on device:
// - CUDA generates the 131072 Equihash leaf rows for a nonce.
// - Each Wagner round builds 16-bit prefix buckets, scatters rows by bucket,
//   pairs rows inside buckets, and compacts the next row set on GPU.
// - The Rust CLI must still re-verify with equihash::is_valid_solution before
//   submitting any transaction.

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <openssl/sha.h>

static constexpr int INPUT_LEN = 81;
static constexpr int NONCE_LEN = 32;
static constexpr int LEAF_HASH_LEN = 12;
static constexpr int FULL_HASH_LEN = 60;
static constexpr int INIT_ROWS = 1 << 17;
static constexpr int INDICES_PER_HASH = 5;
static constexpr int CBYTES = 2;
static constexpr int ROUNDS = 5;
static constexpr int THREADS_PER_BLOCK = 256;
// Block size for init_rows: must be divisible by INDICES_PER_HASH (5).
// 240 = 48 groups × 5 — gives ~547 blocks on 131072 rows, good SM coverage.
static constexpr int INIT_BLOCK = 240;
static constexpr int PREFIX_BUCKETS = 1 << 16;
static constexpr int MAX_ROWS = INIT_ROWS * 8;
static constexpr int MAX_INDEX_LEN = 32;
static constexpr int HASH_STRIDE = LEAF_HASH_LEN;
static constexpr int INDEX_STRIDE = MAX_INDEX_LEN;
// Each Blake2b call produces INDICES_PER_HASH leaves; compute once and reuse.
static constexpr int HASH_GROUPS = (INIT_ROWS + INDICES_PER_HASH - 1) / INDICES_PER_HASH;

struct Row {
    uint8_t hash[LEAF_HASH_LEN];
    uint32_t indices[32];
    uint8_t hash_len;
    uint8_t index_len;
};

__device__ __constant__ uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

__device__ __constant__ uint8_t BLAKE2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ uint64_t load64_le(const uint8_t* p) {
    uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

__device__ __forceinline__ void store64_le(uint8_t* p, uint64_t v) {
    #pragma unroll
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

__device__ __forceinline__ void blake2b_g(uint64_t v[16], int a, int b, int c, int d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr64(v[b] ^ v[c], 63);
}

__device__ void blake2b_60_equihash(const uint8_t input[INPUT_LEN],
                                    const uint8_t nonce[NONCE_LEN],
                                    uint32_t block_index,
                                    uint8_t out[FULL_HASH_LEN]) {
    uint8_t block[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) block[i] = 0;
    for (int i = 0; i < INPUT_LEN; i++) block[i] = input[i];
    for (int i = 0; i < NONCE_LEN; i++) block[INPUT_LEN + i] = nonce[i];
    block[INPUT_LEN + NONCE_LEN + 0] = (uint8_t)(block_index);
    block[INPUT_LEN + NONCE_LEN + 1] = (uint8_t)(block_index >> 8);
    block[INPUT_LEN + NONCE_LEN + 2] = (uint8_t)(block_index >> 16);
    block[INPUT_LEN + NONCE_LEN + 3] = (uint8_t)(block_index >> 24);

    uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] = BLAKE2B_IV[i];

    // BLAKE2b parameter block:
    // digest_length=60, key_length=0, fanout=1, depth=1,
    // personal="ZcashPoW" || 96u32_le || 5u32_le.
    h[0] ^= 0x0101003cULL;
    h[6] ^= 0x576f50687361635aULL;
    h[7] ^= 0x0000000500000060ULL;

    uint64_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) m[i] = load64_le(block + i * 8);

    uint64_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = h[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i + 8] = BLAKE2B_IV[i];
    v[12] ^= 117ULL;
    v[14] = ~v[14];

    #pragma unroll
    for (int r = 0; r < 12; r++) {
        const uint8_t* s = BLAKE2B_SIGMA[r];
        blake2b_g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        blake2b_g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        blake2b_g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        blake2b_g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        blake2b_g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        blake2b_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2b_g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        blake2b_g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];

    uint8_t digest[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) store64_le(digest + i * 8, h[i]);
    #pragma unroll
    for (int i = 0; i < FULL_HASH_LEN; i++) out[i] = digest[i];
}

__global__ void generate_leaves_kernel(const uint8_t* input,
                                       const uint8_t* nonce,
                                       uint8_t* leaf_hashes) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= INIT_ROWS) return;

    uint8_t digest[FULL_HASH_LEN];
    uint32_t block_index = i / INDICES_PER_HASH;
    uint32_t segment = i % INDICES_PER_HASH;
    blake2b_60_equihash(input, nonce, block_index, digest);

    uint8_t* dst = leaf_hashes + (size_t)i * LEAF_HASH_LEN;
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++) {
        dst[j] = digest[segment * LEAF_HASH_LEN + j];
    }
}

// Compute one Blake2b hash per group of INDICES_PER_HASH rows.
// Replaces the old init_rows_kernel which recomputed the same hash 5× per group.
__global__ void compute_hashes_kernel(const uint8_t* input,
                                       const uint8_t* nonce,
                                       uint8_t* hashes) {
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= HASH_GROUPS) return;
    blake2b_60_equihash(input, nonce, g, hashes + (size_t)g * FULL_HASH_LEN);
}

// Fill Row structs from the precomputed hash buffer — one thread per row.
__global__ void build_rows_kernel(const uint8_t* hashes, Row* rows) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= INIT_ROWS) return;
    uint32_t g = i / INDICES_PER_HASH;
    uint32_t seg = i % INDICES_PER_HASH;
    const uint8_t* digest = hashes + (size_t)g * FULL_HASH_LEN;
    Row& r = rows[i];
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++)
        r.hash[j] = digest[seg * LEAF_HASH_LEN + j];
    r.hash_len = LEAF_HASH_LEN;
    r.index_len = 1;
    r.indices[0] = i;
}

// Shared-memory dedup: only 1 of every 5 threads computes Blake2b; the result
// lives in L1 (shared mem) and all 5 threads read from it.  Gives 5× fewer
// Blake2b calls while keeping the same block count and SM occupancy as the
// original single-kernel design.  blockDim.x must be a multiple of INDICES_PER_HASH.
__global__ void init_rows_shmem(const uint8_t* input,
                                 const uint8_t* nonce,
                                 Row* rows) {
    extern __shared__ uint8_t shmem[];  // (blockDim.x / INDICES_PER_HASH) * FULL_HASH_LEN

    uint32_t i = (uint32_t)blockIdx.x * INIT_BLOCK + threadIdx.x;
    if (i >= INIT_ROWS) return;

    uint32_t global_group = i / INDICES_PER_HASH;
    uint32_t seg           = threadIdx.x % INDICES_PER_HASH;
    uint32_t local_group   = threadIdx.x / INDICES_PER_HASH;
    uint8_t* group_digest  = shmem + (size_t)local_group * FULL_HASH_LEN;

    if (seg == 0)
        blake2b_60_equihash(input, nonce, global_group, group_digest);
    __syncthreads();

    Row& r = rows[i];
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++)
        r.hash[j] = group_digest[seg * LEAF_HASH_LEN + j];
    r.hash_len = LEAF_HASH_LEN;
    r.index_len = 1;
    r.indices[0] = i;
}

// Kept for reference; superseded by init_rows_shmem.
__global__ void init_rows_kernel(const uint8_t* input,
                                 const uint8_t* nonce,
                                 Row* rows) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= INIT_ROWS) return;

    uint8_t digest[FULL_HASH_LEN];
    uint32_t block_index = i / INDICES_PER_HASH;
    uint32_t segment = i % INDICES_PER_HASH;
    blake2b_60_equihash(input, nonce, block_index, digest);

    Row& r = rows[i];
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++) {
        r.hash[j] = digest[segment * LEAF_HASH_LEN + j];
    }
    r.hash_len = LEAF_HASH_LEN;
    r.index_len = 1;
    r.indices[0] = i;
}

__device__ __forceinline__ uint32_t row_prefix16(const Row& row) {
    return ((uint32_t)row.hash[0] << 8) | (uint32_t)row.hash[1];
}

// ---- SoA kernels: separate hash and index buffers for the hot path ----

// Plain SoA init — no shared memory, no __syncthreads.
// Each thread independently computes its Blake2b and writes its slice.
// Benchmark on 4090: faster than the shmem variant (shmem __syncthreads
// overhead exceeds the Blake2b redundancy savings at this occupancy).
__global__ void init_rows_soa(const uint8_t* input,
                               const uint8_t* nonce,
                               uint8_t* row_hashes,
                               uint32_t* row_indices) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= INIT_ROWS) return;

    uint8_t digest[FULL_HASH_LEN];
    uint32_t block_index = i / INDICES_PER_HASH;
    uint32_t segment     = i % INDICES_PER_HASH;
    blake2b_60_equihash(input, nonce, block_index, digest);

    uint8_t* dst_hash = row_hashes + (size_t)i * HASH_STRIDE;
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++) {
        dst_hash[j] = digest[segment * LEAF_HASH_LEN + j];
    }
    row_indices[(size_t)i * INDEX_STRIDE] = i;
}

__global__ void init_rows_shmem_soa(const uint8_t* input,
                                    const uint8_t* nonce,
                                    uint8_t* row_hashes,
                                    uint32_t* row_indices) {
    extern __shared__ uint8_t shmem[];

    uint32_t i = (uint32_t)blockIdx.x * INIT_BLOCK + threadIdx.x;
    if (i >= INIT_ROWS) return;

    uint32_t global_group = i / INDICES_PER_HASH;
    uint32_t seg          = threadIdx.x % INDICES_PER_HASH;
    uint32_t local_group  = threadIdx.x / INDICES_PER_HASH;
    uint8_t* group_digest = shmem + (size_t)local_group * FULL_HASH_LEN;

    if (seg == 0)
        blake2b_60_equihash(input, nonce, global_group, group_digest);
    __syncthreads();

    uint8_t* dst_hash = row_hashes + (size_t)i * HASH_STRIDE;
    #pragma unroll
    for (int j = 0; j < LEAF_HASH_LEN; j++) {
        dst_hash[j] = group_digest[seg * LEAF_HASH_LEN + j];
    }
    row_indices[(size_t)i * INDEX_STRIDE] = i;
}

__device__ __forceinline__ uint32_t row_prefix16_soa(const uint8_t* hashes, uint32_t row) {
    const uint8_t* h = hashes + (size_t)row * HASH_STRIDE;
    return ((uint32_t)h[0] << 8) | (uint32_t)h[1];
}

__global__ void histogram_prefix_soa(const uint8_t* hashes,
                                     uint32_t row_count,
                                     uint32_t* counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    atomicAdd(&counts[row_prefix16_soa(hashes, i)], 1);
}

__global__ void scatter_prefix_soa(const uint8_t* hashes,
                                   const uint32_t* indices,
                                   uint8_t* sorted_hashes,
                                   uint32_t* sorted_indices,
                                   uint32_t row_count,
                                   const uint32_t* offsets,
                                   uint32_t* write_counts,
                                   uint8_t hash_len,
                                   uint8_t index_len) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;

    uint32_t prefix = row_prefix16_soa(hashes, i);
    uint32_t pos = offsets[prefix] + atomicAdd(&write_counts[prefix], 1);

    const uint8_t* src_hash = hashes + (size_t)i * HASH_STRIDE;
    uint8_t* dst_hash = sorted_hashes + (size_t)pos * HASH_STRIDE;
    for (int h = 0; h < hash_len; h++) {
        dst_hash[h] = src_hash[h];
    }

    const uint32_t* src_idx = indices + (size_t)i * INDEX_STRIDE;
    uint32_t* dst_idx = sorted_indices + (size_t)pos * INDEX_STRIDE;
    for (int idx = 0; idx < index_len; idx++) {
        dst_idx[idx] = src_idx[idx];
    }
}

__device__ __forceinline__ bool distinct_indices_soa(const uint32_t* a,
                                                     const uint32_t* b,
                                                     uint8_t index_len) {
    for (int i = 0; i < index_len; i++) {
        for (int j = 0; j < index_len; j++) {
            if (a[i] == b[j]) return false;
        }
    }
    return true;
}

__global__ void pair_buckets_per_thread_soa(const uint8_t* sorted_hashes,
                                            const uint32_t* sorted_indices,
                                            uint8_t* out_hashes,
                                            uint32_t* out_indices,
                                            const uint32_t* counts,
                                            const uint32_t* offsets,
                                            uint32_t* out_count,
                                            uint32_t* overflow,
                                            uint8_t hash_len,
                                            uint8_t index_len) {
    uint32_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= PREFIX_BUCKETS) return;
    uint32_t count = counts[bucket];
    if (count < 2) return;

    uint32_t start = offsets[bucket];
    uint8_t out_hash_len = hash_len - CBYTES;
    for (uint32_t ia = 0; ia < count; ia++) {
        for (uint32_t ib = ia + 1; ib < count; ib++) {
            uint32_t row_a = start + ia;
            uint32_t row_b = start + ib;
            const uint8_t* hash_a = sorted_hashes + (size_t)row_a * HASH_STRIDE;
            const uint8_t* hash_b = sorted_hashes + (size_t)row_b * HASH_STRIDE;
            const uint32_t* idx_a = sorted_indices + (size_t)row_a * INDEX_STRIDE;
            const uint32_t* idx_b = sorted_indices + (size_t)row_b * INDEX_STRIDE;
            if (!distinct_indices_soa(idx_a, idx_b, index_len)) continue;

            uint32_t pos = atomicAdd(out_count, 1);
            if (pos >= MAX_ROWS) {
                *overflow = 1;
                return;
            }

            uint8_t* dst_hash = out_hashes + (size_t)pos * HASH_STRIDE;
            uint32_t* dst_idx = out_indices + (size_t)pos * INDEX_STRIDE;
            for (int h = 0; h < out_hash_len; h++) {
                dst_hash[h] = hash_a[h + CBYTES] ^ hash_b[h + CBYTES];
            }

            const uint32_t* first = idx_a;
            const uint32_t* second = idx_b;
            if (idx_a[0] > idx_b[0]) {
                first = idx_b;
                second = idx_a;
            }
            for (int idx = 0; idx < index_len; idx++) {
                dst_idx[idx] = first[idx];
            }
            for (int idx = 0; idx < index_len; idx++) {
                dst_idx[index_len + idx] = second[idx];
            }
        }
    }
}

__global__ void find_solution_soa(const uint8_t* hashes,
                                  const uint32_t* indices,
                                  uint32_t row_count,
                                  uint8_t hash_len,
                                  uint8_t index_len,
                                  uint32_t* found,
                                  uint32_t* out_indices) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    if (*found) return;

    const uint8_t* h = hashes + (size_t)row * HASH_STRIDE;
    for (int i = 0; i < hash_len; i++) {
        if (h[i] != 0) return;
    }

    if (atomicCAS(found, 0u, 1u) == 0u) {
        const uint32_t* src = indices + (size_t)row * INDEX_STRIDE;
        for (int idx = 0; idx < index_len; idx++) {
            out_indices[idx] = src[idx];
        }
    }
}

__device__ __forceinline__ void device_copy_live_row(const Row& src, Row& dst) {
    dst.hash_len = src.hash_len;
    dst.index_len = src.index_len;
    for (int h = 0; h < src.hash_len; h++) {
        dst.hash[h] = src.hash[h];
    }
    for (int idx = 0; idx < src.index_len; idx++) {
        dst.indices[idx] = src.indices[idx];
    }
}

__global__ void histogram_prefix_kernel(const Row* rows,
                                        uint32_t row_count,
                                        uint32_t* counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    atomicAdd(&counts[row_prefix16(rows[i])], 1);
}

__global__ void scatter_prefix_kernel(const Row* rows,
                                      Row* sorted_rows,
                                      uint32_t row_count,
                                      const uint32_t* offsets,
                                      uint32_t* write_counts) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count) return;
    uint32_t prefix = row_prefix16(rows[i]);
    uint32_t pos = offsets[prefix] + atomicAdd(&write_counts[prefix], 1);
    device_copy_live_row(rows[i], sorted_rows[pos]);
}

__device__ __forceinline__ bool device_distinct_indices(const Row& a, const Row& b) {
    for (int i = 0; i < a.index_len; i++) {
        for (int j = 0; j < b.index_len; j++) {
            if (a.indices[i] == b.indices[j]) return false;
        }
    }
    return true;
}

__device__ __forceinline__ void device_pair_rows(const Row& a, const Row& b, Row& out) {
    out.hash_len = a.hash_len - CBYTES;
    out.index_len = a.index_len + b.index_len;

    for (int h = 0; h < out.hash_len; h++) {
        out.hash[h] = a.hash[h + CBYTES] ^ b.hash[h + CBYTES];
    }

    const Row* first = &a;
    const Row* second = &b;
    if (a.indices[0] > b.indices[0]) {
        first = &b;
        second = &a;
    }
    for (int idx = 0; idx < first->index_len; idx++) {
        out.indices[idx] = first->indices[idx];
    }
    for (int idx = 0; idx < second->index_len; idx++) {
        out.indices[first->index_len + idx] = second->indices[idx];
    }
}

__device__ __forceinline__ void pair_from_linear(uint32_t pair_id,
                                                 uint32_t count,
                                                 uint32_t& ia,
                                                 uint32_t& ib) {
    uint32_t remaining = pair_id;
    for (uint32_t a = 0; a + 1 < count; a++) {
        uint32_t pairs_for_a = count - a - 1;
        if (remaining < pairs_for_a) {
            ia = a;
            ib = a + 1 + remaining;
            return;
        }
        remaining -= pairs_for_a;
    }
    ia = 0;
    ib = 0;
}

// Old kernel kept for reference — superseded by pair_buckets_per_thread.
__global__ void pair_buckets_kernel(const Row* sorted_rows,
                                    Row* out_rows,
                                    const uint32_t* counts,
                                    const uint32_t* offsets,
                                    uint32_t* out_count,
                                    uint32_t* overflow) {
    uint32_t bucket = blockIdx.x;
    if (bucket >= PREFIX_BUCKETS) return;
    uint32_t count = counts[bucket];
    if (count < 2) return;

    uint32_t start = offsets[bucket];
    uint32_t pair_count = count * (count - 1) / 2;
    for (uint32_t pair_id = threadIdx.x; pair_id < pair_count; pair_id += blockDim.x) {
        uint32_t ia = 0;
        uint32_t ib = 0;
        pair_from_linear(pair_id, count, ia, ib);
        const Row& a = sorted_rows[start + ia];
        const Row& b = sorted_rows[start + ib];
        if (!device_distinct_indices(a, b)) continue;

        uint32_t pos = atomicAdd(out_count, 1);
        if (pos >= MAX_ROWS) {
            *overflow = 1;
            continue;
        }
        device_pair_rows(a, b, out_rows[pos]);
    }
}

// One thread per bucket: 65536 threads launched as 256 blocks × 256 threads.
// Reduces block-scheduling overhead 256× vs the old 1-block-per-bucket design
// while keeping 100% thread coverage of all buckets.
__global__ void pair_buckets_per_thread(const Row* sorted_rows,
                                        Row* out_rows,
                                        const uint32_t* counts,
                                        const uint32_t* offsets,
                                        uint32_t* out_count,
                                        uint32_t* overflow) {
    uint32_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= PREFIX_BUCKETS) return;
    uint32_t count = counts[bucket];
    if (count < 2) return;

    uint32_t start = offsets[bucket];
    for (uint32_t ia = 0; ia < count; ia++) {
        for (uint32_t ib = ia + 1; ib < count; ib++) {
            const Row& a = sorted_rows[start + ia];
            const Row& b = sorted_rows[start + ib];
            if (!device_distinct_indices(a, b)) continue;

            uint32_t pos = atomicAdd(out_count, 1);
            if (pos >= MAX_ROWS) {
                *overflow = 1;
                return;
            }
            device_pair_rows(a, b, out_rows[pos]);
        }
    }
}

static std::vector<uint8_t> parse_hex(std::string hex, size_t expected, const char* name) {
    if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) hex = hex.substr(2);
    if (hex.size() != expected * 2) {
        throw std::runtime_error(std::string(name) + " must be " + std::to_string(expected) + " bytes");
    }
    std::vector<uint8_t> out(expected);
    for (size_t i = 0; i < expected; i++) {
        out[i] = (uint8_t)std::stoul(hex.substr(i * 2, 2), nullptr, 16);
    }
    return out;
}

static std::string hex_of(const uint8_t* data, size_t len) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) out << std::setw(2) << (unsigned)data[i];
    return out.str();
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
    for (int i = 1; i < argc; i++) {
        std::string item(argv[i]);
        if (item == key && i + 1 < argc) return argv[i + 1];
        if (item.rfind(key + "=", 0) == 0) return item.substr(key.size() + 1);
    }
    return fallback;
}

static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static std::array<uint8_t, NONCE_LEN> make_nonce(uint64_t seed, uint64_t attempt) {
    std::array<uint8_t, NONCE_LEN> nonce{};
    uint64_t x = seed ^ (attempt * 0xd1342543de82ef95ULL);
    for (int word = 0; word < 4; word++) {
        uint64_t v = splitmix64(x);
        for (int b = 0; b < 8; b++) nonce[word * 8 + b] = (uint8_t)(v >> (8 * b));
    }
    return nonce;
}

static bool distinct_indices(const Row& a, const Row& b) {
    for (int i = 0; i < a.index_len; i++) {
        for (int j = 0; j < b.index_len; j++) {
            if (a.indices[i] == b.indices[j]) return false;
        }
    }
    return true;
}

static std::vector<Row> wagner_round(std::vector<Row>& rows) {
    if (rows.empty()) return {};
    const uint8_t hash_len = rows[0].hash_len;
    std::sort(rows.begin(), rows.end(), [hash_len](const Row& a, const Row& b) {
        return std::memcmp(a.hash, b.hash, hash_len) < 0;
    });

    std::vector<Row> out;
    out.reserve(rows.size());
    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i + 1;
        while (j < rows.size() &&
               rows[i].hash[0] == rows[j].hash[0] &&
               rows[i].hash[1] == rows[j].hash[1]) {
            j++;
        }

        for (size_t ia = i; ia < j; ia++) {
            for (size_t ib = ia + 1; ib < j; ib++) {
                const Row& a = rows[ia];
                const Row& b = rows[ib];
                if (!distinct_indices(a, b)) continue;

                Row r{};
                r.hash_len = hash_len - CBYTES;
                r.index_len = a.index_len + b.index_len;
                for (int h = 0; h < r.hash_len; h++) {
                    r.hash[h] = a.hash[h + CBYTES] ^ b.hash[h + CBYTES];
                }
                const Row* first = &a;
                const Row* second = &b;
                if (a.indices[0] > b.indices[0]) {
                    first = &b;
                    second = &a;
                }
                for (int idx = 0; idx < first->index_len; idx++) r.indices[idx] = first->indices[idx];
                for (int idx = 0; idx < second->index_len; idx++) {
                    r.indices[first->index_len + idx] = second->indices[idx];
                }
                out.push_back(r);
            }
        }
        i = j;
    }
    return out;
}

static std::vector<uint8_t> compress_indices(const uint32_t* indices, size_t count) {
    const int bits_per = 17;
    const size_t total_bits = bits_per * count;
    std::vector<uint8_t> out((total_bits + 7) / 8, 0);
    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        uint32_t idx = indices[i];
        for (int b = bits_per - 1; b >= 0; b--) {
            uint8_t bit = (idx >> b) & 1;
            size_t byte = pos / 8;
            int shift = 7 - (int)(pos % 8);
            out[byte] |= bit << shift;
            pos++;
        }
    }
    return out;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while (0)

// Pre-allocated GPU buffers — allocated once, reused across nonces.
struct SolverContext {
    uint8_t* d_input = nullptr;
    uint8_t* d_nonce = nullptr;
    uint8_t* d_hashes = nullptr;  // HASH_GROUPS * FULL_HASH_LEN, for dedup Blake2b
    uint8_t* d_row_hashes_a = nullptr;
    uint8_t* d_row_hashes_b = nullptr;
    uint8_t* d_sorted_hashes = nullptr;
    uint32_t* d_row_indices_a = nullptr;
    uint32_t* d_row_indices_b = nullptr;
    uint32_t* d_sorted_indices = nullptr;
    uint32_t* d_counts = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_write_counts = nullptr;
    uint32_t* d_out_count = nullptr;
    uint32_t* d_overflow = nullptr;
    uint32_t* d_solution_found = nullptr;
    uint32_t* d_solution_indices = nullptr;
    bool allocated = false;

    void alloc() {
        if (allocated) return;
        CUDA_CHECK(cudaMalloc((void**)&d_input, INPUT_LEN));
        CUDA_CHECK(cudaMalloc((void**)&d_nonce, NONCE_LEN));
        CUDA_CHECK(cudaMalloc((void**)&d_hashes, (size_t)HASH_GROUPS * FULL_HASH_LEN));
        CUDA_CHECK(cudaMalloc((void**)&d_row_hashes_a, (size_t)MAX_ROWS * HASH_STRIDE));
        CUDA_CHECK(cudaMalloc((void**)&d_row_hashes_b, (size_t)MAX_ROWS * HASH_STRIDE));
        CUDA_CHECK(cudaMalloc((void**)&d_sorted_hashes, (size_t)MAX_ROWS * HASH_STRIDE));
        CUDA_CHECK(cudaMalloc((void**)&d_row_indices_a, (size_t)MAX_ROWS * INDEX_STRIDE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_row_indices_b, (size_t)MAX_ROWS * INDEX_STRIDE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_sorted_indices, (size_t)MAX_ROWS * INDEX_STRIDE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_counts, PREFIX_BUCKETS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_offsets, PREFIX_BUCKETS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_write_counts, PREFIX_BUCKETS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_out_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_overflow, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_solution_found, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_solution_indices, INDEX_STRIDE * sizeof(uint32_t)));
        allocated = true;
    }

    void release() {
        if (!allocated) return;
        cudaFree(d_input); cudaFree(d_nonce); cudaFree(d_hashes);
        cudaFree(d_row_hashes_a); cudaFree(d_row_hashes_b); cudaFree(d_sorted_hashes);
        cudaFree(d_row_indices_a); cudaFree(d_row_indices_b); cudaFree(d_sorted_indices);
        cudaFree(d_counts); cudaFree(d_offsets); cudaFree(d_write_counts);
        cudaFree(d_out_count); cudaFree(d_overflow);
        cudaFree(d_solution_found); cudaFree(d_solution_indices);
        allocated = false;
    }

    ~SolverContext() { release(); }
};

static bool solve_one_nonce(SolverContext& ctx,
                            const std::array<uint8_t, INPUT_LEN>& input,
                            const std::array<uint8_t, NONCE_LEN>& nonce,
                            std::vector<uint8_t>& soln_indices) {
    ctx.alloc();
    CUDA_CHECK(cudaMemcpy(ctx.d_input, input.data(), INPUT_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_nonce, nonce.data(), NONCE_LEN, cudaMemcpyHostToDevice));

    // We need local copies of the pointers since we swap them each round.
    uint8_t* d_hashes_a = ctx.d_row_hashes_a;
    uint8_t* d_hashes_b = ctx.d_row_hashes_b;
    uint32_t* d_indices_a = ctx.d_row_indices_a;
    uint32_t* d_indices_b = ctx.d_row_indices_b;

    int blocks = (INIT_ROWS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_rows_soa<<<blocks, THREADS_PER_BLOCK>>>(
        ctx.d_input, ctx.d_nonce, d_hashes_a, d_indices_a);
    CUDA_CHECK(cudaGetLastError());

    uint32_t row_count = INIT_ROWS;
    uint8_t hash_len = LEAF_HASH_LEN;
    uint8_t index_len = 1;
    for (int round = 0; round < ROUNDS; round++) {
        CUDA_CHECK(cudaMemset(ctx.d_counts, 0, PREFIX_BUCKETS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(ctx.d_write_counts, 0, PREFIX_BUCKETS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(ctx.d_out_count, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(ctx.d_overflow, 0, sizeof(uint32_t)));

        int row_blocks = (row_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        histogram_prefix_soa<<<row_blocks, THREADS_PER_BLOCK>>>(
            d_hashes_a, row_count, ctx.d_counts);
        CUDA_CHECK(cudaGetLastError());

        thrust::device_ptr<uint32_t> counts_ptr = thrust::device_pointer_cast(ctx.d_counts);
        thrust::device_ptr<uint32_t> offsets_ptr = thrust::device_pointer_cast(ctx.d_offsets);
        thrust::exclusive_scan(counts_ptr, counts_ptr + PREFIX_BUCKETS, offsets_ptr);

        scatter_prefix_soa<<<row_blocks, THREADS_PER_BLOCK>>>(
            d_hashes_a, d_indices_a, ctx.d_sorted_hashes, ctx.d_sorted_indices,
            row_count, ctx.d_offsets, ctx.d_write_counts, hash_len, index_len);
        CUDA_CHECK(cudaGetLastError());

        pair_buckets_per_thread_soa<<<(PREFIX_BUCKETS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                                      THREADS_PER_BLOCK>>>(
            ctx.d_sorted_hashes, ctx.d_sorted_indices, d_hashes_b, d_indices_b,
            ctx.d_counts, ctx.d_offsets, ctx.d_out_count, ctx.d_overflow,
            hash_len, index_len);
        CUDA_CHECK(cudaGetLastError());

        uint32_t overflow = 0, next_count = 0;
        CUDA_CHECK(cudaMemcpy(&overflow, ctx.d_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&next_count, ctx.d_out_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (overflow || next_count > MAX_ROWS) {
            return false; // overflow — skip this nonce instead of crashing
        }
        if (next_count == 0) return false;

        std::swap(d_hashes_a, d_hashes_b);
        std::swap(d_indices_a, d_indices_b);
        row_count = next_count;
        hash_len -= CBYTES;
        index_len *= 2;
    }

    CUDA_CHECK(cudaMemset(ctx.d_solution_found, 0, sizeof(uint32_t)));
    int final_blocks = (row_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    find_solution_soa<<<final_blocks, THREADS_PER_BLOCK>>>(
        d_hashes_a, d_indices_a, row_count, hash_len, index_len,
        ctx.d_solution_found, ctx.d_solution_indices);
    CUDA_CHECK(cudaGetLastError());

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, ctx.d_solution_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (found) {
        std::vector<uint32_t> winner_indices(index_len);
        CUDA_CHECK(cudaMemcpy(winner_indices.data(), ctx.d_solution_indices,
                              (size_t)index_len * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        soln_indices = compress_indices(winner_indices.data(), index_len);
        return true;
    }
    return false;
}

// ---- Daemon mode: persistent stdin/stdout JSON loop ----

static std::string read_json_field(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        return (end != std::string::npos) ? json.substr(pos + 1, end - pos - 1) : "";
    }
    auto end = json.find_first_of(",}\n", pos);
    return json.substr(pos, end - pos);
}

/// SHA256(soln_indices || input), then compare against target (big-endian).
static bool solution_hash_under_target(
    const std::vector<uint8_t>& soln_indices,
    const std::array<uint8_t, INPUT_LEN>& input,
    const uint8_t* target, size_t target_len) {
    SHA256_CTX sha;
    SHA256_Init(&sha);
    SHA256_Update(&sha, soln_indices.data(), soln_indices.size());
    SHA256_Update(&sha, input.data(), INPUT_LEN);
    uint8_t hash[32];
    SHA256_Final(hash, &sha);
    // Big-endian comparison: hash < target means under target
    for (size_t i = 0; i < 32 && i < target_len; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

static void run_daemon() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("no CUDA devices found");
    CUDA_CHECK(cudaSetDevice(0));

    SolverContext ctx;
    std::cout << "{\"type\":\"ready\",\"attempts\":0}" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string cmd = read_json_field(line, "cmd");
        if (cmd == "quit") break;
        if (cmd != "solve") {
            std::cout << "{\"type\":\"error\",\"message\":\"unknown cmd\",\"attempts\":0}" << std::endl;
            continue;
        }

        std::string input_hex = read_json_field(line, "input");
        uint64_t max_nonces = 4096;
        std::string mn = read_json_field(line, "max_nonces");
        if (!mn.empty()) max_nonces = std::stoull(mn);
        std::string seed_str = read_json_field(line, "seed");
        uint64_t seed = seed_str.empty() ? 1311768467463790320ULL : std::stoull(seed_str);

        // Parse optional target for pre-filtering
        std::string target_hex = read_json_field(line, "target");
        std::vector<uint8_t> target_vec;
        bool has_target = false;
        if (!target_hex.empty()) {
            try {
                target_vec = parse_hex(target_hex, 32, "target");
                has_target = true;
            } catch (...) {
                // Ignore bad target, skip filtering
            }
        }

        try {
            auto input_vec = parse_hex(input_hex, INPUT_LEN, "input");
            std::array<uint8_t, INPUT_LEN> input{};
            std::copy(input_vec.begin(), input_vec.end(), input.begin());

            uint64_t attempts = 0;
            bool found = false;
            for (uint64_t n = 0; n < max_nonces; n++) {
                auto nonce = make_nonce(seed ^ (uint64_t)clock(), n);
                attempts++;
                std::vector<uint8_t> soln_indices;
                if (solve_one_nonce(ctx, input, nonce, soln_indices)) {
                    // If target provided, check solution hash against target
                    if (has_target && !solution_hash_under_target(
                            soln_indices, input, target_vec.data(), target_vec.size())) {
                        continue; // above target, try next nonce
                    }
                    std::cout << "{\"type\":\"found\",\"nonce\":\""
                              << hex_of(nonce.data(), nonce.size())
                              << "\",\"soln_indices\":\""
                              << hex_of(soln_indices.data(), soln_indices.size())
                              << "\",\"attempts\":" << attempts << "}" << std::endl;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "{\"type\":\"not_found\",\"attempts\":" << attempts << "}" << std::endl;
            }
        } catch (const std::exception& ex) {
            std::cout << "{\"type\":\"error\",\"message\":\"" << ex.what()
                      << "\",\"attempts\":0}" << std::endl;
        }
    }
}

// ---- CLI entry point ----

int main(int argc, char** argv) {
    try {
        // Check for --daemon flag
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--daemon") {
                run_daemon();
                return 0;
            }
        }

        // Legacy one-shot mode
        std::string input_hex = get_arg(argc, argv, "--input");
        if (input_hex.empty()) throw std::runtime_error("--input 0x... is required");
        uint64_t max_nonces = std::stoull(get_arg(argc, argv, "--max-nonces", "4096"));
        uint64_t seed = std::stoull(get_arg(argc, argv, "--seed", "1311768467463790320"));
        std::string fixed_nonce_hex = get_arg(argc, argv, "--nonce");

        auto input_vec = parse_hex(input_hex, INPUT_LEN, "input");
        std::array<uint8_t, INPUT_LEN> input{};
        std::copy(input_vec.begin(), input_vec.end(), input.begin());

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) throw std::runtime_error("no CUDA devices found");
        CUDA_CHECK(cudaSetDevice(0));

        SolverContext ctx;
        uint64_t attempts = 0;
        for (uint64_t n = 0; n < max_nonces; n++) {
            std::array<uint8_t, NONCE_LEN> nonce{};
            if (!fixed_nonce_hex.empty()) {
                auto nonce_vec = parse_hex(fixed_nonce_hex, NONCE_LEN, "nonce");
                std::copy(nonce_vec.begin(), nonce_vec.end(), nonce.begin());
                max_nonces = 1;
            } else {
                nonce = make_nonce(seed, n);
            }
            attempts++;

            std::vector<uint8_t> soln_indices;
            if (solve_one_nonce(ctx, input, nonce, soln_indices)) {
                std::cout << "{\"type\":\"found\",\"nonce\":\"" << hex_of(nonce.data(), nonce.size())
                          << "\",\"soln_indices\":\"" << hex_of(soln_indices.data(), soln_indices.size())
                          << "\",\"attempts\":" << attempts << "}" << std::endl;
                return 0;
            }
        }

        std::cout << "{\"type\":\"not_found\",\"attempts\":" << attempts << "}" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "{\"type\":\"error\",\"message\":\"" << ex.what() << "\",\"attempts\":0}" << std::endl;
        return 1;
    }
}
