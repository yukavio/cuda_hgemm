// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma async pg2s hgemm

#include "common.h"
#include "cuda_fp16.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256 // ELEM PER CUDA BLOCK
#define BLOCK_COLS 128

// ELEM PER WARP
#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_col_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_row_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_col_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_row_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_col_TILES 8  // WARP_COLS / MMA_N
#define WARP_row_TILES 4  // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_col_WARPS * BLOCK_row_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define SMEM_BANK_ROWS 2  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4


// #define use_branch_1
// #define use_branch_2
//#define use_branch_4
#define use_branch_8

#ifdef use_branch_1
#define BRANCH 1
#endif
#ifdef use_branch_2
#define BRANCH 2
#endif
#ifdef use_branch_4
#define BRANCH 4
#endif
#ifdef use_branch_8
#define BRANCH 8
#endif


union Half2Uint32 {
    uint32_t u32;
    half2 h2;
};

__global__ void mmaAsyncPg2sKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                   size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_row_TILES) : (blockIdx.y * BLOCK_row_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_col_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    constexpr size_t smem_cache_off = BLOCK_ROWS + BLOCK_COLS;

    half *smem_warp_tile_row_ptr = &smem[0][0] + (warp_id / BLOCK_col_WARPS) * C_SMEM_STRIDE * WARP_ROWS / BRANCH;
    const half *smem_warp_stream_ptr = &smem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE / BRANCH;

    // 等于在写的时候 MMA_N / banch 
    const size_t gmem_idx = ((block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N) / BRANCH;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    Half2Uint32 RC[WARP_row_TILES][WARP_col_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_row_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_col_TILES; ++j) {
            RC[i][j][0].u32 = 0;
            RC[i][j][1].u32 = 0;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_off = 0;
    size_t smem_load_off = smem_cache_off;

    size_t A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    int4 *A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    size_t B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    int4 *B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();

#pragma unroll
    for (size_t tile_k = CHUNK_K; tile_k < K_tiles; tile_k += CHUNK_K) {
        smem_store_off ^= smem_cache_off;
        smem_load_off ^= smem_cache_off;

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_row_TILES][4];
            uint32_t RB[WARP_col_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_row_TILES; ++i) {
                size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_col_WARPS) * WARP_ROWS + i * MMA_M;
                uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[A_smem_idx + lane_id % 16]
                         [(k_step * MMA_K + (lane_id / 16) * 8 +
                           (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                          AB_SMEM_STRIDE]);

                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_col_TILES; ++j) {
                size_t B_smem_idx =
                    smem_load_off + B_smem_idx_off + (warp_id % BLOCK_col_WARPS) * WARP_COLS + j * MMA_N;
                uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                    &smem[B_smem_idx + lane_id % 8]
                         [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                           (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                          AB_SMEM_STRIDE]);

                LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_row_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_col_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_col_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0].u32, RC[i][j_s][1].u32, RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0].u32, RC[i][j_s][1].u32);
                }
            }
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        uint32_t RA[WARP_row_TILES][4];
        uint32_t RB[WARP_col_TILES][2];

#pragma unroll
        for (size_t i = 0; i < WARP_row_TILES; ++i) {
            size_t A_smem_idx = smem_store_off + (warp_id / BLOCK_col_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(k_step * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_col_TILES; ++j) {
            size_t B_smem_idx = smem_store_off + B_smem_idx_off + (warp_id % BLOCK_col_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_row_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_col_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_col_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0].u32, RC[i][j_s][1].u32, RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0], RB[j_s][1],
                          RC[i][j_s][0].u32, RC[i][j_s][1].u32);
            }
        }
    }

    __syncthreads();
    for(int i=0; i<256*128/BRANCH; i++){
        half *data = &smem[0][0];
        data[i] = -99.0;
    }
    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_row_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_col_TILES; ++j) {
            #ifdef use_branch_1
            half *lane_ptr0 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_col_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;
            half *lane_ptr1 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_col_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;

            *((uint32_t *)(lane_ptr0)) = RC[i][j][0].u32;
            *((uint32_t *)(lane_ptr1)) = RC[i][j][1].u32;
            #else
            half *lane_ptr0 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE / BRANCH +
                (((warp_id % BLOCK_col_WARPS) * C_SMEM_OFFSET + j * MMA_N)/BRANCH  +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half)/BRANCH + ((lane_id / 4) % 8) * PERMUTED_OFFSET/BRANCH) %
                    (C_SMEM_STRIDE/BRANCH);
            half *lane_ptr1 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE / BRANCH +
                (((warp_id % BLOCK_col_WARPS) * C_SMEM_OFFSET + j * MMA_N)/BRANCH +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half)/BRANCH + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET/BRANCH) %
                    (C_SMEM_STRIDE/BRANCH);
            #endif

            #ifdef use_branch_2
            *(lane_ptr0) = RC[i][j][0].h2.x + RC[i][j][0].h2.y;
            *(lane_ptr1) = RC[i][j][1].h2.x + RC[i][j][1].h2.y;
            #endif

            #ifdef use_branch_4
            half res = RC[i][j][0].h2.x + RC[i][j][0].h2.y;
            res += __shfl_down_sync(0xffffffff, res, 1, WARP_SIZE);
            if(lane_id%2==0){
                *(lane_ptr0) = res;
            }
            res = RC[i][j][1].h2.x + RC[i][j][1].h2.y;
            res += __shfl_down_sync(0xffffffff, res, 1, WARP_SIZE);
            if(threadIdx.x%2==0){
                *(lane_ptr1) = res;
            }
            #endif

            #ifdef use_branch_8
            half res = RC[i][j][0].h2.x + RC[i][j][0].h2.y;
            res += __shfl_down_sync(0xffffffff, res, 1, WARP_SIZE);
            res += __shfl_down_sync(0xffffffff, res, 2, WARP_SIZE);
            if(threadIdx.x%4==0){
                *(lane_ptr0) = res;
            }
            res = RC[i][j][1].h2.x + RC[i][j][1].h2.y;
            res += __shfl_down_sync(0xffffffff, res, 1, WARP_SIZE);
            res += __shfl_down_sync(0xffffffff, res, 2, WARP_SIZE);
            if(threadIdx.x%4==0){
                *(lane_ptr1) = res;
            }
            #endif

        }
    }

    __syncthreads();

#pragma unroll
//每个 warp 存 2行 MMA 的数据
    for (size_t i = 0; i < MMA_M; ++i) {

        #ifdef use_branch_1
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
        #endif

        #ifdef use_branch_2
        *((int2 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N/BRANCH) + lane_id % 16) =
            *((int2 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE / BRANCH) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % 16);
        #endif

        #ifdef use_branch_4
        *((int32_t *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N/BRANCH) + lane_id % 16) =
            *((int32_t *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE / BRANCH) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % 16);
        #endif

        #ifdef use_branch_8
        *((half *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N/BRANCH) + lane_id % 16) =
            *((half *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE / BRANCH) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % 16);
        #endif
    }
}

size_t initMmaAsyncPg2s() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * 2,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaAsyncPg2sKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaAsyncPg2s(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaAsyncPg2s();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaAsyncPg2sKernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}
