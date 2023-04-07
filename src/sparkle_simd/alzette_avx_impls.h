#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include "sparkle_simd.h"

#if 1 // Folding
#define ROT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define ELL(x) (ROT(((x) ^ ((x) << 16)), 16))

// Round constants
static const uint32_t RCON[MAX_BRANCHES] = { \
  0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, \
  0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D  \
};

static const uint32_t rc_256_shuffled_vals[8] = {
    0xB7E15162, 0xBF715880, 0xBB1185EB, 0x4F7C7B57,
    0x38B4DA56, 0x324E7738, 0xCFBFA1C8, 0xC2B3293D
};

static const __m128i* rc_128 = (__m128i*)&RCON;
static const __m256i* rc_256 = (__m256i*)&RCON;
static const __m256i* rc_256_shuffled = (__m256i*)&rc_256_shuffled_vals;

struct state_512 {
    union {
        __m256i vecs[2];
        uint32_t arr[16];
    };
};

struct state_256 {
    union {
        __m128i vecs[2];
        uint32_t arr[8];
    };
};
#endif

#if 1 // Folding
// Equivalent to ROT.
// For each value in `in`: Bitwise or of shift right by count, shift left by 32 - count.
__inline __m256i rot_256(__m256i in, int n) {
    return _mm256_or_si256(
        _mm256_srli_epi32(in, n),
        _mm256_slli_epi32(in, 32 - n));
}

__inline __m128i rot_128(__m128i in, int count) {
    return _mm_or_si128(
        _mm_srli_epi32(in, count),
        _mm_slli_epi32(in, 32 - count));
}

// Equivalent to + ROT.
__inline __m256i rot_add_256(__m256i left, __m256i right, int n) {
    return _mm256_add_epi32(left, rot_256(right, n));
}

__inline __m128i rot_add_128(__m128i left, __m128i right, int n) {
    return _mm_add_epi32(left, rot_128(right, n));
}

// Equivalent to ^ ROT
__inline __m256i rot_xor_256(__m256i left, __m256i right, int n) {
    return _mm256_xor_si256(left, rot_256(right, n));
}

__inline __m128i rot_xor_128(__m128i left, __m128i right, int n) {
    return _mm_xor_si128(left, rot_128(right, n));
}

#define ALZETTE_256(state_j, state_j1, rc)          \
    state_j = rot_add_256(state_j, state_j1, 31);   \
    state_j1 = rot_xor_256(state_j1, state_j, 24);  \
    state_j = _mm256_xor_si256(state_j, rc);        \
                                                    \
    state_j = rot_add_256(state_j, state_j1, 17);   \
    state_j1 = rot_xor_256(state_j1, state_j, 17);  \
    state_j = _mm256_xor_si256(state_j, rc);        \
                                                    \
    state_j = _mm256_add_epi32(state_j, state_j1);  \
    state_j1 = rot_xor_256(state_j1, state_j, 31);  \
    state_j = _mm256_xor_si256(state_j, rc);        \
                                                    \
    state_j = rot_add_256(state_j, state_j1, 24);   \
    state_j1 = rot_xor_256(state_j1, state_j, 16);  \
    state_j = _mm256_xor_si256(state_j, rc);

#define ALZETTE_128(state_j, state_j1, rc)              \
    state_j = rot_add_128(state_j, state_j1, 31);       \
    state_j1 = rot_xor_128(state_j1, state_j, 24);      \
    state_j = _mm_xor_si128(state_j, rc);               \
                                                        \
    state_j = rot_add_128(state_j, state_j1, 17);       \
    state_j1 = rot_xor_128(state_j1, state_j, 17);      \
    state_j = _mm_xor_si128(state_j, rc);               \
                                                        \
    state_j = _mm_add_epi32(state_j, state_j1);         \
    state_j1 = rot_xor_128(state_j1, state_j, 31);      \
    state_j = _mm_xor_si128(state_j, rc);               \
                                                        \
    state_j = rot_add_128(state_j, state_j1, 24);       \
    state_j1 = rot_xor_128(state_j1, state_j, 16);      \
    state_j = _mm_xor_si128(state_j, rc);

// For debugging.
void print_state(uint32_t* in, unsigned int in_len, char* label) {
    printf("%s = [ ", label);
    for (unsigned int i = 0; i < in_len; i++) {
        printf("%s%u", i == 0 ? "" : ", ", in[i]);
    }
    printf(" ]\n");
}
#endif

// Intial implementation.
// Loads alternating state values into 2 256-bit vecs.
// Permutes to get the right order before unpacking.
// Double memcpy with dynamic offset.
__inline void alzette_avx_00(uint32_t* state, int brans) {
    __m256i state_j = _mm256_i32gather_epi32((int*)state,
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sizeof(uint32_t));
    __m256i state_j1 = _mm256_i32gather_epi32((int*)state,
        _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15), sizeof(uint32_t));

    state_j = rot_add_256(state_j, state_j1, 31);
    state_j1 = rot_xor_256(state_j1, state_j, 24);
    state_j = _mm256_xor_si256(state_j, *rc_256);

    state_j = rot_add_256(state_j, state_j1, 17);
    state_j1 = rot_xor_256(state_j1, state_j, 17);
    state_j = _mm256_xor_si256(state_j, *rc_256);

    state_j = _mm256_add_epi32(state_j, state_j1);
    state_j1 = rot_xor_256(state_j1, state_j, 31);
    state_j = _mm256_xor_si256(state_j, *rc_256);

    state_j = rot_add_256(state_j, state_j1, 24);
    state_j1 = rot_xor_256(state_j1, state_j, 16);
    state_j = _mm256_xor_si256(state_j, *rc_256);

    state_j = _mm256_permute4x64_epi64(state_j, 0b11011000);
    state_j1 = _mm256_permute4x64_epi64(state_j1, 0b11011000);
    __m256i state_low = _mm256_unpacklo_epi32(state_j, state_j1);
    __m256i state_high = _mm256_unpackhi_epi32(state_j, state_j1);
    
    uint32_t* lo_ptr = (uint32_t*)&state_low;
    uint32_t* hi_ptr = (uint32_t*)&state_high;
    memcpy(state, lo_ptr, 8 * sizeof(uint32_t));
    memcpy(state + 8, hi_ptr, ((2 * brans) - 8) * sizeof(uint32_t));
}


// Loads in a shuffled order to remove the 2 permutation instructions.
__inline void alzette_avx_01(uint32_t* state, int brans) {
    __m256i state_j = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
    __m256i state_j1 = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));

    ALZETTE_256(state_j, state_j1, *rc_256_shuffled);

    __m256i state_low = _mm256_unpacklo_epi32(state_j, state_j1);
    __m256i state_high = _mm256_unpackhi_epi32(state_j, state_j1);

    uint32_t* lo_ptr = (uint32_t*)&state_low;
    uint32_t* hi_ptr = (uint32_t*)&state_high;
    memcpy(state, lo_ptr, 8 * sizeof(uint32_t));
    memcpy(state + 8, hi_ptr, ((2 * brans) - 8) * sizeof(uint32_t));
}


// Proof of concept for avoiding the gather instructions.
// Sequential load + double unpacklo/unpackhi gets it in the same order as gather.
// Seems to be much slower (~ -4%).
// Each unpack should only have latency of 1.
// Changing the load between _mm256_lddqu_si256, _mm256_loadu_si256, _mm256_loadu_epi32
// doesn't have a noticeable effect.
//
// TODO:
// Try messing with state alignment?
__inline void alzette_avx_02(uint32_t* state, int brans) {
    __m256i state_j = _mm256_lddqu_si256((__m256i*)(state));
    __m256i state_j1 = _mm256_lddqu_si256((__m256i*)(state + 8));

    {
        __m256i temp_j = _mm256_unpacklo_epi32(state_j, state_j1);
        __m256i temp_j1 = _mm256_unpackhi_epi32(state_j, state_j1);

        state_j = _mm256_unpacklo_epi32(temp_j, temp_j1);
        state_j1 = _mm256_unpackhi_epi32(temp_j, temp_j1);
    }

    ALZETTE_256(state_j, state_j1, *rc_256_shuffled);

    __m256i state_low = _mm256_unpacklo_epi32(state_j, state_j1);
    __m256i state_high = _mm256_unpackhi_epi32(state_j, state_j1);

    uint32_t* lo_ptr = (uint32_t*)&state_low;
    uint32_t* hi_ptr = (uint32_t*)&state_high;
    memcpy(state, lo_ptr, 8 * sizeof(uint32_t));
    memcpy(state + 8, hi_ptr, ((2 * brans) - 8) * sizeof(uint32_t));
}


// Current fastest implementation.
// Possible optimizations:
// - Is it faster to load sequentially, then shuffle?
// - Is it faster to make the state a union to only need 1 memcpy?
__inline void alzette_avx_03(uint32_t* state, int brans) {
    if (brans == 8) {
        // Results in the following:
        // state    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        // state_j  = [0, 2, 8, 10, 4, 6, 12, 14]
        // state_j1 = [1, 3, 9, 11, 5, 7, 13, 15]
        __m256i state_j = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
        __m256i state_j1 = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));

        ALZETTE_256(state_j, state_j1, *rc_256_shuffled);

        // Unpack and de-interleave the data from the registers. Results in:
        // state_lo = [0, 1, 2, 3, 4, 5, 6, 7]
        // state_hi = [8, 9, 10, 11, 12, 13, 14, 15]
        // This is then copied back to `state` using memcpy.
        // A different length is required for the second memcpy, and depends on `brans`.
        // e.g. for Sparkle384, brans = 6. Therefore there are 12 32-bit elements. The first
        // memcpy operation copies the first 8, so the latter must copy (2 * 6) - 8 = 4 elements.
        __m256i state_lo = _mm256_unpacklo_epi32(state_j, state_j1);
        __m256i state_hi = _mm256_unpackhi_epi32(state_j, state_j1);

        // TODO: Is a SIMD store instruction here faster than double memcpy?
        uint32_t* lo_ptr = (uint32_t*)&state_lo;
        uint32_t* hi_ptr = (uint32_t*)&state_hi;
        memcpy(state, lo_ptr, 32);
        memcpy(state + 8, hi_ptr, 32);
    }
    else {
        __m128i state_j = _mm_i32gather_epi32(state, _mm_setr_epi32(0, 2, 4, 6), sizeof(uint32_t));
        __m128i state_j1 = _mm_i32gather_epi32(state, _mm_setr_epi32(1, 3, 5, 7), sizeof(uint32_t));

        ALZETTE_128(state_j, state_j1, *rc_128);

        __m128i state_lo = _mm_unpacklo_epi32(state_j, state_j1);
        __m128i state_hi = _mm_unpackhi_epi32(state_j, state_j1);

        // Try _mm256_set_m128(state_lo, state_hi) + a single memcpy here!
        memcpy(state, (uint32_t*)&state_lo, 16);
        memcpy(state + 4, (uint32_t*)&state_hi, 16);

        if (brans == 6) {
            uint32_t rc = RCON[8 >> 1];
            state[8] += ROT(state[9], 31);
            state[9] ^= ROT(state[8], 24);
            state[8] ^= rc;
            state[8] += ROT(state[9], 17);
            state[9] ^= ROT(state[8], 17);
            state[8] ^= rc;
            state[8] += state[9];
            state[9] ^= ROT(state[8], 31);
            state[8] ^= rc;
            state[8] += ROT(state[9], 24);
            state[9] ^= ROT(state[8], 16);
            state[8] ^= rc;

            rc = RCON[10 >> 1];
            state[10] += ROT(state[11], 31);
            state[11] ^= ROT(state[10], 24);
            state[10] ^= rc;
            state[10] += ROT(state[11], 17);
            state[11] ^= ROT(state[10], 17);
            state[10] ^= rc;
            state[10] += state[11];
            state[11] ^= ROT(state[10], 31);
            state[10] ^= rc;
            state[10] += ROT(state[11], 24);
            state[11] ^= ROT(state[10], 16);
            state[10] ^= rc;
        }
    }
}


// Proof of concept for using unions to only need a single 512/256 bit memcpy, instead
// of a memcpy for each vector register.
// Actually seems slower than the double memcpy, at least for sparkle512.
__inline void alzette_avx_04(uint32_t* state, int brans) {
    if (brans == 8) {
        struct state_512 simd_state;

        simd_state.vecs[0] = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
        simd_state.vecs[1] = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));

        ALZETTE_256(simd_state.vecs[0], simd_state.vecs[1], *rc_256_shuffled);

        {
            __m256i temp_j = simd_state.vecs[0];
            simd_state.vecs[0] = _mm256_unpacklo_epi32(temp_j, simd_state.vecs[1]);
            simd_state.vecs[1] = _mm256_unpackhi_epi32(temp_j, simd_state.vecs[1]);
        }

        memcpy(state, &(simd_state.arr), 64);
    }
    else {
        struct state_256 simd_state;

        simd_state.vecs[0] = _mm_i32gather_epi32(state, _mm_setr_epi32(0, 2, 4, 6), sizeof(uint32_t));
        simd_state.vecs[1] = _mm_i32gather_epi32(state, _mm_setr_epi32(1, 3, 5, 7), sizeof(uint32_t));

        ALZETTE_128(simd_state.vecs[0], simd_state.vecs[1], *rc_128);
        
        {
            __m128i temp = simd_state.vecs[0];
            simd_state.vecs[0] = _mm_unpacklo_epi32(temp, simd_state.vecs[1]);
            simd_state.vecs[1] = _mm_unpackhi_epi32(temp, simd_state.vecs[1]);
        }

        memcpy(state, &(simd_state.arr), 32);

        if (brans == 6) {
            uint32_t rc = RCON[8 >> 1];
            state[8] += ROT(state[9], 31);
            state[9] ^= ROT(state[8], 24);
            state[8] ^= rc;
            state[8] += ROT(state[9], 17);
            state[9] ^= ROT(state[8], 17);
            state[8] ^= rc;
            state[8] += state[9];
            state[9] ^= ROT(state[8], 31);
            state[8] ^= rc;
            state[8] += ROT(state[9], 24);
            state[9] ^= ROT(state[8], 16);
            state[8] ^= rc;

            rc = RCON[10 >> 1];
            state[10] += ROT(state[11], 31);
            state[11] ^= ROT(state[10], 24);
            state[10] ^= rc;
            state[10] += ROT(state[11], 17);
            state[11] ^= ROT(state[10], 17);
            state[10] ^= rc;
            state[10] += state[11];
            state[11] ^= ROT(state[10], 31);
            state[10] ^= rc;
            state[10] += ROT(state[11], 24);
            state[11] ^= ROT(state[10], 16);
            state[10] ^= rc;
        }
    }
}


// Proof of concept using SIMD store instructions instead of memcpy.
// Appears to be slower than memcpy, at least for sparkle512.
// No noticeable difference between using _mm256_storeu_epi32 and _mm256_storeu_si256.
//
// Possibly because we only memcpy 32 bytes at a time (small data)?
__inline void alzette_avx_05(uint32_t* state, int brans) {
    if (brans == 8) {
        // Results in the following:
        // state    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        // state_j  = [0, 2, 8, 10, 4, 6, 12, 14]
        // state_j1 = [1, 3, 9, 11, 5, 7, 13, 15]
        __m256i state_j = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
        __m256i state_j1 = _mm256_i32gather_epi32(state,
            _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));

        ALZETTE_256(state_j, state_j1, *rc_256_shuffled);

        // Unpack and de-interleave the data from the registers. Results in:
        // state_lo = [0, 1, 2, 3, 4, 5, 6, 7]
        // state_hi = [8, 9, 10, 11, 12, 13, 14, 15]
        // This is then copied back to `state` using memcpy.
        // A different length is required for the second memcpy, and depends on `brans`.
        // e.g. for Sparkle384, brans = 6. Therefore there are 12 32-bit elements. The first
        // memcpy operation copies the first 8, so the latter must copy (2 * 6) - 8 = 4 elements.
        __m256i state_lo = _mm256_unpacklo_epi32(state_j, state_j1);
        __m256i state_hi = _mm256_unpackhi_epi32(state_j, state_j1);

        _mm256_storeu_epi32(state, state_lo);
        _mm256_storeu_epi32(state + 8, state_hi);
    }
    else {
        __m128i state_j = _mm_i32gather_epi32(state, _mm_setr_epi32(0, 2, 4, 6), sizeof(uint32_t));
        __m128i state_j1 = _mm_i32gather_epi32(state, _mm_setr_epi32(1, 3, 5, 7), sizeof(uint32_t));

        ALZETTE_128(state_j, state_j1, *rc_128);

        __m128i state_lo = _mm_unpacklo_epi32(state_j, state_j1);
        __m128i state_hi = _mm_unpackhi_epi32(state_j, state_j1);

        _mm_storeu_epi32(state, state_lo);
        _mm_storeu_epi32(state + 4, state_hi);

        if (brans == 6) {
            uint32_t rc = RCON[8 >> 1];
            state[8] += ROT(state[9], 31);
            state[9] ^= ROT(state[8], 24);
            state[8] ^= rc;
            state[8] += ROT(state[9], 17);
            state[9] ^= ROT(state[8], 17);
            state[8] ^= rc;
            state[8] += state[9];
            state[9] ^= ROT(state[8], 31);
            state[8] ^= rc;
            state[8] += ROT(state[9], 24);
            state[9] ^= ROT(state[8], 16);
            state[8] ^= rc;

            rc = RCON[10 >> 1];
            state[10] += ROT(state[11], 31);
            state[11] ^= ROT(state[10], 24);
            state[10] ^= rc;
            state[10] += ROT(state[11], 17);
            state[11] ^= ROT(state[10], 17);
            state[10] ^= rc;
            state[10] += state[11];
            state[11] ^= ROT(state[10], 31);
            state[10] ^= rc;
            state[10] += ROT(state[11], 24);
            state[11] ^= ROT(state[10], 16);
            state[10] ^= rc;
        }
    }
}


// Define these as macros to allow changing to match the indices when doing a shuffled
// load with gather.
#define INDEX_0 0
#define INDEX_1 8
#define INDEX_2 1
#define INDEX_3 9
#define INDEX_4 4
#define INDEX_5 12
#define INDEX_6 5
#define INDEX_7 13
#define INDEX_8 2
#define INDEX_9 10
#define INDEX_10 3
#define INDEX_11 11
#define INDEX_12 6
#define INDEX_13 14
#define INDEX_14 7
#define INDEX_15 15

// An unrolled version of the linear layer for 4, 6, or 8 branches.
// Uses the macros for indices, so that it works no matter how the state is shuffled.
__inline void linear_layer_unrolled(uint32_t* state, int brans, uint32_t tmpx, uint32_t tmpy, uint32_t x0, uint32_t y0) {
    tmpx = x0 = state[INDEX_0];
    tmpy = y0 = state[INDEX_1];

    if (brans >= 4) {
        tmpx ^= state[INDEX_2];
        tmpy ^= state[INDEX_3];
    }

    if (brans >= 6) {
        tmpx ^= state[INDEX_4];
        tmpy ^= state[INDEX_5];
    }

    if (brans == 8) {
        tmpx ^= state[INDEX_6];
        tmpy ^= state[INDEX_7];
    }

    tmpx = ELL(tmpx);
    tmpy = ELL(tmpy);

    if (brans == 4) {
        // j = 2
        state[INDEX_0] = state[INDEX_6] ^ state[INDEX_2] ^ tmpy;
        state[INDEX_6] = state[INDEX_2];
        state[INDEX_1] = state[INDEX_7] ^ state[INDEX_3] ^ tmpx;
        state[INDEX_7] = state[INDEX_3];

        state[INDEX_2] = state[INDEX_4] ^ x0 ^ tmpy;
        state[INDEX_4] = x0;
        state[INDEX_3] = state[INDEX_5] ^ y0 ^ tmpx;
        state[INDEX_5] = y0;
    } else if (brans == 6) {
        // j = 2
        state[INDEX_0] = state[INDEX_8] ^ state[INDEX_2] ^ tmpy;
        state[INDEX_8] = state[INDEX_2];
        state[INDEX_1] = state[INDEX_9] ^ state[INDEX_3] ^ tmpx;
        state[INDEX_9] = state[INDEX_3];

        // j = 4
        state[INDEX_2] = state[INDEX_10] ^ state[INDEX_4] ^ tmpy;
        state[INDEX_10] = state[INDEX_4];
        state[INDEX_3] = state[INDEX_11] ^ state[INDEX_5] ^ tmpx;
        state[INDEX_11] = state[INDEX_5];

        state[INDEX_4] = state[INDEX_6] ^ x0 ^ tmpy;
        state[INDEX_6] = x0;
        state[INDEX_5] = state[INDEX_7] ^ y0 ^ tmpx;
        state[INDEX_7] = y0;
    } else if (brans == 8) {
        // j = 2
        state[INDEX_0] = state[INDEX_10] ^ state[INDEX_2] ^ tmpy;
        state[INDEX_10] = state[INDEX_2];
        state[INDEX_1] = state[INDEX_11] ^ state[INDEX_3] ^ tmpx;
        state[INDEX_11] = state[INDEX_3];

        // j = 4
        state[INDEX_2] = state[INDEX_12] ^ state[INDEX_4] ^ tmpy;
        state[INDEX_12] = state[INDEX_4];
        state[INDEX_3] = state[INDEX_13] ^ state[INDEX_5] ^ tmpx;
        state[INDEX_13] = state[INDEX_5];

        // j = 6
        state[INDEX_4] = state[INDEX_14] ^ state[INDEX_6] ^ tmpy;
        state[INDEX_14] = state[INDEX_6];
        state[INDEX_5] = state[INDEX_15] ^ state[INDEX_7] ^ tmpx;
        state[INDEX_15] = state[INDEX_7];

        state[INDEX_6] = state[INDEX_8] ^ x0 ^ tmpy;
        state[INDEX_8] = x0;
        state[INDEX_7] = state[INDEX_9] ^ y0 ^ tmpx;
        state[INDEX_9] = y0;
    }
}


// Original version of the linear layer, without loop unrolling or modified indices.
__inline void linear_layer_rolled(uint32_t* state, int brans, uint32_t tmpx, uint32_t tmpy, uint32_t x0, uint32_t y0) {
    tmpx = x0 = state[0];
    tmpy = y0 = state[1];
    for (unsigned int j = 2; j < brans; j += 2) {
        tmpx ^= state[j];
        tmpy ^= state[j + 1];
    }
    tmpx = ELL(tmpx);
    tmpy = ELL(tmpy);
    for (unsigned int j = 2; j < brans; j += 2) {
        state[j - 2] = state[j + brans] ^ state[j] ^ tmpy;
        state[j + brans] = state[j];
        state[j - 1] = state[j + brans + 1] ^ state[j + 1] ^ tmpx;
        state[j + brans + 1] = state[j + 1];
    }
    state[brans - 2] = state[brans] ^ x0 ^ tmpy;
    state[brans] = x0;
    state[brans - 1] = state[brans + 1] ^ y0 ^ tmpx;
    state[brans + 1] = y0;
}


// Proof of concept for using the unrolled linear layer. Only needs to load to registers
// once, then performs all steps while keeping state in the registers for the linear layer.
// Only works on 8 brans as it's just a test.
//
// This turns out to be much slower than the reference version, as accessing the items in
// the SIMD registers through the union causes (presumably) a lot of reading/writing from
// the registers to main memory and back again. This seems to massively outweigh the benefit
// from not having to load/store at every single step.
__inline void sparkle_unrolled_test_00(uint32_t* state, int brans, int steps) {
    if (brans != 8) return;

    uint32_t rc, tmpx, tmpy, x0, y0;
    
    struct state_512 simd_state;

    simd_state.vecs[0] = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
    simd_state.vecs[1] = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));

    for (unsigned int i = 0; i < steps; i++) {
        simd_state.arr[INDEX_1] ^= RCON[i % MAX_BRANCHES];
        simd_state.arr[INDEX_3] ^= i;

        ALZETTE_256(simd_state.vecs[0], simd_state.vecs[1], *rc_256_shuffled);

        linear_layer_unrolled(&(simd_state.arr), brans, tmpx, tmpy, x0, y0);
    }

    {
        __m256i temp_j = simd_state.vecs[0];
        simd_state.vecs[0] = _mm256_unpacklo_epi32(temp_j, simd_state.vecs[1]);
        simd_state.vecs[1] = _mm256_unpackhi_epi32(temp_j, simd_state.vecs[1]);
    }

    memcpy(state, &(simd_state.arr), 64);
}


// Similar to previous, but instead of trying to perform the entire linear layer in the
// SIMD registers, instead write back linearly after Alzette. The idea is that the
// gather and unpack extra instructions are now only needed in the first and last step
// respectively, and everywhere else can just use a linear load/store.
//
// This does turn out to be faster than reference, but is still approx. 10% behind the
// rolled version (alzette_avx_03). Unsure why this is slower than that version, as it
// should in theory be requiring only a single set of unpack instructions per call instead
// of a set per step. Perhaps the shuffled indices in the linear layer cause more cache
// misses, or are worse for fetching. This needs more investigation.
__inline void sparkle_unrolled_test_01(uint32_t* state, int brans, int steps) {
    if (brans != 8) return;

    uint32_t rc, tmpx, tmpy, x0, y0;

    for (unsigned int i = 0; i < steps; i++) {
        __m256i state_j, state_j1;

        // Only needs a gathered load for the first step, after that linear is fine.
        if (i == 0) {
            // We haven't shuffled the state yet, so use normal indices.
            state[1] ^= RCON[i % MAX_BRANCHES];
            state[3] ^= i;

            state_j = _mm256_i32gather_epi32(state,
                _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
            state_j1 = _mm256_i32gather_epi32(state,
                _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));
        } else {
            state[INDEX_1] ^= RCON[i % MAX_BRANCHES];
            state[INDEX_3] ^= i;

            state_j = _mm256_loadu_epi32(state);
            state_j1 = _mm256_loadu_epi32(state + 8);
        }

        ALZETTE_256(state_j, state_j1, *rc_256_shuffled);

        // If this is the last step, unpack before storing and use the original indices
        // for the linear layer. If not, store without unpacking and use the shuffled
        // indices for the linear layer.
        if (i == steps - 1) {
            __m256i state_lo = _mm256_unpacklo_epi32(state_j, state_j1);
            __m256i state_hi = _mm256_unpackhi_epi32(state_j, state_j1);

            uint32_t* lo_ptr = (uint32_t*)&state_lo;
            uint32_t* hi_ptr = (uint32_t*)&state_hi;
            memcpy(state, lo_ptr, 32);
            memcpy(state + 8, hi_ptr, 32);

            linear_layer_rolled(state, brans, tmpx, tmpy, x0, y0);
        } else {
            uint32_t* lo_ptr = (uint32_t*)&state_j;
            uint32_t* hi_ptr = (uint32_t*)&state_j1;
            memcpy(state, lo_ptr, 32);
            memcpy(state + 8, hi_ptr, 32);

            linear_layer_unrolled(state, brans, tmpx, tmpy, x0, y0);
        }
    }
}