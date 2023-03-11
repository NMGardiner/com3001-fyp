#include <immintrin.h>

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

static const __m128i* rc_128 = &RCON;
static const __m256i* rc_256 = &RCON;
static const __m256i* rc_256_shuffled = &rc_256_shuffled_vals;

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
__inline __m256i rot_256(__m256i in, int count) {
    return _mm256_or_si256(
        _mm256_srli_epi32(in, count),
        _mm256_slli_epi32(in, 32 - count));
}

__inline __m128i rot_128(__m128i in, int count) {
    return _mm_or_epi32(
        _mm_srli_epi32(in, count),
        _mm_slli_epi32(in, 32 - count));
}

// Equivalent to + ROT.
__inline __m256i rot_add_256(__m256i left, __m256i right, int count) {
    return _mm256_add_epi32(left, rot_256(right, count));
}

__inline __m128i rot_add_128(__m128i left, __m128i right, int count) {
    return _mm_add_epi32(left, rot_128(right, count));
}

// Equivalent to ^ ROT
__inline __m256i rot_xor_256(__m256i left, __m256i right, int count) {
    return _mm256_xor_si256(left, rot_256(right, count));
}

__inline __m128i rot_xor_128(__m128i left, __m128i right, int count) {
    return _mm_xor_epi32(left, rot_128(right, count));
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
    state_j = _mm_xor_epi32(state_j, rc);               \
                                                        \
    state_j = rot_add_128(state_j, state_j1, 17);       \
    state_j1 = rot_xor_128(state_j1, state_j, 17);      \
    state_j = _mm_xor_epi32(state_j, rc);               \
                                                        \
    state_j = _mm_add_epi32(state_j, state_j1);         \
    state_j1 = rot_xor_128(state_j1, state_j, 31);      \
    state_j = _mm_xor_epi32(state_j, rc);               \
                                                        \
    state_j = rot_add_128(state_j, state_j1, 24);       \
    state_j1 = rot_xor_128(state_j1, state_j, 16);      \
    state_j = _mm_xor_epi32(state_j, rc);

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
    __m256i state_j = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sizeof(uint32_t));
    __m256i state_j1 = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15), sizeof(uint32_t));

    ALZETTE_256(state_j, state_j1, *rc_256);

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
