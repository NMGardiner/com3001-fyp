///////////////////////////////////////////////////////////////////////////////
// sparkle_opt.c: Optimized C99 implementation of the SPARKLE permutation.   //
// This file is part of the SPARKLE submission to NIST's LW Crypto Project.  //
// Version 1.1.2 (2020-10-30), see <http://www.cryptolux.org/> for updates.  //
// Authors: The SPARKLE Group (C. Beierle, A. Biryukov, L. Cardoso dos       //
// Santos, J. Groszschaedl, L. Perrin, A. Udovenko, V. Velichkov, Q. Wang).  //
// License: GPLv3 (see LICENSE file), other licenses available upon request. //
// Copyright (C) 2019-2020 University of Luxembourg <http://www.uni.lu/>.    //
// ------------------------------------------------------------------------- //
// This program is free software: you can redistribute it and/or modify it   //
// under the terms of the GNU General Public License as published by the     //
// Free Software Foundation, either version 3 of the License, or (at your    //
// option) any later version. This program is distributed in the hope that   //
// it will be useful, but WITHOUT ANY WARRANTY; without even the implied     //
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  //
// GNU General Public License for more details. You should have received a   //
// copy of the GNU General Public License along with this program. If not,   //
// see <http://www.gnu.org/licenses/>.                                       //
///////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include "sparkle_simd.h"

#define ENABLE_SIMD 1
// Test if gathering elements with a set stride (e.g. 0, 2, 4, 6...) is any faster
// than gathering elements 'at random'. Timing suggests it's actually slower due to
// needing additional permute instructions to store, but look into this more!
#define SHUFFLE_STATE 1 

#if ENABLE_SIMD
#include <immintrin.h>
#endif // ENABLE_SIMD


#define ROT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define ELL(x) (ROT(((x) ^ ((x) << 16)), 16))


// Round constants
static const uint32_t RCON[MAX_BRANCHES] = {      \
  0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, \
  0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D  \
};

#if ENABLE_SIMD
// Equivalent to ROT.
// For each value in `in`: Bitwise or of shift right by count, shift left by 32 - count.
__inline __m256i rot_simd(__m256i in, int count) {
    return _mm256_or_si256(
        _mm256_srlv_epi32(in, _mm256_set1_epi32(count)),
        _mm256_sllv_epi32(in, _mm256_set1_epi32(32 - count)));
}

// Equivalent to + ROT.
__inline __m256i rot_add_simd(__m256i left, __m256i right, int count) {
    return _mm256_add_epi32(left, rot_simd(right, count));
}

// Equivalent to ^ ROT
__inline __m256i rot_xor_simd(__m256i left, __m256i right, int count) {
    return _mm256_xor_si256(left, rot_simd(right, count));
}

// For debugging.
void print_m256i(__m256i in, unsigned int in_len, char* label) {
    uint32_t* ptr = (uint32_t*)&in;
    printf("%s = [ ", label);
    for (int i = 0; i < in_len; i++) {
        printf("%s%u", i == 0 ? "" : ", ", ptr[i]);
    }
    printf(" ]\n");
}

// For debugging.
void print_state(uint32_t* in, unsigned int in_len,char* label) {
    printf("%s = [ ", label);
    for (int i = 0; i < in_len; i++) {
        printf("%s%u", i == 0 ? "" : ", ", in[i]);
    }
    printf(" ]\n");
}
#endif // ENABLE_SIMD

void sparkle_simd(uint32_t *state, int brans, int steps)
{
  int i, j;  // Step and branch counter
  uint32_t rc, tmpx, tmpy, x0, y0;

#if ENABLE_SIMD
#if SHUFFLE_STATE
  // RCON, but rearranged to match the order of pairs after loading them
  // into the vector registers.
  const __m256i round_constants = _mm256_setr_epi32(
      0xB7E15162, 0xBF715880, 0xBB1185EB, 0x4F7C7B57,
      0x38B4DA56, 0x324E7738, 0xCFBFA1C8, 0xC2B3293D);
#else
  const __m256i round_constants = _mm256_setr_epi32(
      0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738,
      0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D);
#endif // SHUFFLE_STATE
#endif // ENABLE_SIMD

  for(i = 0; i < steps; i ++) {
    // Add round constant
    state[1] ^= RCON[i%MAX_BRANCHES];
    state[3] ^= i;

    // ARXBOX layer
#if ENABLE_SIMD
#if SHUFFLE_STATE
    // Results in the following:
    // state    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    // state_j  = [0, 2, 8, 10, 4, 6, 12, 14]
    // state_j1 = [1, 3, 9, 11, 5, 7, 13, 15]
    __m256i state_j = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(0, 2, 8, 10, 4, 6, 12, 14), sizeof(uint32_t));
    __m256i state_j1 = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(1, 3, 9, 11, 5, 7, 13, 15), sizeof(uint32_t));
#else
    __m256i state_j = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14), sizeof(uint32_t));
    __m256i state_j1 = _mm256_i32gather_epi32(state,
        _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15), sizeof(uint32_t));
#endif // SHUFFLE_STATE

    // Equivalent to:
    // state[j] += ROT(state[j + 1], 31);
    // state[j + 1] ^= ROT(state[j], 24);
    // state[j] ^= rc;
    state_j = rot_add_simd(state_j, state_j1, 31);
    state_j1 = rot_xor_simd(state_j1, state_j, 24);
    state_j = _mm256_xor_si256(state_j, round_constants);

    // Equivalent to:
    // state[j] += ROT(state[j + 1], 17);
    // state[j + 1] ^= ROT(state[j], 17);
    // state[j] ^= rc;
    state_j = rot_add_simd(state_j, state_j1, 17);
    state_j1 = rot_xor_simd(state_j1, state_j, 17);
    state_j = _mm256_xor_si256(state_j, round_constants);

    // Equivalent to:
    // state[j] += state[j + 1];
    // state[j + 1] ^= ROT(state[j], 31);
    // state[j] ^= rc;
    state_j = _mm256_add_epi32(state_j, state_j1);
    state_j1 = rot_xor_simd(state_j1, state_j, 31);
    state_j = _mm256_xor_si256(state_j, round_constants);

    // Equivalent to:
    // state[j] += ROT(state[j + 1], 24);
    // state[j + 1] ^= ROT(state[j], 16);
    // state[j] ^= rc;
    state_j = rot_add_simd(state_j, state_j1, 24);
    state_j1 = rot_xor_simd(state_j1, state_j, 16);
    state_j = _mm256_xor_si256(state_j, round_constants);

    // Unpack and de-interleave the data from the registers. Results in:
    // state_lo = [0, 1, 2, 3, 4, 5, 6, 7]
    // state_hi = [8, 9, 10, 11, 12, 13, 14, 15]
    // This is then copied back to `state` using memcpy.
    // A different length is required for the second memcpy, and depends on `brans`.
    // e.g. for Sparkle384, brans = 6. Therefore there are 12 32-bit elements. The first
    // memcpy operation copies the first 8, so the latter must copy (2 * 6) - 8 = 4 elements.
#if SHUFFLE_STATE
    __m256i state_lo = _mm256_unpacklo_epi32(state_j, state_j1);
    __m256i state_hi = _mm256_unpackhi_epi32(state_j, state_j1);
#else
    __m256i state_j_perm = _mm256_permute4x64_epi64(state_j, 0b11011000);
    __m256i state_j1_perm = _mm256_permute4x64_epi64(state_j1, 0b11011000);

    __m256i state_lo = _mm256_unpacklo_epi32(state_j_perm, state_j1_perm);
    __m256i state_hi = _mm256_unpackhi_epi32(state_j_perm, state_j1_perm);
#endif // SHUFFLE_STATE

    // TODO: Is a SIMD store instruction here faster than double memcpy?
    uint32_t* lo_ptr = (uint32_t*)&state_lo;
    uint32_t* hi_ptr = (uint32_t*)&state_hi;
    memcpy(state, lo_ptr, 8 * sizeof(uint32_t));
    memcpy(state + 8, hi_ptr, ((2 * brans) - 8) * sizeof(uint32_t));
#else
    for (j = 0; j < 2 * brans; j += 2) {
        rc = RCON[j >> 1];
        state[j] += ROT(state[j + 1], 31);
        state[j + 1] ^= ROT(state[j], 24);
        state[j] ^= rc;
        state[j] += ROT(state[j + 1], 17);
        state[j + 1] ^= ROT(state[j], 17);
        state[j] ^= rc;
        state[j] += state[j + 1];
        state[j + 1] ^= ROT(state[j], 31);
        state[j] ^= rc;
        state[j] += ROT(state[j + 1], 24);
        state[j + 1] ^= ROT(state[j], 16);
        state[j] ^= rc;
    }
#endif // ENABLE_SIMD
    // Linear layer
    tmpx = x0 = state[0];
    tmpy = y0 = state[1];
    for(j = 2; j < brans; j += 2) {
      tmpx ^= state[j];
      tmpy ^= state[j+1];
    }
    tmpx = ELL(tmpx);
    tmpy = ELL(tmpy);
    for (j = 2; j < brans; j += 2) {
      state[j-2] = state[j+brans] ^ state[j] ^ tmpy;
      state[j+brans] = state[j];
      state[j-1] = state[j+brans+1] ^ state[j+1] ^ tmpx;
      state[j+brans+1] = state[j+1];
    }
    state[brans-2] = state[brans] ^ x0 ^ tmpy;
    state[brans] = x0;
    state[brans-1] = state[brans+1] ^ y0 ^ tmpx;
    state[brans+1] = y0;
  }
}


void sparkle_inv_simd(uint32_t *state, int brans, int steps)
{
  int i, j;  // Step and branch counter
  uint32_t rc, tmpx, tmpy, xb1, yb1;
  
  for(i = steps-1; i >= 0; i --) {
    // Linear layer
    tmpx = tmpy = 0;
    xb1 = state[brans-2];
    yb1 = state[brans-1];
    for (j = brans-2; j > 0; j -= 2) {
      tmpx ^= (state[j] = state[j+brans]);
      state[j+brans] = state[j-2];
      tmpy ^= (state[j+1] = state[j+brans+1]);
      state[j+brans+1] = state[j-1];
    }
    tmpx ^= (state[0] = state[brans]);
    state[brans] = xb1;
    tmpy ^= (state[1] = state[brans+1]);
    state[brans+1] = yb1;
    tmpx = ELL(tmpx);
    tmpy = ELL(tmpy);
    for(j = brans-2; j >= 0; j -= 2) {
      state[j+brans] ^= (tmpy ^ state[j]);
      state[j+brans+1] ^= (tmpx ^ state[j+1]);
    }
    // ARXBOX layer
    for(j = 0; j < 2*brans; j += 2) {
      rc = RCON[j>>1];
      state[j] ^= rc;
      state[j+1] ^= ROT(state[j], 16);
      state[j] -= ROT(state[j+1], 24);
      state[j] ^= rc;
      state[j+1] ^= ROT(state[j], 31);
      state[j] -= state[j+1];
      state[j] ^= rc;
      state[j+1] ^= ROT(state[j], 17);
      state[j] -= ROT(state[j+1], 17);
      state[j] ^= rc;
      state[j+1] ^= ROT(state[j], 24);
      state[j] -= ROT(state[j+1], 31);
    }
    // Add round constant
    state[1] ^= RCON[i%MAX_BRANCHES];
    state[3] ^= i;
  }
}


void clear_state_simd(uint32_t *state, int brans)
{
  int i;
  
  for (i = 0; i < 2*brans; i ++)
    state[i] = 0;
}


void print_state_simd(const uint32_t *state, int brans)
{
  uint8_t *sbytes = (uint8_t *) state;
  int i, j;
  
  for (i = 0; i < brans; i ++) {
    j = 8*i;
    printf("(%02x%02x%02x%02x %02x%02x%02x%02x)",       \
    sbytes[j],   sbytes[j+1], sbytes[j+2], sbytes[j+3], \
    sbytes[j+4], sbytes[j+5], sbytes[j+6], sbytes[j+7]);
    if (i < brans-1) printf(" ");
  }
  printf("\n");
}


void test_sparkle_simd(int brans, int steps)
{
  uint32_t state[2*MAX_BRANCHES] = { 0 };
  
  printf("input:\n");
  print_state_simd(state, brans);
  sparkle_simd(state, brans, steps);
  printf("sparkle:\n");
  print_state_simd(state, brans);
  sparkle_inv_simd(state, brans, steps);
  printf("sparkle inv:\n");
  print_state_simd(state, brans);
  printf("\n");
}
