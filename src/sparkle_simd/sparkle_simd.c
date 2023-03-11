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

#include "platform_defines.h"
#include <string.h>

// Test if gathering elements with a set stride (e.g. 0, 2, 4, 6...) is any faster
// than gathering elements 'at random'. Timing suggests it's actually slower due to
// needing additional permute instructions to store, but look into this more!
#define SHUFFLE_STATE 1 

#if USE_AVX2
#include "sparkle_simd/alzette_avx_impls.h"
#elif USE_NEON
#include <arm_neon.h>
#endif

#define ROT(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define ELL(x) (ROT(((x) ^ ((x) << 16)), 16))


#if !(USE_AVX2 || USE_NEON)
// Round constants
static const uint32_t RCON[MAX_BRANCHES] = {      \
  0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738, \
  0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D  \
};
#endif // !(USE_AVX2 || USE_NEON)

#if USE_AVX2

#elif USE_NEON
// 128-bit registers, so split into 2.
static const uint32x4x2_t round_constants = {{
  { 0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738 },
  { 0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D }
}};
#endif

#if USE_AVX2

#elif USE_NEON
__inline uint32x4_t rot_simd(uint32x4_t in, int count) {
  return vorrq_u32(
    vshrq_n_u32(in, count),
    vshlq_n_u32(in, 32 - count));
}

__inline uint32x4_t rot_add_simd(uint32x4_t left, uint32x4_t right, int count) {
  return vaddq_u32(left, rot_simd(right, count));
}

__inline uint32x4_t rot_xor_simd(uint32x4_t left, uint32x4_t right, int count) {
  return veorq_u32(left, rot_simd(right, count));
}

// Process 4 'pairs' of the state using NEON instructions, and memcpy back.
__inline static void alzette_simd(uint32_t* state, unsigned int rc) {
  uint32x4x2_t state_simd = vld2q_u32(state);

  state_simd.val[0] = rot_add_simd(state_simd.val[0], state_simd.val[1], 31);
  state_simd.val[1] = rot_xor_simd(state_simd.val[1], state_simd.val[0], 24);
  state_simd.val[0] = veorq_u32(state_simd.val[0], round_constants.val[rc]);
  
  state_simd.val[0] = rot_add_simd(state_simd.val[0], state_simd.val[1], 17);
  state_simd.val[1] = rot_xor_simd(state_simd.val[1], state_simd.val[0], 17);
  state_simd.val[0] = veorq_u32(state_simd.val[0], round_constants.val[rc]);
  
  state_simd.val[0] = vaddq_u32(state_simd.val[0], state_simd.val[1]);
  state_simd.val[1] = rot_xor_simd(state_simd.val[1], state_simd.val[0], 31);
  state_simd.val[0] = veorq_u32(state_simd.val[0], round_constants.val[rc]);

  state_simd.val[0] = rot_add_simd(state_simd.val[0], state_simd.val[1], 24);
  state_simd.val[1] = rot_xor_simd(state_simd.val[1], state_simd.val[0], 16);
  state_simd.val[0] = veorq_u32(state_simd.val[0], round_constants.val[rc]);

  uint32x4x2_t state_zip = vzipq_u32(state_simd.val[0], state_simd.val[1]);
  memcpy(state, &state_zip, 32);
}
#endif

void sparkle_simd(uint32_t *state, int brans, int steps)
{
  int i, j;  // Step and branch counter
  uint32_t rc, tmpx, tmpy, x0, y0;

  for(i = 0; i < steps; i ++) {
    // Add round constant
    state[1] ^= RCON[i%MAX_BRANCHES];
    state[3] ^= i;

    // ARXBOX layer
#if USE_AVX2
    alzette_avx_03(state, brans);
#elif USE_NEON
    // Process the first 4 pairs of the state.
    alzette_simd(state, 0);

    if (brans == 8) {
      // If there's 4 more pairs to process, then do so.
      alzette_simd(state + 8, 1);
    } else if (brans == 6) {
        // Running alzette_simd with only 2 pairs is inefficient, so it's
        // faster to just handle the remaining 2 pairs like this.

        rc = RCON[8 >> 1];
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
#endif
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
