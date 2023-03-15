/*
===============================================================================

    Reference implementation of Pyjamask block ciphers in C
    
    Copyright (C) 2019  Dahmun Goudarzi, Jérémy Jean, Stefan Kölbl, 
    Thomas Peyrin, Matthieu Rivain, Yu Sasaki, Siang Meng Sim

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

===============================================================================
 */

#include <stdint.h>

#include "platform_defines.h"

#if USE_AVX2
#include <immintrin.h>
#endif

//==============================================================================
//=== Parameters
//==============================================================================

#define STATE_SIZE_96        3
#define STATE_SIZE_128       4

#define NB_ROUNDS_96        14
#define NB_ROUNDS_128       14
#define NB_ROUNDS_KS        14

//==============================================================================
//=== Macros
//==============================================================================

#define right_rotate(row) \
    row = (row >> 1) | (row << 31);

#define left_rotate(row,n) \
    row = (row >> n) | (row << (32-n));

//==============================================================================
//=== Constants
//==============================================================================

#define COL_M0          0xa3861085
#define COL_M1          0x63417021
#define COL_M2          0x692cf280
#define COL_M3          0x48a54813
#define COL_MK          0xb881b9ca

#define COL_INV_M0      0x2037a121
#define COL_INV_M1      0x108ff2a0 
#define COL_INV_M2      0x9054d8c0 
#define COL_INV_M3      0x3354b117

#define KS_CONSTANT_0   0x00000080
#define KS_CONSTANT_1   0x00006a00
#define KS_CONSTANT_2   0x003f0000
#define KS_CONSTANT_3   0x24000000

#define KS_ROT_GAP1      8
#define KS_ROT_GAP2     15
#define KS_ROT_GAP3     18

//==============================================================================
//=== Common functions
//==============================================================================

#if USE_AVX2
// The serial implementation packs the 8-bit integers into 32-bit ints in the reversed
// order, so we need to shuffle after loading to flip the order of bytes within each
// 32-bit int in the vector register to match this reverse order.
//
// Example:
// Plaintext bytes P0, P1, P2, P3 get packed as [P3, P2, P1, P0].
// SIMD load will load them as [P0, P1, P2, P3], so needs to be shuffled.
void pj128simd_load_state_x8(const uint8_t* plaintext, __m256i* state, int state_size)
{
    __m256i shuffle_order = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
        19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28);

    for (unsigned int i = 0; i < state_size; i++) {
        __m256i temp = _mm256_i32gather_epi32(
            plaintext,
            _mm256_setr_epi32(0 + i, 4 + i, 8 + i, 12 + i, 16 + i, 20 + i, 24 + i, 28 + i),
            sizeof(uint32_t));

        state[i] = _mm256_shuffle_epi8(temp, shuffle_order);
    }
}
#endif

void pj128simd_load_state(const uint8_t *plaintext, uint32_t *state, int state_size)
{
    int i;

    for (i=0; i<state_size; i++)
    {
        state[i] =                   plaintext[4*i+0];
        state[i] = (state[i] << 8) | plaintext[4*i+1];
        state[i] = (state[i] << 8) | plaintext[4*i+2];
        state[i] = (state[i] << 8) | plaintext[4*i+3];
    }
}

#if USE_AVX2
void pj128simd_unload_state_x8(uint8_t* ciphertext, const __m256i* state, int state_size)
{
    __m256i shuffle_order = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
        19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28);

    // As the order of bytes was shuffled when loading, they need to be shuffled back before
    // unloading.
    uint32_t state_0[8];
    _mm256_storeu_si256(state_0, _mm256_shuffle_epi8(state[0], shuffle_order));
    uint32_t state_1[8];
    _mm256_storeu_si256(state_1, _mm256_shuffle_epi8(state[1], shuffle_order));
    uint32_t state_2[8];
    _mm256_storeu_si256(state_2, _mm256_shuffle_epi8(state[2], shuffle_order));
    uint32_t state_3[8];
    _mm256_storeu_si256(state_3, _mm256_shuffle_epi8(state[3], shuffle_order));

    for (unsigned int i = 0; i < 8; i++) {
        ((uint32_t*)ciphertext)[(i * 4) + 0] = state_0[i];
        ((uint32_t*)ciphertext)[(i * 4) + 1] = state_1[i];
        ((uint32_t*)ciphertext)[(i * 4) + 2] = state_2[i];
        ((uint32_t*)ciphertext)[(i * 4) + 3] = state_3[i];
    }
}
#endif

void pj128simd_unload_state(uint8_t *ciphertext, const uint32_t *state, int state_size)
{
    int i;

    for (i=0; i<state_size; i++)
    {
        ciphertext [4*i+0] = (uint8_t) (state[i] >> 24);
        ciphertext [4*i+1] = (uint8_t) (state[i] >> 16);
        ciphertext [4*i+2] = (uint8_t) (state[i] >>  8);
        ciphertext [4*i+3] = (uint8_t) (state[i] >>  0);
    }
}

#if USE_AVX2
__m256i pj128simd_mat_mult_x8(__m256i mat_col, __m256i vec) {
    __m256i mask, res = _mm256_set1_epi32(0);

    for (int i = 31; i >= 0; i--) {
        // vec >> i
        mask = _mm256_srli_epi32(vec, i);
        // & 1
        mask = _mm256_and_si256(mask, _mm256_set1_epi32(1));
        // -
        mask = _mm256_sign_epi32(mask, _mm256_set1_epi32(-1));

        // res ^= mask & mat_col;
        res = _mm256_xor_si256(res, _mm256_and_si256(mask, mat_col));

        // right_rotate(mat_col);
        mat_col = _mm256_or_epi32(
            _mm256_srli_epi32(mat_col, 1),
            _mm256_slli_epi32(mat_col, 31));
    }

    return res;
}
#endif

uint32_t pj128simd_mat_mult(uint32_t mat_col, uint32_t vec)
{
    int i;
    uint32_t mask, res=0;

    for (i = 31; i>=0; i--)
    {
        mask = -((vec >> i) & 1);
        res ^= mask & mat_col;
        right_rotate(mat_col);
    }

    return res;
}

//==============================================================================
//=== Key schedule
//==============================================================================

void pj128simd_ks_mix_comlumns(const uint32_t *ks_prev, uint32_t *ks_next)
{
    uint32_t tmp;

    tmp = ks_prev[0] ^ ks_prev[1] ^ ks_prev[2] ^ ks_prev[3];

    ks_next[0] = ks_prev[0] ^ tmp;
    ks_next[1] = ks_prev[1] ^ tmp;
    ks_next[2] = ks_prev[2] ^ tmp;
    ks_next[3] = ks_prev[3] ^ tmp;
}

void pj128simd_ks_mix_rotate_rows(uint32_t *ks_state)
{
    ks_state[0] = pj128simd_mat_mult(COL_MK, ks_state[0]);
    left_rotate(ks_state[1],KS_ROT_GAP1)
    left_rotate(ks_state[2],KS_ROT_GAP2)
    left_rotate(ks_state[3],KS_ROT_GAP3)
}

void pj128simd_ks_add_constant(uint32_t *ks_state, const uint32_t ctr)
{
    ks_state[0] ^= KS_CONSTANT_0 ^ ctr;
    ks_state[1] ^= KS_CONSTANT_1;
    ks_state[2] ^= KS_CONSTANT_2;
    ks_state[3] ^= KS_CONSTANT_3;
}

void pj128simd_key_schedule(const uint8_t *key, uint32_t* round_keys)
{
    int r;
    uint32_t *ks_state = round_keys;

    pj128simd_load_state(key, ks_state, 4);

    for (r=0; r<NB_ROUNDS_KS; r++)
    {
        ks_state += 4;

        pj128simd_ks_mix_comlumns(ks_state-4, ks_state);
        pj128simd_ks_mix_rotate_rows(ks_state);
        pj128simd_ks_add_constant(ks_state,r);

    }    
}

//==============================================================================
//=== Pyjamask-96 (encryption)
//==============================================================================

void pj128simd_mix_rows_96(uint32_t *state)
{
    state[0] = pj128simd_mat_mult(COL_M0, state[0]);
    state[1] = pj128simd_mat_mult(COL_M1, state[1]);
    state[2] = pj128simd_mat_mult(COL_M2, state[2]);
}

void pj128simd_sub_bytes_96(uint32_t *state)
{
    state[0] ^= state[1];
    state[1] ^= state[2];
    state[2] ^= state[0] & state[1];
    state[0] ^= state[1] & state[2];
    state[1] ^= state[0] & state[2];
    state[2] ^= state[0];
    state[0] ^= state[1];
    state[2] = ~state[2];

    // swap state[0] <-> state[1]
    state[0] ^= state[1];
    state[1] ^= state[0];
    state[0] ^= state[1];
}

void pj128simd_add_round_key_96(uint32_t *state, const uint32_t *round_key, int r)
{
    state[0] ^= round_key[4*r+0];
    state[1] ^= round_key[4*r+1];
    state[2] ^= round_key[4*r+2];
}

void pj128simd_pyjamask_96_enc(const uint8_t *plaintext, const uint8_t *key, uint8_t *ciphertext)
{
    int r;
    uint32_t state[STATE_SIZE_96];
    uint32_t round_keys[4*(NB_ROUNDS_KS+1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state(plaintext, state, STATE_SIZE_96);

    for (r=0; r<NB_ROUNDS_96; r++)
    {
        pj128simd_add_round_key_96(state, round_keys, r);
        pj128simd_sub_bytes_96(state);
        pj128simd_mix_rows_96(state);
    }

    pj128simd_add_round_key_96(state, round_keys, NB_ROUNDS_96);

    pj128simd_unload_state(ciphertext, state, STATE_SIZE_96);
}


//==============================================================================
//=== Pyjamask-96 (decryption)
//==============================================================================

void pj128simd_inv_mix_rows_96(uint32_t *state)
{
    state[0] = pj128simd_mat_mult(COL_INV_M0, state[0]);
    state[1] = pj128simd_mat_mult(COL_INV_M1, state[1]);
    state[2] = pj128simd_mat_mult(COL_INV_M2, state[2]);
}

void pj128simd_inv_sub_bytes_96(uint32_t *state)
{
    // swap state[0] <-> state[1]
    state[0] ^= state[1];
    state[1] ^= state[0];
    state[0] ^= state[1];

    state[2] = ~state[2];
    state[0] ^= state[1];
    state[2] ^= state[0];
    state[1] ^= state[2] & state[0];
    state[0] ^= state[1] & state[2];
    state[2] ^= state[0] & state[1];
    state[1] ^= state[2];
    state[0] ^= state[1];
}

void pj128simd_pyjamask_96_dec(const uint8_t *ciphertext, const uint8_t *key, uint8_t *plaintext)
{
    int r;
    uint32_t state[STATE_SIZE_96];
    uint32_t round_keys[4*(NB_ROUNDS_KS+1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state(ciphertext, state, STATE_SIZE_96);

    pj128simd_add_round_key_96(state, round_keys, NB_ROUNDS_96);

    for (r=NB_ROUNDS_96-1; r>=0; r--)
    {
        pj128simd_inv_mix_rows_96(state);
        pj128simd_inv_sub_bytes_96(state);
        pj128simd_add_round_key_96(state, round_keys, r);
    }

    pj128simd_unload_state(plaintext, state, STATE_SIZE_96);
}

//==============================================================================
//=== Pyjamask-128 (encryption)
//==============================================================================

#if USE_AVX2
void pj128simd_mix_rows_128_x8(__m256i* state)
{
    state[0] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_M0), state[0]);
    state[1] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_M1), state[1]);
    state[2] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_M2), state[2]);
    state[3] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_M3), state[3]);
}
#endif

void pj128simd_mix_rows_128(uint32_t *state)
{
    state[0] = pj128simd_mat_mult(COL_M0, state[0]);
    state[1] = pj128simd_mat_mult(COL_M1, state[1]);
    state[2] = pj128simd_mat_mult(COL_M2, state[2]);
    state[3] = pj128simd_mat_mult(COL_M3, state[3]);
}

#if USE_AVX2
void pj128simd_sub_bytes_128_x8(__m256i* state) {
    state[0] = _mm256_xor_si256(state[0], state[3]);
    state[3] = _mm256_xor_si256(state[3], _mm256_and_si256(state[0], state[1]));
    state[0] = _mm256_xor_si256(state[0], _mm256_and_si256(state[1], state[2]));
    state[1] = _mm256_xor_si256(state[1], _mm256_and_si256(state[2], state[3]));
    state[2] = _mm256_xor_si256(state[2], _mm256_and_si256(state[0], state[3]));
    state[2] = _mm256_xor_si256(state[2], state[1]);
    state[1] = _mm256_xor_si256(state[1], state[0]);
    // state[3] = ~state[3] is equivalent to XOR with all 1's.
    // _mm256_set1_epi32(-1) usually compiles to cmpeq(a, a), but force it anyway.
    // Supposedly quicker in the cases that it doesn't optimise to cmpeq automatically.
    state[3] = _mm256_xor_si256(state[3], _mm256_cmpeq_epi32(state[3], state[3]));

    // swap state[2] <-> state[3]
    state[2] = _mm256_xor_si256(state[2], state[3]);
    state[3] = _mm256_xor_si256(state[3], state[2]);
    state[2] = _mm256_xor_si256(state[2], state[3]);
}
#endif

void pj128simd_sub_bytes_128(uint32_t *state)
{
    state[0] ^= state[3];
    state[3] ^= state[0] & state[1];
    state[0] ^= state[1] & state[2];
    state[1] ^= state[2] & state[3];
    state[2] ^= state[0] & state[3];
    state[2] ^= state[1];
    state[1] ^= state[0];
    state[3] = ~state[3];

    // swap state[2] <-> state[3]
    state[2] ^= state[3];
    state[3] ^= state[2];
    state[2] ^= state[3];
}

#if USE_AVX2
void pj128simd_add_round_key_128_x8(__m256i* state, const uint32_t* round_key, int r)
{
    state[0] = _mm256_xor_si256(state[0], _mm256_set1_epi32(round_key[4 * r + 0]));
    state[1] = _mm256_xor_si256(state[1], _mm256_set1_epi32(round_key[4 * r + 1]));
    state[2] = _mm256_xor_si256(state[2], _mm256_set1_epi32(round_key[4 * r + 2]));
    state[3] = _mm256_xor_si256(state[3], _mm256_set1_epi32(round_key[4 * r + 3]));
}
#endif

void pj128simd_add_round_key_128(uint32_t *state, const uint32_t *round_key, int r)
{
    state[0] ^= round_key[4*r+0];
    state[1] ^= round_key[4*r+1];
    state[2] ^= round_key[4*r+2];
    state[3] ^= round_key[4*r+3];
}

#if USE_AVX2
void pj128simd_pyjamask_128_enc_x8(const uint8_t* plaintext, const uint8_t* key, uint8_t* ciphertext) {
    int r;
    __m256i state[STATE_SIZE_128];
    uint32_t round_keys[4 * (NB_ROUNDS_KS + 1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state_x8(plaintext, state, STATE_SIZE_128);


    for (r = 0; r < NB_ROUNDS_128; r++)
    {
        pj128simd_add_round_key_128_x8(state, round_keys, r);
        pj128simd_sub_bytes_128_x8(state);
        pj128simd_mix_rows_128_x8(state);
    }

    pj128simd_add_round_key_128_x8(state, round_keys, NB_ROUNDS_128);

    pj128simd_unload_state_x8(ciphertext, state, STATE_SIZE_128);
}
#endif

void pj128simd_pyjamask_128_enc(const uint8_t *plaintext, const uint8_t *key, uint8_t *ciphertext)
{
    int r;
    uint32_t state[STATE_SIZE_128];
    uint32_t round_keys[4*(NB_ROUNDS_KS+1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state(plaintext, state, STATE_SIZE_128);


    for (r=0; r<NB_ROUNDS_128; r++)
    {
        pj128simd_add_round_key_128(state, round_keys, r);
        pj128simd_sub_bytes_128(state);
        pj128simd_mix_rows_128(state);
    }

    pj128simd_add_round_key_128(state, round_keys, NB_ROUNDS_128);
    
    pj128simd_unload_state(ciphertext, state, STATE_SIZE_128);
}

//==============================================================================
//=== Pyjamask-128 (decryption)
//==============================================================================

#if USE_AVX2
void pj128simd_inv_mix_rows_128_x8(__m256i* state)
{
    state[0] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_INV_M0), state[0]);
    state[1] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_INV_M1), state[1]);
    state[2] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_INV_M2), state[2]);
    state[3] = pj128simd_mat_mult_x8(_mm256_set1_epi32(COL_INV_M3), state[3]);
}
#endif

void pj128simd_inv_mix_rows_128(uint32_t *state)
{
    state[0] = pj128simd_mat_mult(COL_INV_M0, state[0]);
    state[1] = pj128simd_mat_mult(COL_INV_M1, state[1]);
    state[2] = pj128simd_mat_mult(COL_INV_M2, state[2]);
    state[3] = pj128simd_mat_mult(COL_INV_M3, state[3]);
}

#if USE_AVX2
void pj128simd_inv_sub_bytes_128_x8(__m256i* state)
{
    // swap state[2] <-> state[3]
    state[2] = _mm256_xor_si256(state[2], state[3]);
    state[3] = _mm256_xor_si256(state[3], state[2]);
    state[2] = _mm256_xor_si256(state[2], state[3]);

    state[3] = _mm256_xor_si256(state[3], _mm256_cmpeq_epi32(state[3], state[3]));
    state[1] = _mm256_xor_si256(state[1], state[0]);
    state[2] = _mm256_xor_si256(state[2], state[1]);
    state[2] = _mm256_xor_si256(state[2], _mm256_and_si256(state[3], state[0]));
    state[1] = _mm256_xor_si256(state[1], _mm256_and_si256(state[2], state[3]));
    state[0] = _mm256_xor_si256(state[0], _mm256_and_si256(state[1], state[2]));
    state[3] = _mm256_xor_si256(state[3], _mm256_and_si256(state[0], state[1]));
    state[0] = _mm256_xor_si256(state[0], state[3]);
}
#endif

void pj128simd_inv_sub_bytes_128(uint32_t *state)
{
    // swap state[2] <-> state[3]
    state[2] ^= state[3];
    state[3] ^= state[2];
    state[2] ^= state[3];

    state[3] = ~state[3];
    state[1] ^= state[0];
    state[2] ^= state[1];
    state[2] ^= state[3] & state[0];
    state[1] ^= state[2] & state[3];
    state[0] ^= state[1] & state[2];
    state[3] ^= state[0] & state[1];
    state[0] ^= state[3];
}

#if USE_AVX2
void pj128simd_pyjamask_128_dec_x8(const uint8_t* ciphertext, const uint8_t* key, uint8_t* plaintext)
{
    int r;
    __m256i state[STATE_SIZE_128];
    uint32_t round_keys[4 * (NB_ROUNDS_KS + 1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state_x8(ciphertext, state, STATE_SIZE_128);

    pj128simd_add_round_key_128_x8(state, round_keys, NB_ROUNDS_128);

    for (r = NB_ROUNDS_128 - 1; r >= 0; r--)
    {
        pj128simd_inv_mix_rows_128_x8(state);
        pj128simd_inv_sub_bytes_128_x8(state);
        pj128simd_add_round_key_128_x8(state, round_keys, r);
    }

    pj128simd_unload_state_x8(plaintext, state, STATE_SIZE_128);
}
#endif

void pj128simd_pyjamask_128_dec(const uint8_t *ciphertext, const uint8_t *key, uint8_t *plaintext)
{
    int r;
    uint32_t state[STATE_SIZE_128];
    uint32_t round_keys[4*(NB_ROUNDS_KS+1)];

    pj128simd_key_schedule(key, round_keys);
    pj128simd_load_state(ciphertext, state, STATE_SIZE_128);

    pj128simd_add_round_key_128(state, round_keys, NB_ROUNDS_128);
    
    for (r=NB_ROUNDS_128-1; r>=0; r--)
    {
        pj128simd_inv_mix_rows_128(state);
        pj128simd_inv_sub_bytes_128(state);
        pj128simd_add_round_key_128(state, round_keys, r);
    }

    pj128simd_unload_state(plaintext, state, STATE_SIZE_128);
}


