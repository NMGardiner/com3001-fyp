/*
===============================================================================

    Header file for Pyjamsk block ciphers in C

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

void pjsimd_pyjamask_96_enc (const unsigned char *plaintext,  const unsigned char *key, unsigned char *ciphertext);
void pjsimd_pyjamask_96_dec (const unsigned char *ciphertext, const unsigned char *key, unsigned char *plaintext);

void pjsimd_pyjamask_128_enc(const unsigned char *plaintext,  const unsigned char *key, unsigned char *ciphertext);
void pjsimd_pyjamask_128_dec(const unsigned char *ciphertext, const unsigned char *key, unsigned char *plaintext );

#if __AVX2__
// 8 instances of Pyjamask96 in parallel.
void pjsimd_pyjamask_96_enc_x8(const unsigned char* plaintext, const unsigned char* key, unsigned char* ciphertext);
void pjsimd_pyjamask_96_dec_x8(const unsigned char* ciphertext, const unsigned char* key, unsigned char* plaintext);

// 8 instances of Pyjamask128 in parallel.
void pjsimd_pyjamask_128_enc_x8(const unsigned char* plaintext, const unsigned char* key, unsigned char* ciphertext);
void pjsimd_pyjamask_128_dec_x8(const unsigned char* ciphertext, const unsigned char* key, unsigned char* plaintext);
#elif __ARM_NEON
// 4 instances of Pyjamask96 in parallel.
void pjsimd_pyjamask_96_enc_x4(const unsigned char* plaintext, const unsigned char* key, unsigned char* ciphertext);
void pjsimd_pyjamask_96_dec_x4(const unsigned char* ciphertext, const unsigned char* key, unsigned char* plaintext);

// 4 instances of Pyjamask128 in parallel.
void pjsimd_pyjamask_128_enc_x4(const unsigned char* plaintext, const unsigned char* key, unsigned char* ciphertext);
void pjsimd_pyjamask_128_dec_x4(const unsigned char* ciphertext, const unsigned char* key, unsigned char* plaintext);
#endif