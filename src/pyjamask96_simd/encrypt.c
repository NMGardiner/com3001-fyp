/*
Pyjamask-96-OCB reference code
Modified by: Siang Meng Sim
Email: crypto.s.m.sim@gmail.com
Date : 25/02/2019
*/

/*
// CAESAR OCB v1 reference code. Info: http://www.cs.ucdavis.edu/~rogaway/ocb
//
// ** This version is slow and susceptible to side-channel attacks. **
// ** Do not use for any purpose other than to understand OCB.      **
//
// Written by Ted Krovetz (ted@krovetz.net). Last modified 13 May 2014.
//
// Phillip Rogaway holds patents relevant to OCB. See the following for
// his free patent grant: http://www.cs.ucdavis.edu/~rogaway/ocb/grant.htm
//
// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>
*/

#include <string.h>

#include "api.h"
#define KEYBYTES   PJ96SIMD_CRYPTO_KEYBYTES
#define NONCEBYTES PJ96SIMD_CRYPTO_NPUBBYTES
#define TAGBYTES   PJ96SIMD_CRYPTO_ABYTES

#include "pyjamask_simd/pyjamask.h"

#if __AVX2__
#include <stdint.h>
#endif

typedef unsigned char block[12];

/* ------------------------------------------------------------------------- */

static void xor_block(block d, block s1, block s2) {
    unsigned i;
    for (i=0; i<12; i++)
        d[i] = s1[i] ^ s2[i];
}

/* ------------------------------------------------------------------------- */

static void double_block(block d, block s) {
    /*irreducible polynomial = x^{96} + x^{10} + x^{9} + x^{6} + 1*/
    unsigned i;
    unsigned char tmp = s[0];
    for (i=0; i<11; i++)
        d[i] = (s[i] << 1) | (s[i+1] >> 7);
    d[11] = (s[11] << 1) ^ ((tmp >> 7) * 65);   /*2^6 + 1*/
    d[10] ^= ((tmp >> 7) * 6); /*2^{10-8} + 2^{9-8}*/
}

/* ------------------------------------------------------------------------- */

static void calc_L_i(block l, block ldollar, unsigned i) {
    double_block(l, ldollar);         /* l is now L_0               */
    for ( ; (i&1)==0 ; i>>=1)
        double_block(l,l);            /* double for each trailing 0 */
}

/* ------------------------------------------------------------------------- */

static void hash(block result, unsigned char *k,
                 unsigned char *a, unsigned abytes) {
    block lstar, ldollar, offset, sum, tmp;
    unsigned i;

    /* Key-dependent variables */

    /* L_* = ENCIPHER(K, zeros(96)) */
//    AES_set_encrypt_key(k, KEYBYTES*8, &aes_key);
    memset(tmp, 0, 12);
    pjsimd_pyjamask_96_enc(tmp, k, lstar);
    /* L_$ = double(L_*) */
    double_block(ldollar, lstar);

    /* Process any whole blocks */

    /* Sum_0 = zeros(96) */
    memset(sum, 0, 12);
    /* Offset_0 = zeros(96) */
    memset(offset, 0, 12);
    for (i=1; i<=abytes/12; i++, a = a + 12) {
        /* Offset_i = Offset_{i-1} xor L_{ntz(i)} */
        calc_L_i(tmp, ldollar, i);
        xor_block(offset, offset, tmp);
        /* Sum_i = Sum_{i-1} xor ENCIPHER(K, A_i xor Offset_i) */
        xor_block(tmp, offset, a);
        pjsimd_pyjamask_96_enc(tmp, k, tmp);
        xor_block(sum, sum, tmp);
    }

    /* Process any final partial block; compute final hash value */

    abytes = abytes % 12;  /* Bytes in final block */
    if (abytes > 0) {
        /* Offset_* = Offset_m xor L_* */
        xor_block(offset, offset, lstar);
        /* tmp = (A_* || 1 || zeros(95-bitlen(A_*))) xor Offset_* */
        memset(tmp, 0, 12);
        memcpy(tmp, a, abytes);
        tmp[abytes] = 0x80;
        xor_block(tmp, offset, tmp);
        /* Sum = Sum_m xor ENCIPHER(K, tmp) */
        pjsimd_pyjamask_96_enc(tmp, k, tmp);
        xor_block(sum, tmp, sum);
    }

    memcpy(result, sum, 12);
}

/* ------------------------------------------------------------------------- */

static int pj96simd_ocb_crypt(unsigned char *out, unsigned char *k, unsigned char *n,
                     unsigned char *a, unsigned abytes,
                     unsigned char *in, unsigned inbytes, int encrypting) {
    block lstar, ldollar, sum, offset, ktop, pad, nonce, tag, tmp, ad_hash;
    unsigned char stretch[20];
    unsigned bottom, byteshift, bitshift, i;

    /* Setup AES and strip ciphertext of its tag */
    if ( ! encrypting ) {
         if (inbytes < TAGBYTES) return -1;
         inbytes -= TAGBYTES;
    }

    /* Key-dependent variables */

    /* L_* = ENCIPHER(K, zeros(128)) */
    memset(tmp, 0, 12);
    pjsimd_pyjamask_96_enc(tmp, k, lstar);
    /* L_$ = double(L_*) */
    double_block(ldollar, lstar);

    /* Nonce-dependent and per-encryption variables */

    /* Nonce = zeros(127-bitlen(N)) || 1 || N */
    memset(nonce,0,12);
    memcpy(&nonce[12-NONCEBYTES],n,NONCEBYTES);
    nonce[0] = (unsigned char)(((TAGBYTES * 8) % 128) << 1);
    nonce[12-NONCEBYTES-1] |= 0x01;
    /* bottom = str2num(Nonce[91..96]) */
    bottom = nonce[11] & 0x3F;
    /* Ktop = ENCIPHER(K, Nonce[1..90] || zeros(6)) */
    nonce[11] &= 0xC0;
    pjsimd_pyjamask_96_enc(nonce, k, ktop);
    /* Stretch = Ktop || (Ktop[1..64] xor Ktop[9..72]) */
    memcpy(stretch, ktop, 12);
    for(i=0;i<8;i++){
        tmp[i] = (ktop[i+1]<<1) | (ktop[i+2]>>7);   /*OCB96: leftshift 9 bits*/
    }
    xor_block(tmp, tmp, ktop);
    memcpy(&stretch[12],tmp,8);
    /* Offset_0 = Stretch[1+bottom..128+bottom] */
    byteshift = bottom/8;
    bitshift  = bottom%8;
    if (bitshift != 0)
        for (i=0; i<12; i++)
            offset[i] = (stretch[i+byteshift] << bitshift) |
                        (stretch[i+byteshift+1] >> (8-bitshift));
    else
        for (i=0; i<12; i++)
            offset[i] = stretch[i+byteshift];
    /* Checksum_0 = zeros(96) */
    memset(sum, 0, 12);

    /* Hash associated data */
    hash(ad_hash, k, a, abytes);

    /* Process any whole blocks */

    // Don't combine AVX2/NEON into one, as we may do 4 blocks with AVX eventually.
#if __AVX2__
    // Process groups of 8 96-bit blocks first.
    for (i = 1; i + 7 <= (inbytes / 12); i += 8, in += (12 * 8), out += (12 * 8)) {
        block tmp_x8[8];
        block offset_x8[8];

        // Pre-process everything for the next 8 blocks.
        for (unsigned int j = 0; j < 8; j++) {
            // Calculate L_i.
            calc_L_i(tmp_x8[j], ldollar, j + i);

            // XOR L_i into the offset.
            xor_block(offset_x8[j], (j == 0) ? offset : offset_x8[j - 1], tmp_x8[j]);

            // XOR the plaintext block and offset into tmp.
            xor_block(tmp_x8[j], offset_x8[j], in + (12 * j));

            if (encrypting) {
                // For encryption, XOR in the plaintext block to the sum.
                xor_block(sum, in + (12 * j), sum);
            }
        }

        // Run 8 instances of the block cipher in parallel.
        if (encrypting) {
            pjsimd_pyjamask_96_enc_x8(tmp_x8, k, tmp_x8);
        }
        else {
            pjsimd_pyjamask_96_dec_x8(tmp_x8, k, tmp_x8);
        }

        // Finally, XOR tmp and the offset into the output block. For decryption, also
        // XOR the resulting plaintext block into the sum.
        for (unsigned int j = 0; j < 8; j++) {
            xor_block(out + (12 * j), offset_x8[j], tmp_x8[j]);

            if (!encrypting) {
                xor_block(sum, out + (12 * j), sum);
            }
        }

        // Copy back offset from the final block.
        memcpy(offset, offset_x8[7], 12);
    }
#elif __ARM_NEON
    // Process groups of 4 96-bit blocks first.
    for (i = 1; i + 3 <= (inbytes / 12); i += 4, in += (12 * 4), out += (12 * 4)) {
        block tmp_x4[4];
        block offset_x4[4];

        // Pre-process everything for the next 4 blocks.
        for (unsigned int j = 0; j < 4; j++) {
            // Calculate L_i.
            calc_L_i(tmp_x4[j], ldollar, j + i);

            // XOR L_i into the offset.
            xor_block(offset_x4[j], (j == 0) ? offset : offset_x4[j - 1], tmp_x4[j]);

            // XOR the plaintext block and offset into tmp.
            xor_block(tmp_x4[j], offset_x4[j], in + (12 * j));

            if (encrypting) {
                // For encryption, XOR in the plaintext block to the sum.
                xor_block(sum, in + (12 * j), sum);
            }
        }

        // Run 4 instances of the block cipher in parallel.
        if (encrypting) {
            pjsimd_pyjamask_96_enc_x4(tmp_x4, k, tmp_x4);
        }
        else {
            pjsimd_pyjamask_96_dec_x4(tmp_x4, k, tmp_x4);
        }

        // Finally, XOR tmp and the offset into the output block. For decryption, also
        // XOR the resulting plaintext block into the sum.
        for (unsigned int j = 0; j < 4; j++) {
            xor_block(out + (12 * j), offset_x4[j], tmp_x4[j]);

            if (!encrypting) {
                xor_block(sum, out + (12 * j), sum);
            }
        }

        // Copy back offset from the final block.
        memcpy(offset, offset_x4[3], 12);
    }
#endif

#if __AVX2__ || __ARM_NEON
    // Process any remaining full blocks.
    for (; i <= inbytes / 12; i++, in = in + 12, out = out + 12) {
#else
    for (i=1; i<=inbytes/12; i++, in=in+12, out=out+12) {
#endif
        /* Offset_i = Offset_{i-1} xor L_{ntz(i)} */
        calc_L_i(tmp, ldollar, i);
        xor_block(offset, offset, tmp);

        xor_block(tmp, offset, in);
        if (encrypting) {
            /* Checksum_i = Checksum_{i-1} xor P_i */
            xor_block(sum, in, sum);
            /* C_i = Offset_i xor ENCIPHER(K, P_i xor Offset_i) */
            pjsimd_pyjamask_96_enc(tmp, k, tmp);
            xor_block(out, offset, tmp);
        } else {
            /* P_i = Offset_i xor DECIPHER(K, C_i xor Offset_i) */
            pjsimd_pyjamask_96_dec(tmp, k, tmp);
            xor_block(out, offset, tmp);
            /* Checksum_i = Checksum_{i-1} xor P_i */
            xor_block(sum, out, sum);
        }
    }

    /* Process any final partial block and compute raw tag */

    inbytes = inbytes % 12;  /* Bytes in final block */
    if (inbytes > 0) {
        /* Offset_* = Offset_m xor L_* */
        xor_block(offset, offset, lstar);
        /* Pad = ENCIPHER(K, Offset_*) */
        pjsimd_pyjamask_96_enc(offset, k, pad);

        if (encrypting) {
            /* Checksum_* = Checksum_m xor (P_* || 1 || zeros(127-bitlen(P_*))) */
            memset(tmp, 0, 12);
            memcpy(tmp, in, inbytes);
            tmp[inbytes] = 0x80;
            xor_block(sum, tmp, sum);
            /* C_* = P_* xor Pad[1..bitlen(P_*)] */
            xor_block(pad, tmp, pad);
            memcpy(out, pad, inbytes);
            out = out + inbytes;
        } else {
            /* P_* = C_* xor Pad[1..bitlen(C_*)] */
            memcpy(tmp, pad, 12);
            memcpy(tmp, in, inbytes);
            xor_block(tmp, pad, tmp);
            tmp[inbytes] = 0x80;     /* tmp == P_* || 1 || zeros(127-bitlen(P_*)) */
            memcpy(out, tmp, inbytes);
            /* Checksum_* = Checksum_m xor (P_* || 1 || zeros(127-bitlen(P_*))) */
            xor_block(sum, tmp, sum);
            in = in + inbytes;
        }
    }

    /* Tag = ENCIPHER(K, Checksum xor Offset xor L_$) xor HASH(K,A) */
    xor_block(tmp, sum, offset);
    xor_block(tmp, tmp, ldollar);
    pjsimd_pyjamask_96_enc(tmp, k, tag);
    xor_block(tag, ad_hash, tag);

    if (encrypting) {
        memcpy(out, tag, TAGBYTES);
        return 0;
    } else
        return (memcmp(in,tag,TAGBYTES) ? -1 : 0);     /* Check for validity */
}

/* ------------------------------------------------------------------------- */

#define OCB_ENCRYPT 1
#define OCB_DECRYPT 0

void pj96simd_ocb_encrypt(unsigned char *c, unsigned char *k, unsigned char *n,
                 unsigned char *a, unsigned abytes,
                 unsigned char *p, unsigned pbytes) {
    pj96simd_ocb_crypt(c, k, n, a, abytes, p, pbytes, OCB_ENCRYPT);
}

/* ------------------------------------------------------------------------- */

int pj96simd_ocb_decrypt(unsigned char *p, unsigned char *k, unsigned char *n,
                unsigned char *a, unsigned abytes,
                unsigned char *c, unsigned cbytes) {
    return pj96simd_ocb_crypt(p, k, n, a, abytes, c, cbytes, OCB_DECRYPT);
}

/* ------------------------------------------------------------------------- */

int pj96simd_crypto_aead_encrypt(
unsigned char *c,unsigned long long *clen,
const unsigned char *m,unsigned long long mlen,
const unsigned char *ad,unsigned long long adlen,
const unsigned char *nsec,
const unsigned char *npub,
const unsigned char *k
)
{
    (void) (nsec); // unused argument
    *clen = mlen + TAGBYTES;
    pj96simd_ocb_crypt(c, (unsigned char *)k, (unsigned char *)npub, (unsigned char *)ad,
            adlen, (unsigned char *)m, mlen, OCB_ENCRYPT);
    return 0;
}

int pj96simd_crypto_aead_decrypt(
unsigned char *m,unsigned long long *mlen,
unsigned char *nsec,
const unsigned char *c,unsigned long long clen,
const unsigned char *ad,unsigned long long adlen,
const unsigned char *npub,
const unsigned char *k
)
{
    (void) (nsec); // unused argument
    *mlen = clen - TAGBYTES;
    return pj96simd_ocb_crypt(m, (unsigned char *)k, (unsigned char *)npub,
            (unsigned char *)ad, adlen, (unsigned char *)c, clen, OCB_DECRYPT);
}

