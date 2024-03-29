/*
Pyjamask-128-OCB reference code
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
#define KEYBYTES   PJ128REF_CRYPTO_KEYBYTES
#define NONCEBYTES PJ128REF_CRYPTO_NPUBBYTES
#define TAGBYTES   PJ128REF_CRYPTO_ABYTES

#include "pyjamask_ref/pyjamask.h"

typedef unsigned char block[16];

/* ------------------------------------------------------------------------- */

static void xor_block(block d, block s1, block s2) {
    unsigned i;
    for (i=0; i<16; i++)
        d[i] = s1[i] ^ s2[i];
}

/* ------------------------------------------------------------------------- */

static void double_block(block d, block s) {
    unsigned i;
    unsigned char tmp = s[0];
    for (i=0; i<15; i++)
        d[i] = (s[i] << 1) | (s[i+1] >> 7);
    d[15] = (s[15] << 1) ^ ((tmp >> 7) * 135);
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

    /* L_* = ENCIPHER(K, zeros(128)) */
//    AES_set_encrypt_key(k, KEYBYTES*8, &aes_key);
    memset(tmp, 0, 16);
    pjref_pyjamask_128_enc(tmp, k, lstar);
    /* L_$ = double(L_*) */
    double_block(ldollar, lstar);

    /* Process any whole blocks */

    /* Sum_0 = zeros(128) */
    memset(sum, 0, 16);
    /* Offset_0 = zeros(128) */
    memset(offset, 0, 16);
    for (i=1; i<=abytes/16; i++, a = a + 16) {
        /* Offset_i = Offset_{i-1} xor L_{ntz(i)} */
        calc_L_i(tmp, ldollar, i);
        xor_block(offset, offset, tmp);
        /* Sum_i = Sum_{i-1} xor ENCIPHER(K, A_i xor Offset_i) */
        xor_block(tmp, offset, a);
        pjref_pyjamask_128_enc(tmp, k, tmp);
        xor_block(sum, sum, tmp);
    }

    /* Process any final partial block; compute final hash value */

    abytes = abytes % 16;  /* Bytes in final block */
    if (abytes > 0) {
        /* Offset_* = Offset_m xor L_* */
        xor_block(offset, offset, lstar);
        /* tmp = (A_* || 1 || zeros(127-bitlen(A_*))) xor Offset_* */
        memset(tmp, 0, 16);
        memcpy(tmp, a, abytes);
        tmp[abytes] = 0x80;
        xor_block(tmp, offset, tmp);
        /* Sum = Sum_m xor ENCIPHER(K, tmp) */
        pjref_pyjamask_128_enc(tmp, k, tmp);
        xor_block(sum, tmp, sum);
    }

    memcpy(result, sum, 16);
}

/* ------------------------------------------------------------------------- */

static int pj128ref_ocb_crypt(unsigned char *out, unsigned char *k, unsigned char *n,
                     unsigned char *a, unsigned abytes,
                     unsigned char *in, unsigned inbytes, int encrypting) {
    block lstar, ldollar, sum, offset, ktop, pad, nonce, tag, tmp, ad_hash;
    unsigned char stretch[24];
    unsigned bottom, byteshift, bitshift, i;

    /* Setup AES and strip ciphertext of its tag */
    if ( ! encrypting ) {
         if (inbytes < TAGBYTES) return -1;
         inbytes -= TAGBYTES;
    }

    /* Key-dependent variables */

    /* L_* = ENCIPHER(K, zeros(128)) */
    memset(tmp, 0, 16);
    pjref_pyjamask_128_enc(tmp, k, lstar);
    /* L_$ = double(L_*) */
    double_block(ldollar, lstar);

    /* Nonce-dependent and per-encryption variables */

    /* Nonce = zeros(127-bitlen(N)) || 1 || N */
    memset(nonce,0,16);
    memcpy(&nonce[16-NONCEBYTES],n,NONCEBYTES);
    nonce[0] = (unsigned char)(((TAGBYTES * 8) % 128) << 1);
    nonce[16-NONCEBYTES-1] |= 0x01;
    /* bottom = str2num(Nonce[123..128]) */
    bottom = nonce[15] & 0x3F;
    /* Ktop = ENCIPHER(K, Nonce[1..122] || zeros(6)) */
    nonce[15] &= 0xC0;
    pjref_pyjamask_128_enc(nonce, k, ktop);
    /* Stretch = Ktop || (Ktop[1..64] xor Ktop[9..72]) */
    memcpy(stretch, ktop, 16);
    memcpy(tmp, &ktop[1], 8);
    xor_block(tmp, tmp, ktop);
    memcpy(&stretch[16],tmp,8);
    /* Offset_0 = Stretch[1+bottom..128+bottom] */
    byteshift = bottom/8;
    bitshift  = bottom%8;
    if (bitshift != 0)
        for (i=0; i<16; i++)
            offset[i] = (stretch[i+byteshift] << bitshift) |
                        (stretch[i+byteshift+1] >> (8-bitshift));
    else
        for (i=0; i<16; i++)
            offset[i] = stretch[i+byteshift];
    /* Checksum_0 = zeros(128) */
    memset(sum, 0, 16);

    /* Hash associated data */
    hash(ad_hash, k, a, abytes);

    /* Process any whole blocks */

    for (i=1; i<=inbytes/16; i++, in=in+16, out=out+16) {
        /* Offset_i = Offset_{i-1} xor L_{ntz(i)} */
        calc_L_i(tmp, ldollar, i);
        xor_block(offset, offset, tmp);

        xor_block(tmp, offset, in);
        if (encrypting) {
            /* Checksum_i = Checksum_{i-1} xor P_i */
            xor_block(sum, in, sum);
            /* C_i = Offset_i xor ENCIPHER(K, P_i xor Offset_i) */
            pjref_pyjamask_128_enc(tmp, k, tmp);
            xor_block(out, offset, tmp);
        } else {
            /* P_i = Offset_i xor DECIPHER(K, C_i xor Offset_i) */
            pjref_pyjamask_128_dec(tmp, k, tmp);
            xor_block(out, offset, tmp);
            /* Checksum_i = Checksum_{i-1} xor P_i */
            xor_block(sum, out, sum);
        }
    }

    /* Process any final partial block and compute raw tag */

    inbytes = inbytes % 16;  /* Bytes in final block */
    if (inbytes > 0) {
        /* Offset_* = Offset_m xor L_* */
        xor_block(offset, offset, lstar);
        /* Pad = ENCIPHER(K, Offset_*) */
        pjref_pyjamask_128_enc(offset, k, pad);

        if (encrypting) {
            /* Checksum_* = Checksum_m xor (P_* || 1 || zeros(127-bitlen(P_*))) */
            memset(tmp, 0, 16);
            memcpy(tmp, in, inbytes);
            tmp[inbytes] = 0x80;
            xor_block(sum, tmp, sum);
            /* C_* = P_* xor Pad[1..bitlen(P_*)] */
            xor_block(pad, tmp, pad);
            memcpy(out, pad, inbytes);
            out = out + inbytes;
        } else {
            /* P_* = C_* xor Pad[1..bitlen(C_*)] */
            memcpy(tmp, pad, 16);
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
    pjref_pyjamask_128_enc(tmp, k, tag);
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

void pj128ref_ocb_encrypt(unsigned char *c, unsigned char *k, unsigned char *n,
                 unsigned char *a, unsigned abytes,
                 unsigned char *p, unsigned pbytes) {
    pj128ref_ocb_crypt(c, k, n, a, abytes, p, pbytes, OCB_ENCRYPT);
}

/* ------------------------------------------------------------------------- */

int pj128ref_ocb_decrypt(unsigned char *p, unsigned char *k, unsigned char *n,
                unsigned char *a, unsigned abytes,
                unsigned char *c, unsigned cbytes) {
    return pj128ref_ocb_crypt(p, k, n, a, abytes, c, cbytes, OCB_DECRYPT);
}

/* ------------------------------------------------------------------------- */

int pj128ref_crypto_aead_encrypt(
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
    pj128ref_ocb_crypt(c, (unsigned char *)k, (unsigned char *)npub, (unsigned char *)ad,
            adlen, (unsigned char *)m, mlen, OCB_ENCRYPT);
    return 0;
}

int pj128ref_crypto_aead_decrypt(
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
    return pj128ref_ocb_crypt(m, (unsigned char *)k, (unsigned char *)npub,
            (unsigned char *)ad, adlen, (unsigned char *)c, clen, OCB_DECRYPT);
}

