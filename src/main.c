#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include <immintrin.h>

#include <inttypes.h>

// TODO: Sort this!
#include "esch256_ref/api.h"
#include "esch256_simd/api.h"

#include "esch384_ref/api.h"
#include "esch384_simd/api.h"

#include "schwaemm128128_ref/api.h"
#include "schwaemm128128_simd/api.h"

#include "schwaemm192192_ref/api.h"
#include "schwaemm192192_simd/api.h"

#include "schwaemm256128_ref/api.h"
#include "schwaemm256128_simd/api.h"

#include "schwaemm256256_ref/api.h"
#include "schwaemm256256_simd/api.h"

#define ESCH_MAX_MESSAGE_LENGTH 1024
#define SCHWAEMM_MAX_MESSAGE_LENGTH 32
#define MAX_ASSOCIATED_DATA_LENGTH 32
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

void print_uchar_arr(const UChar* arr, ULLInt length) {
    for (int i = 0; i < length; i++) {
        printf("%02X", arr[i]);
    }
}

void time_schwaemm(
    unsigned int num_runs,
    ULLInt input_len,
    ULLInt output_len,
    ULLInt key_len,
    ULLInt nonce_len,
    ULLInt ad_len,
    int (*ref_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k),
    int (*ref_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k),
    int (*simd_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k),
    int (*simd_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k)
) {
    UChar* plaintext = malloc(input_len * sizeof(UChar));
    for (int i = 0; i < input_len; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar* key = malloc(key_len * sizeof(UChar));
    for (int i = 0; i < key_len; i++) {
        key[i] = (UChar)i;
    }

    UChar* nonce = malloc(nonce_len * sizeof(UChar));
    for (int i = 0; i < nonce_len; i++) {
        nonce[i] = (UChar)i;
    }

    UChar* associated_data = malloc(ad_len * sizeof(UChar));
    for (int i = 0; i < ad_len; i++) {
        associated_data[i] = (UChar)i;
    }

    UChar** ref_enc_results = malloc(num_runs * sizeof(UChar*));
    UChar** ref_dec_results = malloc(num_runs * sizeof(UChar*));
    UChar** simd_enc_results = malloc(num_runs * sizeof(UChar*));
    UChar** simd_dec_results = malloc(num_runs * sizeof(UChar*));
    for (ULLInt i = 0; i < num_runs; i++) {
        ref_enc_results[i] = malloc(output_len * sizeof(UChar));
        ref_dec_results[i] = malloc(input_len * sizeof(UChar));
        simd_enc_results[i] = malloc(output_len * sizeof(UChar));
        simd_dec_results[i] = malloc(input_len * sizeof(UChar));
    }

    ULLInt ciphertext_len = 0;

    LARGE_INTEGER start, end, elapsed_microseconds;
    LARGE_INTEGER frequency;

    //
    // Reference encryption
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        ref_enc_function(ref_enc_results[i], &ciphertext_len, plaintext, input_len, associated_data, ad_len,
            NULL, nonce, key);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER ref_enc_time = elapsed_microseconds;

    //
    // Reference decryption
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        ref_dec_function(ref_dec_results[i], &ciphertext_len, NULL, ref_enc_results[i], output_len,
            associated_data, ad_len, nonce, key);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER ref_dec_time = elapsed_microseconds;

    //
    // SIMD encryption
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        simd_enc_function(simd_enc_results[i], &ciphertext_len, plaintext, input_len, associated_data,
            ad_len, NULL, nonce, key);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER simd_enc_time = elapsed_microseconds;

    //
    // SIMD decryption
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        ref_dec_function(simd_dec_results[i], &ciphertext_len, NULL, simd_enc_results[i], output_len,
            associated_data, ad_len, nonce, key);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER simd_dec_time = elapsed_microseconds;

    //
    // Validation
    //

    int do_results_match = 1;

    for (unsigned int i = 0; i < num_runs; i++) {
        int ref_decryption_does_match = memcmp(ref_dec_results[i], plaintext, input_len) == 0;
        int simd_decryption_does_match = memcmp(simd_dec_results[i], plaintext, input_len) == 0;
        int encryption_results_match = memcmp(ref_enc_results[i], simd_enc_results[i], output_len) == 0;

        if (!ref_decryption_does_match || !simd_decryption_does_match || !encryption_results_match) {
            do_results_match = 0;
        }
    }

    printf("\nReference time (enc): %"PRId64"us\n", ref_enc_time.QuadPart);
    printf("Reference time (dec): %"PRId64"us\n", ref_dec_time.QuadPart);
    printf("Optimised time (enc): %"PRId64"us\n", simd_enc_time.QuadPart);
    printf("Optimised time (dec): %"PRId64"us\n", simd_dec_time.QuadPart);
    printf("Results %s\n", do_results_match ? "match." : "do not match!");

    //
    // Tidy up
    //

    free(key);
    free(nonce);
    free(associated_data);

    for (int i = 0; i < num_runs; i++) {
        free(ref_enc_results[i]);
        free(ref_dec_results[i]);
        free(simd_enc_results[i]);
        free(simd_dec_results[i]);
    }

    free(ref_enc_results);
    free(ref_dec_results);
    free(simd_enc_results);
    free(simd_dec_results);
}

int verify_schwaemm(
    int debug,
    ULLInt output_len,
    ULLInt key_len,
    ULLInt nonce_len,
    int (*ref_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k),
    int (*ref_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k),
    int (*simd_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k),
    int (*simd_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k)
){
    int validation_result = 1;

    UChar plaintext[SCHWAEMM_MAX_MESSAGE_LENGTH];
    UChar decryption_scratch[SCHWAEMM_MAX_MESSAGE_LENGTH];
    for (int i = 0; i < SCHWAEMM_MAX_MESSAGE_LENGTH; i++) {
        plaintext[i] = (UChar)i;
        decryption_scratch[i] = (UChar)0;
    }

    UChar* key = malloc(key_len * sizeof(UChar));
    for (int i = 0; i < key_len; i++) {
        key[i] = (UChar)i;
    }

    UChar* nonce = malloc(nonce_len * sizeof(UChar));
    for (int i = 0; i < nonce_len; i++) {
        nonce[i] = (UChar)i;
    }

    UChar associated_data[MAX_ASSOCIATED_DATA_LENGTH];
    for (int i = 0; i < MAX_ASSOCIATED_DATA_LENGTH; i++) {
        associated_data[i] = (UChar)i;
    }

    const unsigned int results_count = (SCHWAEMM_MAX_MESSAGE_LENGTH + 1) * (MAX_ASSOCIATED_DATA_LENGTH + 1);

    UChar** ref_results = malloc(results_count * sizeof(UChar*));
    UChar** simd_results = malloc(results_count * sizeof(UChar*));

    for (ULLInt plaintext_len = 0; plaintext_len <= SCHWAEMM_MAX_MESSAGE_LENGTH; plaintext_len++) {
        for (ULLInt ad_len = 0; ad_len <= MAX_ASSOCIATED_DATA_LENGTH; ad_len++) {
            const int index = (plaintext_len * (MAX_ASSOCIATED_DATA_LENGTH + 1)) + ad_len;
            ULLInt ciphertext_len = 0;
            ULLInt decrypted_ciphertext_len = 0;
            
            // Allocate and obtain a result from running the reference function.
            ref_results[index] = malloc(output_len * sizeof(UChar));
            ref_enc_function(ref_results[index], &ciphertext_len, plaintext, plaintext_len,
                associated_data, ad_len, NULL, nonce, key);
            ref_dec_function(decryption_scratch, &decrypted_ciphertext_len, NULL, ref_results[index],
                ciphertext_len, associated_data, ad_len, nonce, key);

            int does_decryption_match = memcmp(plaintext, decryption_scratch, plaintext_len) == 0;

            if (!does_decryption_match) {
                validation_result = 0;
            }

            if (debug) {
                printf("[%d] Ref (enc) = ", index);
                print_uchar_arr(ref_results[index], ciphertext_len);
                printf("\n[%d] Ref (dec) = ", index);
                print_uchar_arr(decryption_scratch, plaintext_len);
                printf("\nDecryption %s\n", does_decryption_match ? "matches." : "does not match!");
            }

            // Allocate and obtain a result from running the SIMD-optimised function.
            simd_results[index] = malloc(output_len * sizeof(UChar));
            simd_enc_function(simd_results[index], &ciphertext_len, plaintext, plaintext_len,
                associated_data, ad_len, NULL, nonce, key);
            simd_dec_function(decryption_scratch, &decrypted_ciphertext_len, NULL, simd_results[index],
                ciphertext_len, associated_data, ad_len, nonce, key);

            does_decryption_match = memcmp(plaintext, decryption_scratch, plaintext_len) == 0;
            int do_ciphertexts_match = memcmp(ref_results[index], simd_results[index], ciphertext_len) == 0;

            if (!do_ciphertexts_match || !does_decryption_match) {
                validation_result = 0;
            }

            if (debug) {
                printf("[%d] Opt (enc) = ", index);
                print_uchar_arr(simd_results[index], ciphertext_len);
                printf("\n[%d] Opt (dec) = ", index);
                print_uchar_arr(decryption_scratch, plaintext_len);
                printf("\nDecryption %s\n\n", does_decryption_match ? "matches." : "does not match!");
            }
        }
    }

    //
    // Tidy up
    //

    free(key);
    free(nonce);

    for (int i = 0; i < SCHWAEMM_MAX_MESSAGE_LENGTH + 1; i++) {
        free(ref_results[i]);
        free(simd_results[i]);
    }

    free(ref_results);
    free(simd_results);

    return validation_result;
}

void time_esch(
    unsigned int num_runs,
    ULLInt input_len,
    ULLInt output_len,
    int (*ref_function)(UChar* out, const UChar* in, ULLInt inlen),
    int (*simd_function)(UChar* out, const UChar* in, ULLInt inlen)
) {
    UChar* plaintext = malloc(input_len * sizeof(UChar));
    for (int i = 0; i < input_len; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar** ref_results = malloc(num_runs * sizeof(UChar*));
    UChar** simd_results = malloc(num_runs * sizeof(UChar*));
    for (ULLInt i = 0; i < num_runs; i++) {
        ref_results[i] = malloc(output_len * sizeof(UChar));
        simd_results[i] = malloc(output_len * sizeof(UChar));
    }

    LARGE_INTEGER start, end, elapsed_microseconds;
    LARGE_INTEGER frequency;

    //
    // Reference
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        ref_function(ref_results[i], plaintext, input_len);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER ref_time = elapsed_microseconds;
    
    //
    // SIMD
    //

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    for (unsigned int i = 0; i < num_runs; i++) {
        simd_function(simd_results[i], plaintext, input_len);
    }

    QueryPerformanceCounter(&end);
    elapsed_microseconds.QuadPart = end.QuadPart - start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= frequency.QuadPart;

    LARGE_INTEGER simd_time = elapsed_microseconds;

    //
    // Validation
    //

    int do_results_match = 1;

    for (unsigned int i = 0; i < num_runs; i++) {
        int result = memcmp(ref_results[i], simd_results[i], output_len);
        if (result != 0) {
            do_results_match = 0;
        }
    }

    printf("\nReference time: %"PRId64"us\n", ref_time.QuadPart);
    printf("Optimised time: %"PRId64"us\n", simd_time.QuadPart);
    printf("Results %s\n", do_results_match ? "match." : "do not match!");

    //
    // Tidy up
    //

    free(plaintext);
   
    for (unsigned int i = 0; i < num_runs; i++) {
        free(ref_results[i]);
        free(simd_results[i]);
    }

    free(ref_results);
    free(simd_results);
}

int verify_esch(
    int debug,
    ULLInt output_len,
    int (*ref_function)(UChar* out, const UChar* in, ULLInt inlen),
    int (*simd_function)(UChar* out, const UChar* in, ULLInt inlen)
) {
    int validation_result = 1;
    
    UChar plaintext[ESCH_MAX_MESSAGE_LENGTH];
    for (int i = 0; i < ESCH_MAX_MESSAGE_LENGTH; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar** ref_results = malloc((ESCH_MAX_MESSAGE_LENGTH + 1) * sizeof(UChar *));
    UChar** simd_results = malloc((ESCH_MAX_MESSAGE_LENGTH + 1) * sizeof(UChar*));

    for (int i = 0; i <= ESCH_MAX_MESSAGE_LENGTH; i++) {
        // Allocate and obtain a result from running the reference function.
        ref_results[i] = malloc(output_len * sizeof(UChar));
        ref_function(ref_results[i], plaintext, i);

        // Allocate and obtain a result from running the SIMD-optimised function.
        simd_results[i] = malloc(output_len * sizeof(UChar));
        simd_function(simd_results[i], plaintext, i);

        int do_hashes_match = memcmp(ref_results[i], simd_results[i], output_len) == 0;

        if (!do_hashes_match) {
            validation_result = 0;
        }

        if (debug) {
            printf("[%d] Ref = ", i);
            print_uchar_arr(ref_results[i], output_len);
            printf("\n[%d] Opt = ", i);
            print_uchar_arr(simd_results[i], output_len);
            printf("\nHashes %s\n\n", do_hashes_match ? "match." : "do not match!");
        }
    }

    //
    // Tidy up
    //

    for (int i = 0; i <= ESCH_MAX_MESSAGE_LENGTH; i++) {
        free(ref_results[i]);
        free(simd_results[i]);
    }

    free(ref_results);
    free(simd_results);

    return validation_result;
}

int main()
{
    int esch256_validation = verify_esch(
        0,
        E256REF_CRYPTO_BYTES,
        e256ref_crypto_hash,
        e256simd_crypto_hash);

    printf("Esch256 validation: %s\n", esch256_validation ? "SUCCESS" : "FAILURE");

    int esch384_validation = verify_esch(
        0,
        E384REF_CRYPTO_BYTES,
        e384ref_crypto_hash,
        e384simd_crypto_hash);

    printf("Esch2384 validation: %s\n", esch384_validation ? "SUCCESS" : "FAILURE");

    int s128128_validation = verify_schwaemm(
        0,
        SCHWAEMM_MAX_MESSAGE_LENGTH + S128128REF_CRYPTO_ABYTES,
        S128128REF_CRYPTO_KEYBYTES,
        S128128REF_CRYPTO_NPUBBYTES,
        s128128ref_crypto_aead_encrypt,
        s128128ref_crypto_aead_decrypt,
        s128128simd_crypto_aead_encrypt,
        s128128simd_crypto_aead_decrypt);

    printf("Schwaemm128_128 validation: %s\n", s128128_validation ? "SUCCESS" : "FAILURE");

    int s192192_validation = verify_schwaemm(
        0,
        SCHWAEMM_MAX_MESSAGE_LENGTH + S192192REF_CRYPTO_ABYTES,
        S192192REF_CRYPTO_KEYBYTES,
        S192192REF_CRYPTO_NPUBBYTES,
        s192192ref_crypto_aead_encrypt,
        s192192ref_crypto_aead_decrypt,
        s192192simd_crypto_aead_encrypt,
        s192192simd_crypto_aead_decrypt);

    printf("Schwaemm192_192 validation: %s\n", s192192_validation ? "SUCCESS" : "FAILURE");

    int s256128_validation = verify_schwaemm(
        0,
        SCHWAEMM_MAX_MESSAGE_LENGTH + S256128REF_CRYPTO_ABYTES,
        S256128REF_CRYPTO_KEYBYTES,
        S256128REF_CRYPTO_NPUBBYTES,
        s256128ref_crypto_aead_encrypt,
        s256128ref_crypto_aead_decrypt,
        s256128simd_crypto_aead_encrypt,
        s256128simd_crypto_aead_decrypt);

    printf("Schwaemm256_128 validation: %s\n", s256128_validation ? "SUCCESS" : "FAILURE");

    int s256256_validation = verify_schwaemm(
        0,
        SCHWAEMM_MAX_MESSAGE_LENGTH + S256256REF_CRYPTO_ABYTES,
        S256256REF_CRYPTO_KEYBYTES,
        S256256REF_CRYPTO_NPUBBYTES,
        s256256ref_crypto_aead_encrypt,
        s256256ref_crypto_aead_decrypt,
        s256256simd_crypto_aead_encrypt,
        s256256simd_crypto_aead_decrypt);

    printf("Schwaemm256_256 validation: %s\n", s256256_validation ? "SUCCESS" : "FAILURE");

    printf("\nEsch384 Timing:\n");
    for (unsigned int i = 0; i < 8; i++) {
        time_esch(1, 1 << 12, E384REF_CRYPTO_BYTES, e384ref_crypto_hash, e384simd_crypto_hash);
    }

    printf("\nSchwaemm256_256 Timing:\n");
    for (unsigned int i = 0; i < 8; i++) {
        time_schwaemm(
            1,
            1 << 12,
            (1 << 12) + S256256REF_CRYPTO_ABYTES,
            S256256REF_CRYPTO_KEYBYTES,
            S256256REF_CRYPTO_NPUBBYTES,
            32,
            s256256ref_crypto_aead_encrypt,
            s256256ref_crypto_aead_decrypt,
            s256256simd_crypto_aead_encrypt,
            s256256simd_crypto_aead_decrypt);
    }

    return 1;
}