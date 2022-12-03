#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include <immintrin.h>

#include "toggle_defines.h"

#if ENABLE_ESCH384
#include "esch384-ref/api.h"
#include "esch384-simd/api.h"

// Do we really need this?
#define MAX_MESSAGE_LENGTH 4096
#endif // ENABLE_ESCH384

#if ENABLE_SCHWAEMM256_256
#include "schwaemm256_256-ref/api.h"
#endif // ENABLE_SCHWAEMM256_256

#if ENABLE_ESCH384
void esch384_time_execution(void);
#endif // ENABLE_ESCH384

#if ENABLE_SCHWAEMM256_256
void schwaemm256_256_time_execution(void);
#endif // ENABLE_SCHWAEMM256_256

int main()
{
#if ENABLE_ESCH384
    // Time the Esch384 algorithms 10 times, to get an average.
    for (int i = 0; i < 10; i++) {
       esch384_time_execution();
    }
#endif // ENABLE_ESCH384

#if ENABLE_SCHWAEMM256_256
    schwaemm256_256_time_execution();
#endif // ENABLE_SCHWAEMM256_256

    return 1;
}

#if ENABLE_ESCH384
void print_hash(const UChar* hash, ULLInt length) {
    for (int i = 0; i < length; i++) {
        printf("%02X", hash[i]);
    }
}

// The setup for hashing is taken from esch384-ref/genkat_hash.c
void esch384_time_execution(void) {
    // Generate a test message.
    UChar test_message[MAX_MESSAGE_LENGTH];
    for (int i = 0; i < MAX_MESSAGE_LENGTH; i++) {
        test_message[i] = (UChar)i;
    }

    UChar ref_hash_outputs[MAX_MESSAGE_LENGTH + 1][CRYPTO_BYTES];
    UChar opt_hash_outputs[MAX_MESSAGE_LENGTH + 1][CRYPTO_BYTES];

    // Time the reference implementation.
    clock_t start_time_ref = clock();
    for (int i = 0; i < MAX_MESSAGE_LENGTH + 1; i++) {
        crypto_hash_ref(ref_hash_outputs[i], test_message, i);
        //printf("H(R) = ");
        //print_hash(ref_hash_outputs[i], CRYPTO_BYTES);
        //printf("\n");
    }

    clock_t end_time_ref = clock();
    double execution_time_ref = (double)(end_time_ref - start_time_ref);

    // Time the SIMD optimised implementation.
    clock_t start_time_opt = clock();
    for (int i = 0; i < MAX_MESSAGE_LENGTH + 1; i++) {
        crypto_hash_simd(opt_hash_outputs[i], test_message, i);
        //printf("H(O) = ");
        //print_hash(opt_hash_outputs[i], CRYPTO_BYTES);
        //printf("\n");
    }

    clock_t end_time_opt = clock();
    double execution_time_opt = (double)(end_time_opt - start_time_opt);

    printf("Timed Esch384 algorithms. Ref = %f ticks, Opt = %f ticks.\n", execution_time_ref, execution_time_opt);

    // Validate the outputs;
    int hashes_match = 1;
    for (int i = 0; i < MAX_MESSAGE_LENGTH + 1; i++) {
        int this_hash_matches = 1;
        for (int j = 0; j < CRYPTO_BYTES; j++) {
            if (ref_hash_outputs[i][j] != opt_hash_outputs[i][j]) {
                this_hash_matches = 0;
            }
        }
        if (!this_hash_matches) {
            printf("ERR - Hash %d does not match! - ", i);
            print_hash(ref_hash_outputs[i], CRYPTO_BYTES);
            printf(" vs ");
            print_hash(opt_hash_outputs[i], CRYPTO_BYTES);
            printf("\n");
            hashes_match = 0;
        }
    }
    if (!hashes_match) {
        printf("One or more hashes do not match! Check prior output!\n");
    }
    /*else {
        printf("Hash validation successful.\n");
    }*/
}
#endif // ENABLE_ESCH384

#if ENABLE_SCHWAEMM256_256
void print_ciphertext(const UChar* ciphertext, ULLInt length) {
    for (int i = 0; i < length; i++) {
        printf("%02X", ciphertext[i]);
    }
}

// The setup for encryption is taken from schwaemm256_256-ref/genkat_aead.c
void schwaemm256_256_time_execution(void) {
    UChar key[CRYPTO_KEYBYTES];
    for (int i = 0; i < sizeof(key); i++) {
        key[i] = (UChar)i;
    }

    UChar nonce[CRYPTO_NPUBBYTES];
    for (int i = 0; i < sizeof(nonce); i++) {
        nonce[i] = (UChar)i;
    }

    UChar plaintext[MAX_MESSAGE_LENGTH];
    for (int i = 0; i < sizeof(plaintext); i++) {
        plaintext[i] = (UChar)i;
    }

    UChar associated_data[MAX_ASSOCIATED_DATA_LENGTH];
    for (int i = 0; i < sizeof(associated_data); i++) {
        associated_data[i] = (UChar)i;
    }

    UChar ciphertext[MAX_MESSAGE_LENGTH + CRYPTO_ABYTES];
    ULLInt ciphertext_len = 0;

    for (ULLInt plaintext_len = 0; plaintext_len <= MAX_MESSAGE_LENGTH; plaintext_len++) {
        for (ULLInt ad_len = 0; ad_len <= MAX_ASSOCIATED_DATA_LENGTH; ad_len++) {
            int ret_val = crypto_aead_encrypt(ciphertext, &ciphertext_len, plaintext,
                plaintext_len, associated_data, ad_len, NULL, nonce, key);

            printf("Ciphertext = ");
            print_ciphertext(ciphertext, ciphertext_len);
            printf("\n");
        }
    }
}
#endif // ENABLE_SCHWAEMM256_256