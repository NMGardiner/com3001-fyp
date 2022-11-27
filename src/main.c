#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include <immintrin.h>

#include "esch384-ref/api.h"
#include "esch384-simd/api.h"

#define MAX_MESSAGE_LENGTH 4096

void esch384_time_execution(void);

void print_hash(const UChar* hash, ULLInt length) {
    for (int i = 0; i < length; i++) {
        printf("%02X", hash[i]);
    }
}

int main()
{
    // Time the Esch384 algorithms 10 times, to get an average.
    for (int i = 0; i < 10; i++) {
       esch384_time_execution();
    }

    return 1;
}

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