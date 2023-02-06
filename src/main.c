#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <Windows.h>
#endif // defined(_WIN32)

#include <inttypes.h>
#include <string.h>
#include <sys/time.h>

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

#define NUM_RESULTS 20

//
// A timer abstraction for working with MSVC and GCC
//

struct timer {
#if defined(_WIN32)
    LARGE_INTEGER start;
    LARGE_INTEGER frequency;
#else
    struct timeval start;
#endif
};

void start_timer(struct timer* t) {
#if defined(_WIN32)
    QueryPerformanceFrequency(&(t->frequency));
    QueryPerformanceCounter(&(t->start));
#else
    gettimeofday(&(t->start), NULL);
#endif
}

uint64_t end_timer(struct timer* t) {
#if defined(_WIN32)
    LARGE_INTEGER end;
    QueryPerformanceCounter(&end);
    LARGE_INTEGER elapsed_microseconds;
    elapsed_microseconds.QuadPart = end.QuadPart - t->start.QuadPart;
    elapsed_microseconds.QuadPart *= 1000000;
    elapsed_microseconds.QuadPart /= t->frequency.QuadPart;
    return elapsed_microseconds.QuadPart;
#else
    struct timeval end;
    gettimeofday(&end, NULL);
    uint64_t elapsed_microseconds = ((end.tv_sec - t->start.tv_sec) * 1000000) + (end.tv_usec - t->start.tv_usec);
    return elapsed_microseconds;
#endif
}

struct schwaemm_variant {
    const char* name;
    ULLInt key_len;
    ULLInt ad_crypt_len;
    ULLInt nonce_len;
    int (*ref_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k);
    int (*ref_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k);
    int (*simd_enc_function)(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
        const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
        const UChar* k);
    int (*simd_dec_function)(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
        ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
        const UChar* k);
};

struct esch_variant {
    const char* name;
    ULLInt output_len;
    int (*ref_function)(UChar* out, const UChar* in, ULLInt inlen);
    int (*simd_function)(UChar* out, const UChar* in, ULLInt inlen);
};

void print_uchar_arr(const UChar* arr, ULLInt length) {
    for (unsigned int i = 0; i < length; i++) {
        printf("%02X", arr[i]);
    }
}

void print_results_array(long long* arr, unsigned int in_len, const char* label) {
    printf("%s = [ ", label);
    for (unsigned int i = 0; i < in_len; i++) {
        printf("%s%lldus", i ? ", " : "", arr[i]);
    }
    printf(" ]\n");
}

void print_double_array(double *arr, unsigned int in_len, const char* label) {
    printf("%s = [ ", label);
    for (unsigned int i = 0; i < in_len; i++) {
        printf("%s%.1f%%", i ? ", " : "", arr[i]);
    }
    printf(" ]\n");
}

double calculate_average(double* arr, unsigned int in_len) {
    double result = 0.0f;
    unsigned int discard_count = 0;
    for (unsigned int i = 0; i < in_len; i++) {
        // Ignore results greater than 50%, as they are almost definitely outliers.
        if (arr[i] >= 50.0f || arr[i] <= -50.0f) {
            discard_count++;
        } else {
            result += arr[i];
        }
    }
    printf("(Avg discarded %d results)\n", discard_count);
    return result / (double)(in_len - discard_count);
}

int verify_schwaemm(int debug, struct schwaemm_variant* variant) {
    int validation_result = 1;

    ULLInt output_len = SCHWAEMM_MAX_MESSAGE_LENGTH + variant->ad_crypt_len;

    UChar plaintext[SCHWAEMM_MAX_MESSAGE_LENGTH];
    UChar decryption_scratch[SCHWAEMM_MAX_MESSAGE_LENGTH];
    for (unsigned int i = 0; i < SCHWAEMM_MAX_MESSAGE_LENGTH; i++) {
        plaintext[i] = (UChar)i;
        decryption_scratch[i] = (UChar)0;
    }

    UChar* key = malloc(variant->key_len * sizeof(UChar));
    for (unsigned int i = 0; i < variant->key_len; i++) {
        key[i] = (UChar)i;
    }

    UChar* nonce = malloc(variant->nonce_len * sizeof(UChar));
    for (unsigned int i = 0; i < variant->nonce_len; i++) {
        nonce[i] = (UChar)i;
    }

    UChar associated_data[MAX_ASSOCIATED_DATA_LENGTH];
    for (unsigned int i = 0; i < MAX_ASSOCIATED_DATA_LENGTH; i++) {
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
            variant->ref_enc_function(ref_results[index], &ciphertext_len, plaintext, plaintext_len,
                associated_data, ad_len, NULL, nonce, key);
            variant->ref_dec_function(decryption_scratch, &decrypted_ciphertext_len, NULL, ref_results[index],
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
            variant->simd_enc_function(simd_results[index], &ciphertext_len, plaintext, plaintext_len,
                associated_data, ad_len, NULL, nonce, key);
            variant->simd_dec_function(decryption_scratch, &decrypted_ciphertext_len, NULL, simd_results[index],
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

void time_schwaemm(unsigned int num_runs, ULLInt input_len, ULLInt ad_len, struct schwaemm_variant* variant) {
    ULLInt output_len = input_len + variant->ad_crypt_len;

    UChar* plaintext = malloc(input_len * sizeof(UChar));
    for (unsigned int i = 0; i < input_len; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar* key = malloc(variant->key_len * sizeof(UChar));
    for (unsigned int i = 0; i < variant->key_len; i++) {
        key[i] = (UChar)i;
    }

    UChar* nonce = malloc(variant->nonce_len * sizeof(UChar));
    for (unsigned int i = 0; i < variant->nonce_len; i++) {
        nonce[i] = (UChar)i;
    }

    UChar* associated_data = malloc(ad_len * sizeof(UChar));
    for (unsigned int i = 0; i < ad_len; i++) {
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

    long long* ref_enc_timings  = malloc(NUM_RESULTS * sizeof(long long));
    long long* ref_dec_timings  = malloc(NUM_RESULTS * sizeof(long long));
    long long* simd_enc_timings  = malloc(NUM_RESULTS * sizeof(long long));
    long long* simd_dec_timings  = malloc(NUM_RESULTS * sizeof(long long));

    ULLInt ciphertext_len = 0;

    for (ULLInt i = 0; i < num_runs; i++) {
        struct timer t;

        //
        // Reference (Encryption)
        //

        start_timer(&t);

        variant->ref_enc_function(ref_enc_results[i], &ciphertext_len, plaintext,
            input_len, associated_data, ad_len, NULL, nonce, key);

        ref_enc_timings[i] = end_timer(&t);

        //
        // Reference (Decryption)
        //

        start_timer(&t);

        variant->ref_dec_function(ref_dec_results[i], &ciphertext_len, NULL,
            ref_enc_results[i], output_len, associated_data, ad_len, nonce, key);

        ref_dec_timings[i] = end_timer(&t);

        //
        // SIMD (Encryption)
        //

        start_timer(&t);

        variant->simd_enc_function(simd_enc_results[i], &ciphertext_len, plaintext,
            input_len, associated_data, ad_len, NULL, nonce, key);

        simd_enc_timings[i] = end_timer(&t);

        //
        // SIMD (Decryption)
        //

        start_timer(&t);

        variant->simd_dec_function(simd_dec_results[i], &ciphertext_len, NULL,
            ref_enc_results[i], output_len, associated_data, ad_len, nonce, key);

        simd_dec_timings[i] = end_timer(&t);
    }

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

    int does_validation_pass = verify_schwaemm(0, variant);

    printf("\n%s:\n", variant->name);
    if (do_results_match) {
        print_results_array(ref_enc_timings, NUM_RESULTS, "ref_e");
        print_results_array(simd_enc_timings, NUM_RESULTS, "opt_e");

        double diffs[NUM_RESULTS];
        for (int i = 0; i < NUM_RESULTS; i++) {
            long long diff = ref_enc_timings[i] - simd_enc_timings[i];
            diffs[i] = ((double)diff / ref_enc_timings[i]) * 100.0f;
        }
        print_double_array(diffs, NUM_RESULTS, "%diff");
        printf("Avg = %.2f%%\n", calculate_average(diffs, NUM_RESULTS));

        print_results_array(ref_dec_timings, NUM_RESULTS, "ref_d");
        print_results_array(simd_dec_timings, NUM_RESULTS, "opt_d");

        for (int i = 0; i < NUM_RESULTS; i++) {
            long long diff = ref_dec_timings[i] - simd_dec_timings[i];
            diffs[i] = ((double)diff / ref_dec_timings[i]) * 100.0f;
        }
        print_double_array(diffs, NUM_RESULTS, "%diff");
        printf("Avg = %.2f%%\n", calculate_average(diffs, NUM_RESULTS));
    } else {
        printf("[ERROR] %s timing yielded incorrect results!\n", variant->name);
    }

    printf("Validation: %s\n", does_validation_pass ? "SUCCESS" : "FAILURE");

    //
    // Tidy up
    //

    free(key);
    free(nonce);
    free(associated_data);

    for (unsigned int i = 0; i < num_runs; i++) {
        free(ref_enc_results[i]);
        free(ref_dec_results[i]);
        free(simd_enc_results[i]);
        free(simd_dec_results[i]);
    }

    free(ref_enc_results);
    free(ref_dec_results);
    free(simd_enc_results);
    free(simd_dec_results);

    free(ref_enc_timings);
    free(ref_dec_timings);
    free(simd_enc_timings);
    free(simd_dec_timings);
}

int verify_esch(int debug, struct esch_variant* variant) {
    int validation_result = 1;
    
    UChar plaintext[ESCH_MAX_MESSAGE_LENGTH];
    for (int i = 0; i < ESCH_MAX_MESSAGE_LENGTH; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar** ref_results = malloc((ESCH_MAX_MESSAGE_LENGTH + 1) * sizeof(UChar *));
    UChar** simd_results = malloc((ESCH_MAX_MESSAGE_LENGTH + 1) * sizeof(UChar*));

    for (int i = 0; i <= ESCH_MAX_MESSAGE_LENGTH; i++) {
        // Allocate and obtain a result from running the reference function.
        ref_results[i] = malloc(variant->output_len * sizeof(UChar));
        variant->ref_function(ref_results[i], plaintext, i);

        // Allocate and obtain a result from running the SIMD-optimised function.
        simd_results[i] = malloc(variant->output_len * sizeof(UChar));
        variant->simd_function(simd_results[i], plaintext, i);

        int do_hashes_match = memcmp(ref_results[i], simd_results[i], variant->output_len) == 0;

        if (!do_hashes_match) {
            validation_result = 0;
        }

        if (debug) {
            printf("[%d] Ref = ", i);
            print_uchar_arr(ref_results[i], variant->output_len);
            printf("\n[%d] Opt = ", i);
            print_uchar_arr(simd_results[i], variant->output_len);
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

void time_esch(unsigned int num_runs, ULLInt input_len, struct esch_variant* variant) {
    UChar* plaintext = malloc(input_len * sizeof(UChar));
    for (unsigned int i = 0; i < input_len; i++) {
        plaintext[i] = (UChar)i;
    }

    UChar** ref_results = malloc(num_runs * sizeof(UChar*));
    UChar** simd_results = malloc(num_runs * sizeof(UChar*));
    for (ULLInt i = 0; i < num_runs; i++) {
        ref_results[i] = malloc(variant->output_len * sizeof(UChar));
        simd_results[i] = malloc(variant->output_len * sizeof(UChar));
    }

    long long* ref_timings  = malloc(NUM_RESULTS * sizeof(long long));
    long long* simd_timings  = malloc(NUM_RESULTS * sizeof(long long));

    for (ULLInt i = 0; i < num_runs; i++) {
        struct timer t;

        //
        // Reference
        //

        start_timer(&t);

        variant->ref_function(ref_results[i], plaintext, input_len);

        ref_timings[i] = end_timer(&t);

        //
        // SIMD
        //

        start_timer(&t);

        variant->simd_function(simd_results[i], plaintext, input_len);

        simd_timings[i] = end_timer(&t);
    }

    //
    // Validation
    //

    int do_results_match = 1;

    for (unsigned int i = 0; i < num_runs; i++) {
        int result = memcmp(ref_results[i], simd_results[i], variant->output_len);
        if (result != 0) {
            do_results_match = 0;
        }
    }

    int does_validation_pass = verify_esch(0, variant);

    printf("\n%s:\n", variant->name);
    if (do_results_match) {
        print_results_array(ref_timings, NUM_RESULTS, "ref  ");
        print_results_array(simd_timings, NUM_RESULTS, "opt  ");

        double diffs[NUM_RESULTS];
        for (int i = 0; i < NUM_RESULTS; i++) {
            long long diff = ref_timings[i] - simd_timings[i];
            diffs[i] = ((double)diff / ref_timings[i]) * 100.0f;
        }
        print_double_array(diffs, NUM_RESULTS, "%diff");
        printf("Avg = %.2f%%\n", calculate_average(diffs, NUM_RESULTS));
    } else {
        printf("[ERROR] %s timing yielded incorrect results!\n", variant->name);
    }

    printf("Validation: %s\n", does_validation_pass ? "SUCCESS" : "FAILURE");

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

    free(ref_timings);
    free(simd_timings);
}

int main()
{
    struct schwaemm_variant s128128 = {
        "Schwaemm128_128",
        S128128REF_CRYPTO_KEYBYTES,
        S128128REF_CRYPTO_ABYTES,
        S128128REF_CRYPTO_NPUBBYTES,
        s128128ref_crypto_aead_encrypt,
        s128128ref_crypto_aead_decrypt,
        s128128simd_crypto_aead_encrypt,
        s128128simd_crypto_aead_decrypt
    };

    struct schwaemm_variant s192192 = {
        "Schwaemm192_192",
        S192192REF_CRYPTO_KEYBYTES,
        S192192REF_CRYPTO_ABYTES,
        S192192REF_CRYPTO_NPUBBYTES,
        s192192ref_crypto_aead_encrypt,
        s192192ref_crypto_aead_decrypt,
        s192192simd_crypto_aead_encrypt,
        s192192simd_crypto_aead_decrypt
    };

    struct schwaemm_variant s256128 = {
        "Schwaemm256_128",
        S256128REF_CRYPTO_KEYBYTES,
        S256128REF_CRYPTO_ABYTES,
        S256128REF_CRYPTO_NPUBBYTES,
        s256128ref_crypto_aead_encrypt,
        s256128ref_crypto_aead_decrypt,
        s256128simd_crypto_aead_encrypt,
        s256128simd_crypto_aead_decrypt
    };

    struct schwaemm_variant s256256 = {
        "Schwaemm256_256",
        S256256REF_CRYPTO_KEYBYTES,
        S256256REF_CRYPTO_ABYTES,
        S256256REF_CRYPTO_NPUBBYTES,
        s256256ref_crypto_aead_encrypt,
        s256256ref_crypto_aead_decrypt,
        s256256simd_crypto_aead_encrypt,
        s256256simd_crypto_aead_decrypt
    };

    struct esch_variant e256 = {
        "Esch256",
        E256REF_CRYPTO_BYTES,
        e256ref_crypto_hash,
        e256simd_crypto_hash
    };

    struct esch_variant e384 = {
        "Esch384",
        E384REF_CRYPTO_BYTES,
        e384ref_crypto_hash,
        e384simd_crypto_hash
    };

    time_schwaemm(NUM_RESULTS, 1 << 16, 1 << 12, &s128128);
    time_schwaemm(NUM_RESULTS, 1 << 16, 1 << 12, &s192192);
    time_schwaemm(NUM_RESULTS, 1 << 16, 1 << 12, &s256128);
    time_schwaemm(NUM_RESULTS, 1 << 16, 1 << 12, &s256256);

    time_esch(NUM_RESULTS, 1 << 16, &e256);
    time_esch(NUM_RESULTS, 1 << 16, &e384);

    return 1;
}