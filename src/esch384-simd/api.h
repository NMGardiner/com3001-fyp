#define CRYPTO_BYTES 48



// 
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int crypto_hash_simd(UChar* out, const UChar* in, ULLInt inlen);