#define E256SIMD_CRYPTO_BYTES 32

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int e256simd_crypto_hash(UChar* out, const UChar* in, ULLInt inlen);
//---------------------------------------------------------