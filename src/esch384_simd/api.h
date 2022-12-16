#define E384SIMD_CRYPTO_BYTES 48

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int e384simd_crypto_hash(UChar* out, const UChar* in, ULLInt inlen);
//---------------------------------------------------------