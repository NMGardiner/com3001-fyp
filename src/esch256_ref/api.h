#define E256REF_CRYPTO_BYTES 32

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int e256ref_crypto_hash(UChar* out, const UChar* in, ULLInt inlen);
//---------------------------------------------------------