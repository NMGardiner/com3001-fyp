#define S192192SIMD_CRYPTO_KEYBYTES 24
#define S192192SIMD_CRYPTO_NSECBYTES 0
#define S192192SIMD_CRYPTO_NPUBBYTES 24
#define S192192SIMD_CRYPTO_ABYTES 24
#define S192192SIMD_CRYPTO_NOOVERLAP 1

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
int s192192simd_crypto_aead_encrypt(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
	const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
	const UChar* k);

int s192192simd_crypto_aead_decrypt(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
	ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
	const UChar* k);
//---------------------------------------------------------