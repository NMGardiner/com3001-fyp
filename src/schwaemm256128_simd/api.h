#define S256128SIMD_CRYPTO_KEYBYTES 16
#define S256128SIMD_CRYPTO_NSECBYTES 0
#define S256128SIMD_CRYPTO_NPUBBYTES 32
#define S256128SIMD_CRYPTO_ABYTES 16
#define S256128SIMD_CRYPTO_NOOVERLAP 1

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
int s256128simd_crypto_aead_encrypt(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
	const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
	const UChar* k);

int s256128simd_crypto_aead_decrypt(UChar* m, ULLInt* mlen, UChar* nsec, const UChar* c, \
	ULLInt clen, const UChar* ad, ULLInt adlen, const UChar* npub, \
	const UChar* k);
//---------------------------------------------------------