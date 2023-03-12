#define PJ128REF_CRYPTO_KEYBYTES     16
#define PJ128REF_CRYPTO_NSECBYTES    0
#define PJ128REF_CRYPTO_NPUBBYTES    12
#define PJ128REF_CRYPTO_ABYTES       16
#define PJ128REF_CRYPTO_NOOVERLAP    1

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
int pj128ref_crypto_aead_encrypt(
	unsigned char* c, unsigned long long* clen,
	const unsigned char* m, unsigned long long mlen,
	const unsigned char* ad, unsigned long long adlen,
	const unsigned char* nsec,
	const unsigned char* npub,
	const unsigned char* k);

int pj128ref_crypto_aead_decrypt(
	unsigned char* m, unsigned long long* mlen,
	unsigned char* nsec,
	const unsigned char* c, unsigned long long clen,
	const unsigned char* ad, unsigned long long adlen,
	const unsigned char* npub,
	const unsigned char* k);
//---------------------------------------------------------