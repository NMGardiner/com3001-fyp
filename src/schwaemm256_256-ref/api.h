#include "toggle_defines.h"

#if ENABLE_SCHWAEMM256_256

#define CRYPTO_KEYBYTES 32
#define CRYPTO_NSECBYTES 0
#define CRYPTO_NPUBBYTES 32
#define CRYPTO_ABYTES 32
#define CRYPTO_NOOVERLAP 1

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
#define MAX_FILE_NAME              256
#define MAX_MESSAGE_LENGTH         32
#define MAX_ASSOCIATED_DATA_LENGTH 32

typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int crypto_aead_encrypt(UChar* c, ULLInt* clen, const UChar* m, ULLInt mlen, \
	const UChar* ad, ULLInt adlen, const UChar* nsec, const UChar* npub, \
	const UChar* k);
//---------------------------------------------------------


#endif // ENABLE_SCHWAEMM256_256