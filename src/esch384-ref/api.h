#include "toggle_defines.h"

#if ENABLE_ESCH384

#define CRYPTO_BYTES 48

//---------------------------------------------------------
// Added for use in the testbench.
//---------------------------------------------------------
typedef unsigned char UChar;
typedef unsigned long long int ULLInt;

int crypto_hash_ref(UChar* out, const UChar* in, ULLInt inlen);
//---------------------------------------------------------

#endif // ENABLE_ESCH384