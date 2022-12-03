#pragma once

#define ENABLE_ESCH384 0
#define ENABLE_SCHWAEMM256_256 1

#if ENABLE_ESCH384
#define ENABLE_SPARKLE 1
#elif ENABLE_SCHWAEMM256_256
#define ENABLE_SPARKLE 1
#else
#define ENABLE_SPARKLE 0
#endif