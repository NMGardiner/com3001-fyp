#if defined(__ARM_NEON) // Path for Arm NEON
#define USE_NEON 1
#define USE_AVX2 0
#elif defined(__AVX2__)  // Path for AVX2
#define USE_NEON 0
#define USE_AVX2 1
#else
#define USE_NEON 0
#define USE_AVX2 0
#endif