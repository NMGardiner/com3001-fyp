#if defined(__arm__) && defined(__linux__) // Path for Arm NEON
#define USE_NEON 1
#define USE_AVX2 0
#elif defined(_WIN32)  // Path for AVX2
#define USE_NEON 0
#define USE_AVX2 1
#endif 