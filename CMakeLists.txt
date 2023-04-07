cmake_minimum_required(VERSION 3.0.0)
project(com3001-fyp C)

# We check if we are compiling for ARM, in which case we enable ARM Neon.
# If we are not compiling for ARM, we enable AVX2. MSVC uses a different form for the
# compiler options, so handle that accordingly.
IF(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
	add_compile_options(-Wall -Wextra -Wshadow -mfpu=neon)
ELSEIF(MSVC)
	add_compile_options(/Wall /arch:AVX2)
ELSE()
	add_compile_options(-Wall -Wextra -Wshadow -mavx2)
ENDIF()

add_executable(testbench
	# Testbench
	"src/main.c"

	# Sparkle reference algorithm
	"src/sparkle_ref/sparkle_ref.c"

	# Sparkle SIMD-optimised algorithm
	"src/sparkle_simd/sparkle_simd.c"

	# AVX Alzette implementations
	"src/sparkle_simd/alzette_avx_impls.h"

	# Esch256 reference algorithm
	"src/esch256_ref/hash.c"

	# Esch256 SIMD-optimised algorithm
	"src/esch256_simd/hash.c"
	
	# Esch384 reference algorithm
	"src/esch384_ref/hash.c"

	# Esch384 SIMD-optimised algorithm
	"src/esch384_simd/hash.c"

	# Schwaemm128_128 reference algorithm
	"src/schwaemm128128_ref/encrypt.c"

	# Schwaemm128_128 SIMD-optimised algorithm
	"src/schwaemm128128_simd/encrypt.c"

	# Schwaemm192_192 reference algorithm
	"src/schwaemm192192_ref/encrypt.c"

	# Schwaemm192_192 SIMD-optimised algorithm
	"src/schwaemm192192_simd/encrypt.c"

	# Schwaemm256_128 reference algorithm
	"src/schwaemm256128_ref/encrypt.c"

	# Schwaemm256_128 SIMD-optimised algorithm
	"src/schwaemm256128_simd/encrypt.c"

	# Schwaemm256_256 reference algorithm
	"src/schwaemm256256_ref/encrypt.c"

	# Schwaemm256_256 SIMD-optimised algorithm
	"src/schwaemm256256_simd/encrypt.c"

	# Pyjamask reference algorithm
	"src/pyjamask_ref/pyjamask.c"

	# Pyjamask SIMD-optimised algorithm
	"src/pyjamask_simd/pyjamask.c"

	# Pyjamask128 reference algorithm
	"src/pyjamask128_ref/encrypt.c"

	# Pyjamask128 SIMD-optimised algorithm
	"src/pyjamask128_simd/encrypt.c"

	# Pyjamask96 reference algorithm
	"src/pyjamask96_ref/encrypt.c"

	# Pyjamask96 SIMD-optimised algorithm
	"src/pyjamask96_simd/encrypt.c"
)

target_include_directories(testbench PRIVATE
	# Require specifying directory to avoid ref/SIMD mixups.
	"src/"
)

# MSVC does not require linking the math library for sqrt, but GCC etc do.
IF(NOT MSVC)
	target_link_libraries(testbench PRIVATE m)
ENDIF()
