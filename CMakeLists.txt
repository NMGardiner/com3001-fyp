cmake_minimum_required(VERSION 3.0.0)
project(com3001-fyp C)

#
# Compile options must come before add_executable()!!!
#

add_compile_options(
	#-std=c99
	-Wall
	#-fsanitize=address,undefined
)

IF (WIN32)
# -mavx2 equivalent for MSVC.
# Left out for now, since enabling it seems to make the optimised code run slower?
# AVX2 instructions like vpunpckldq are still in the disassembly whether the flag is
# added or not, so I can't figure out what's changing.
add_compile_options(/arch:AVX2)
ELSE()
# MSVC doesn't like these flags.
add_compile_options(-Wextra -Wshadow -mfpu=neon)
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

	# Pyjamask128 reference algorithm
	"src/pyjamask128_ref/encrypt.c"
	"src/pyjamask128_ref/pyjamask.c"

	# Pyjamask128 SIMD-optimised algorithm
	"src/pyjamask128_simd/encrypt.c"
	"src/pyjamask128_simd/pyjamask.c"
)

target_include_directories(testbench PRIVATE
	# Require specifying directory to avoid ref/SIMD mixups.
	"src/"
)