cmake_minimum_required(VERSION 3.0.0)
project(com3001-fyp C)

add_executable(testbench
	# Testbench
	"src/main.c"

	# Sparkle reference algorithm
	"src/sparkle_ref/sparkle_ref.c"

	# Sparkle SIMD-optimised algorithm
	"src/sparkle_simd/sparkle_simd.c"

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
)

target_include_directories(testbench PRIVATE
	# Require specifying directory to avoid ref/SIMD mixups.
	"src/"
)

add_compile_options(
	-std=c99 -Wall -Wextra -Wshadow -fsanitize=address,undefined -mavx
)