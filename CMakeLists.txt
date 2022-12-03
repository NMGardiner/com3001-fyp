cmake_minimum_required(VERSION 3.0.0)
project(com3001-fyp C)

add_executable(testbench
	# Testbench
	"src/main.c"

	# Sparkle reference
	"src/sparkle-ref/sparkle_ref.c"

	# Sparkle SIMD
	"src/sparkle-simd/sparkle_simd.c"

	# Esch384 reference
	"src/esch384-ref/hash.c"

	# Esch384 SIMD
	"src/esch384-simd/hash.c"

	# Schwaemm256_256 reference
	"src/schwaemm256_256-ref/encrypt.c"
)

target_include_directories(testbench PRIVATE
	# Require specifying directory to avoid ref/SIMD mixups.
	"src/"
)

add_compile_options(
	-std=c99 -Wall -Wextra -Wshadow -fsanitize=address,undefined
)