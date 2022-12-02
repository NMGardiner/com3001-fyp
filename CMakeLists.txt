cmake_minimum_required(VERSION 3.0.0)
project(com3001-fyp C)

add_executable(testbench
	src/main.c
	# Esch384 reference
	"src/esch384-ref/hash.c"
	"src/esch384-ref/sparkle_ref.c"
	# Esch384 SIMD
	"src/esch384-simd/hash.c"
	"src/esch384-simd/sparkle_simd.c"
)

target_include_directories(testbench PRIVATE
	# Esch384 reference
	src/esch384-ref/
	# Esch384 SIMD
	src/esch384-simd/
)

add_compile_options(
	-std=c99 -Wall -Wextra -Wshadow -fsanitize=address,undefined
)