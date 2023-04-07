# com3001-fyp

## TODO
- [X] Clean up CMakeLists.txt, especially platform macros
- [X] Clean up platform_defines.h, can probably replace with feature macros
- [ ] Add drop-in implementations at the end

## Build Instructions

### Unix/Linux

Note: These instructions have only been tested on the following platforms:
- Ubuntu 22.04 WSL2 (GCC 11.3.0)
- Raspbian 11 (GCC 10.2.1)

They may however work on other platforms e.g. Cygwin.

Clone the repo, and `cd` into the project root directory (containing CMakeLists.txt).

Build the project using the following CMake commands:  
`cmake -DCMAKE_BUILD_TYPE=Release -S . -B out`  
`cmake --build out`

This will build the testbench executable, which will be located in `out/testbench`.

### Windows

Note: These instructions have been tested on the following platform:
- Windows 11 (Visual Studio 17.5.3, MSVC 14.35.32215)

Clone the repo, and `cd` into the project root directory (containing CMakeLists.txt).

Build the project using the following CMake commands:  
`cmake -S . -B out`  
`cmake --build out --config Release`

This will build the testbench executable, which will be located in `out\Release\testbench.exe`.

## Running

From the command line, run the testbench executable built in the previous section.
This will benchmark the performance of all 8 algorithms in the testbench against the reference
implementations, and print the timing results.