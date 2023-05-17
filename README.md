# com3001-fyp

## Introduction

This repository contains the source code the my final year project, in the `src/`
directory. In this directory, `main.c` contains the code for the testing environment
which performs performance testing on the reference and SIMD algorithms. The subfolders
of `src/` contain the various implementations of the Sparkle Suite and Pyjamask AEAD
algorithms tested in this project. All reference implementations have the `_ref` suffix,
and all SIMD implementations have the `_simd` suffix.

Where necessary to avoid function naming collisions, names have been given a prefix
specific to the algorithm and implementation. For example, the Esch256 reference
algorithm has the `e256ref_` prefix, or `e256simd_` for the SIMD implementation. The
Sparkle permutation has been separated into a separate subfolder (`sparkle_ref` and
`sparkle_simd` respectively) to share it between the Sparkle Suite algorithms. Likewise,
the Pyjamask block ciphers are in a separate `pyjamask_ref`/`pyjamask_simd` subfolder for
the same reason.

The `sparkle_simd` subfolder also contains a file called `alzette_impls.h`, which contains
a number of experiments that were performed to find the optimal approach. Only one of
these is used in the final implementation, but the rest are preserved for sake of posterity.

Note: This repository is also hosted publicly on Github at https://github.com/NMGardiner/com3001-fyp.

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

## License and Disclaimer

The reference C implementations of all algorithms used in this project are licensed
as free software, with the Sparkle Suite and Pyjamask block cipher under GPLv3 and
the OCB portion of Pyjamask-AEAD under Unilicense. For this reason, these modified
SIMD implementations are licensed under GPLv3 as well (see LICENSE.md).

These modified implementations have not undergone security testing, so no assumptions
should be made as to the security provided. The following disclaimer applies:

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. This program is distributed in the hope that
it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. You should have received a
copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>. 