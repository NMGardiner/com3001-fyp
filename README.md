# com3001-fyp

## TODO
- [ ] Look at having a core Sparkle permutation shared between all Esch/Schwaemm implementations
- [ ] Look into more detailed timing for the benchmarking, as it's a small scale already.

## Notes
- E/P cores on the Intel chip could vary the results.
- Possible to lock the core clock on the RPi, perhaps also on the Intel?

## SIMD Optimisation Notes
Sparkle:
- Data dependency between steps, but can likely parallellize a single step.
-  ARXBox layer operates on each 64-bit block sequentially. 2/4 at once?