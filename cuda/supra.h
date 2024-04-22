#include <cstdint>
#include "stdio.h"
#include <cuda_runtime.h>
#include "ptx.cuh"
#include "era_bellman.h"

#define __LIMB_BITS 32
#define __UINT_LIMBS 12

typedef struct {
    uint32_t arr[__UINT_LIMBS];
} __uint_t;

typedef uint32_t __limb_t;
typedef uint64_t __llimb_t;

static constexpr __uint_t MODULUS_UINT = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 
                                          0xf38512bf, 0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
                                          
static constexpr uint32_t INV_UINT = 0xfffcfffd;


extern "C" {
  void montmul_supra(storage *ret, const storage *points, uint32_t num_points);
}
