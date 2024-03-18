#define DEVICE_INLINE __device__ __forceinline__
  
#include <cstdint>
#include "stdio.h"
#include <cuda_runtime.h>
#include "ptx.cuh"

#define TLC 12
#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))


template <unsigned LIMBS_COUNT> struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) ff_storage {
  static constexpr unsigned LC = LIMBS_COUNT;
  uint32_t limbs[LIMBS_COUNT];
};

static constexpr ff_storage<TLC> MODULUS = {0x1a0111ea, 0x397fe69a, 0x4b1ba7b6, 0x434bacd7, 0x64774b84, 0xf38512bf, 
                                            0x6730d2a0, 0xf6b0f624, 0x1eabfffe, 0xb153ffff, 0xb9feffff, 0xffffaaab};
static constexpr uint32_t INV = 0xfffcfffd;

typedef ff_storage<TLC> storage;

extern "C" {
  void montmul_raw(const storage *points, storage *ret, uint32_t num_points);
}