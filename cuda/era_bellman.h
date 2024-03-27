#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  
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

typedef ff_storage<TLC> storage;

static constexpr ff_storage<TLC> MODULUS = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 
                                            0xf38512bf, 0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
                                            
static constexpr uint32_t INV = 0xfffcfffd;

extern "C" {
  void montmul_raw(storage *ret, const storage *points, uint32_t num_points);
}