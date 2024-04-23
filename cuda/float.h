#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <cuda_runtime.h>
#include "era_bellman.h"

#define N 8

typedef struct {
    double limbs[N];
} dpf_storage;

template <unsigned LIMBS_COUNT> struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) ff_storage64 {
  static constexpr unsigned LC = LIMBS_COUNT;
  uint64_t limbs[LIMBS_COUNT];
};

typedef ff_storage64<N> storage8;


__device__ constexpr dpf_storage MODU = {0xeffffffffaaab, 0xfeb153ffffb9f, 0x6b0f6241eabff, 0x12bf6730d2a0f, 
                                         0x764774b84f385, 0x1ba7b6434bacd, 0x1ea397fe69a4b, 0x1a011};
                                         
constexpr uint64_t INV_D = 0x3fffcfffcfffd;