#include "era_bellman.h"

#define CUDA_CHECK(call) if((errorState=call)!=0) { cudaError("Call \"" #call "\" failed.", __FILE__, __LINE__); return errorState; }


static __device__ __forceinline__ void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) { 
#pragma unroll
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
}

static DEVICE_INLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
}

static DEVICE_INLINE void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi) {
    constexpr uint32_t n = TLC;
#pragma unroll
    for (size_t i = 0; i < n - 2; i += 2) {
      odd[i] = ptx::madc_lo_cc(a[i], bi, odd[i + 2]);
      odd[i + 1] = ptx::madc_hi_cc(a[i], bi, odd[i + 3]);
    }
    odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
  }


static DEVICE_INLINE void mad_n_redc(uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, bool first = false) {
    constexpr uint32_t n = TLC;
    constexpr auto modulus = MODULUS;
    const uint32_t *const MOD = modulus.limbs;
    if (first) {
      mul_n(odd, a + 1, bi);
      mul_n(even, a, bi);
    } else {
      even[0] = ptx::add_cc(even[0], odd[1]);
      madc_n_rshift(odd, a + 1, bi);
      cmad_n(even, a, bi);
      odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }
    uint32_t mi = even[0] * INV;
    cmad_n(odd, MOD + 1, mi);
    cmad_n(even, MOD, mi);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
  }


__global__ void montmul_raw_kernel(const storage &a_in, const storage &b_in, storage &r_in) {
    constexpr uint32_t n = TLC;
    constexpr auto modulus = MODULUS;
    const uint32_t *const MOD = modulus.limbs;
    const uint32_t *a = a_in.limbs;
    const uint32_t *b = b_in.limbs;
    uint32_t *even = r_in.limbs;
    __align__(8) uint32_t odd[n + 1];
    size_t i;
#pragma unroll
    for (i = 0; i < n; i += 2) {
      mad_n_redc(&even[0], &odd[0], a, b[i], i == 0);
      mad_n_redc(&odd[0], &even[0], a, b[i + 1]);
    }
    // merge |even| and |odd|
    even[0] = ptx::add_cc(even[0], odd[1]);
#pragma unroll
    for (i = 1; i < n - 1; i++)
      even[i] = ptx::addc_cc(even[i], odd[i + 1]);
    even[i] = ptx::addc(even[i], 0);
    // final reduction from [0, 2*mod) to [0, mod) not done here, instead performed optionally in mul_device wrapper
  }


void montmul_raw(const storage &a_host, const storage &b_host, storage &r_host) {
    printf("montmul_raw\n");
    
    storage *a_device, *b_device, *r_device;

    cudaMalloc(&a_device, sizeof(storage));
    cudaMalloc(&b_device, sizeof(storage));
    cudaMalloc(&r_device, sizeof(storage));

    // Copy data from host to device
    cudaMemcpy(a_device, &a_host, sizeof(storage), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, &b_host, sizeof(storage), cudaMemcpyHostToDevice);

    // Launch the kernel
    montmul_raw_kernel<<<1, 32>>>(*a_device, *b_device, *r_device);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&r_host, r_device, sizeof(storage), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(r_device);
}
