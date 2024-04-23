#include "era_bellman.h"
#include <inttypes.h>

#define CUDA_CHECK(call) if((errorState=call)!=0) { cudaError("Call \"" #call "\" failed.", __FILE__, __LINE__); return errorState; }

void print_device_properties(){
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);

  printf("SM count: %d\n", properties.multiProcessorCount);
  printf("Max blocks per SM: %d\n", properties.maxBlocksPerMultiProcessor);
  printf("Max threads per block: %d\n", properties.maxThreadsPerBlock);
  printf("Max threads per SM: %d\n", properties.maxThreadsPerMultiProcessor);
  printf("Max shared memory per block: %zu\n", properties.sharedMemPerBlock);
  printf("Max shared memory per SM: %zu\n", properties.sharedMemPerMultiprocessor);
  printf("Max registers per block: %d\n", properties.regsPerBlock);
  printf("Max registers per SM: %d\n", properties.regsPerMultiprocessor);
  printf("Max threads per warp: %d\n", properties.warpSize);
}

template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> struct carry_chain {
  unsigned index;

  constexpr __host__ __device__ __forceinline__ carry_chain() : index(0) {}

  __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::add(x, y);
    else if (index == 1 && !CARRY_IN)
      return ptx::add_cc(x, y);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::addc_cc(x, y);
    else
      return ptx::addc(x, y);
  }


  __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::sub(x, y);
    else if (index == 1 && !CARRY_IN)
      return ptx::sub_cc(x, y);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::subc_cc(x, y);
    else
      return ptx::subc(x, y);
  }

};

template <bool SUBTRACT, bool CARRY_OUT> static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(const storage &xs, const storage &ys, storage &rs) {
    const uint32_t *x = xs.limbs;
    const uint32_t *y = ys.limbs;
    uint32_t *r = rs.limbs;
    carry_chain<CARRY_OUT ? TLC + 1 : TLC> chain;
#pragma unroll
    for (unsigned i = 0; i < TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    if (!CARRY_OUT)
      return 0;
    return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
  }

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

static DEVICE_INLINE void montmul_era(storage *ret, const uint32_t *a, const uint32_t *b) {

  const storage modulus = MODULUS;

  uint32_t *even = ret->limbs;
  __align__(8) uint32_t odd[TLC + 1];
  size_t i;
  #pragma unroll
  for (i = 0; i < TLC; i += 2) {
    mad_n_redc(&even[0], &odd[0], a, b[i], i == 0);
    mad_n_redc(&odd[0], &even[0], a, b[i + 1]);
  }
  // merge |even| and |odd|
  even[0] = ptx::add_cc(even[0], odd[1]);
  #pragma unroll
  for (i = 1; i < TLC - 1; i++)
    even[i] = ptx::addc_cc(even[i], odd[i + 1]);
  even[i] = ptx::addc(even[i], 0);

  storage rs;
  add_sub_limbs_device<true, true>(*ret, modulus, rs);
}


__global__ void montmul_era_kernel(storage *results, const storage *points, uint32_t num_points) {


    uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t globalStride = blockDim.x * gridDim.x;

    for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride) {
      
      storage r_in;

      montmul_era(&r_in, points[j].limbs, points[j + 1].limbs);

      results[j / 2] = r_in;

    }
  }

void montmul_era(storage *ret, const storage *points, uint32_t num_points) {

    // print_device_properties();
    bool timing = true;

    // init memory
    storage *pointsPtrGPU;
    storage *retPtrGPU;

    cudaMalloc(&pointsPtrGPU, sizeof(storage) * num_points);
    cudaMalloc(&retPtrGPU, sizeof(storage) * num_points / 2);

    cudaMemcpy(pointsPtrGPU, points, sizeof(storage) * num_points, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    // Launch the kernel
    montmul_era_kernel<<<40, 384>>>(retPtrGPU, pointsPtrGPU, num_points);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    if (timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time: %f ms\n", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy result back to host
    cudaMemcpy(ret, retPtrGPU, sizeof(storage) * num_points / 2, cudaMemcpyDeviceToHost);
}

int main() {
	// Sample random points
  int num_points = 20000000;
	storage* points = (storage*)malloc(num_points * sizeof(storage));

	points[0] = {2434398911, 1341486800, 2149629466, 760351369, 865586749, 302494279, 3012983145, 950309675, 3687163001, 311611070, 4166041132, 3633413113};
	points[1] = {1366576235, 909555713, 1431863607, 3937335020, 3339380049, 2503284124, 1569754050, 610316959, 2201712813, 2217731649, 322256437, 2053650267};
	
	storage* result = (storage*)malloc(num_points / 2 * sizeof(storage));

  for (int i = 2; i < num_points; i+=2) {
    points[i] = points[0];
    points[i + 1] = points[1];
  }

	montmul_era(result, points, num_points);

	for(int i = 0; i < 12; i++) {
    printf("%u\n", result[0].limbs[i]);
  }

	return 0;

}

