#include "era_bellman.h"
#include <inttypes.h>

#define CUDA_CHECK(call) if((errorState=call)!=0) { cudaError("Call \"" #call "\" failed.", __FILE__, __LINE__); return errorState; }

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

template <bool SUBTRACT, bool CARRY_OUT> static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(storage *rs, const storage *xs, const storage *ys) {
    const uint32_t *x = xs->limbs;
    const uint32_t *y = ys->limbs;
    uint32_t *r = rs->limbs;
    carry_chain<CARRY_OUT ? TLC + 1 : TLC> chain;
#pragma unroll
    for (unsigned i = 0; i < TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    if (!CARRY_OUT)
      return 0;
    return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
  }

static DEVICE_INLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) { 
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

static DEVICE_INLINE storage montmul_era(const uint32_t *a, const uint32_t *b) {

	storage ret;
    const storage modulus = MODULUS;

	uint32_t *even = ret.limbs;
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

  	add_sub_limbs_device<true, true>(&ret, &ret, &modulus);
	return ret;
}


__global__ void montmul_era_kernel(storage *results, const storage *points, uint32_t num_points) {


    uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t globalStride = blockDim.x * gridDim.x;

    for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride) {
      
      results[j / 2] = montmul_era(points[j].limbs, points[j + 1].limbs);

    }
}

void montmul_era(storage *ret, const storage *points, uint32_t num_points) {

    // print_device_properties();
    bool timing = false;

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
    montmul_era_kernel<<<1024, 64>>>(retPtrGPU, pointsPtrGPU, num_points);

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



void montmul_era_piped(storage *ret, const storage *points, uint32_t num_points) {

    cudaFree(0);
    cudaFree(0);

    cudaStream_t memoryStreamHostToDevice, memoryStreamDeviceToHost, runStream;
    cudaStreamCreate(&memoryStreamHostToDevice);
    cudaStreamCreate(&memoryStreamDeviceToHost);
    cudaStreamCreate(&runStream);

    // print_device_properties();

    // init memory
    storage *pointsRegionA, *pointsRegionB;
    storage *resultRegionA, *resultRegionB;

    cudaMalloc(&pointsRegionA, sizeof(storage) * num_points);
    cudaMalloc(&pointsRegionB, sizeof(storage) * num_points);
    cudaMalloc(&resultRegionA, sizeof(storage) * (num_points / 2));
    cudaMalloc(&resultRegionB, sizeof(storage) * (num_points / 2));

    printf("Allocated memory\n");

    // Initial configuration
    cudaMemcpy(resultRegionA, points, sizeof(storage) * (num_points / 2), cudaMemcpyHostToDevice);
    cudaMemcpy(pointsRegionB, points, sizeof(storage) * num_points, cudaMemcpyHostToDevice);

    for (int i = 0; i < 2; i++) {

        if(i % 2 == 0) {
            cudaMemcpyAsync(pointsRegionA, points, sizeof(storage) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            montmul_era_kernel<<<1024, 64, 0, runStream>>>(resultRegionB, pointsRegionB, num_points);
            cudaMemcpyAsync(ret, resultRegionA, sizeof(storage) * (num_points / 2), cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
        } else {
            cudaMemcpyAsync(pointsRegionB, points, sizeof(storage) * num_points, cudaMemcpyHostToDevice, memoryStreamHostToDevice);
            montmul_era_kernel<<<1024, 64, 0, runStream>>>(resultRegionA, pointsRegionA, num_points);
            cudaMemcpyAsync(ret, resultRegionB, sizeof(storage) * (num_points / 2), cudaMemcpyDeviceToHost, memoryStreamDeviceToHost);
        }

        // Wait for the GPU to finish
        cudaStreamSynchronize(memoryStreamHostToDevice); 
        cudaStreamSynchronize(runStream);
        cudaStreamSynchronize(memoryStreamDeviceToHost);

        // Copy result back to host
        //cudaMemcpy(ret, retPtrGPU, sizeof(storage) * (num_points / 2), cudaMemcpyDeviceToHost);

    }

    // Destroy streams
    cudaStreamDestroy(memoryStreamDeviceToHost);
    cudaStreamDestroy(memoryStreamHostToDevice);
    cudaStreamDestroy(runStream);

    printf("Destroyed streams\n");

    // Free memory 
    cudaFree(pointsRegionA);
    cudaFree(pointsRegionB);
    cudaFree(resultRegionA);
    cudaFree(resultRegionB);
}

int main() {
	// Sample random points
    int num_points = 1024 * 64;

    // Allocate page-locked memory for points
	storage* points;
    cudaHostAlloc(&points, num_points * sizeof(storage), cudaHostAllocDefault);

    // Allocate page-locked memory for results
	storage* results;
    cudaHostAlloc(&results, (num_points / 2) * sizeof(storage), cudaHostAllocDefault);

    points[0] = {2434398911, 1341486800, 2149629466, 760351369, 865586749, 302494279, 3012983145, 950309675, 3687163001, 311611070, 4166041132, 3633413113};
	points[1] = {1366576235, 909555713, 1431863607, 3937335020, 3339380049, 2503284124, 1569754050, 610316959, 2201712813, 2217731649, 322256437, 2053650267};

    for (int i = 2; i < num_points; i+=2) {
        points[i] = points[0];
        points[i + 1] = points[1];
    }

	montmul_era_piped(results, points, num_points);

	for(int i = 0; i < 12; i++) {
        printf("%u\n", results[0].limbs[i]);
    }

    // Free allocated memory
    cudaFreeHost(points);
    cudaFreeHost(results);

	return 0;

}

