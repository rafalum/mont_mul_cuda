#include "supra.h"

static __device__ void montmul(__uint_t *ret, const __uint_t *a, const __uint_t *b, const __uint_t *p, uint32_t p0) {
	__limb_t mask, borrow, mx, hi, tmp[__UINT_LIMBS + 1], carry;
	__llimb_t limbx;
	size_t i, j;

	mx = b->arr[0];
	hi = 0;
	for (i = 0; i < __UINT_LIMBS; i++) {
		limbx = (mx * (__llimb_t) a->arr[i]) + hi;
		tmp[i] = (__limb_t) limbx;
		hi = (__limb_t) (limbx >> __LIMB_BITS);
	}
	mx = p0 * tmp[0];
	tmp[i] = hi;


	for (carry=0, j=0; ; ) {
		limbx = (mx * (__llimb_t) p->arr[0]) + tmp[0];
		hi = (__limb_t) (limbx >> __LIMB_BITS);
		for (i = 1; i < __UINT_LIMBS; i++) {
			limbx = (mx * (__llimb_t) p->arr[i] + hi) + tmp[i];
			tmp[i - 1] = (__limb_t) limbx;
			hi = (__limb_t) (limbx >> __LIMB_BITS);
		}
		limbx = tmp[i] + (hi + (__llimb_t) carry);
		tmp[i - 1] = (__limb_t) limbx;
		carry = (__limb_t) (limbx >> __LIMB_BITS);

		if (++j == __UINT_LIMBS)
			break;

		mx = b->arr[j];
		hi = 0;
		for (i = 0; i < __UINT_LIMBS; i++) {
			limbx = (mx * (__llimb_t) a->arr[i] + hi) + tmp[i];
			tmp[i] = (__limb_t) limbx;
			hi = (__limb_t) (limbx >> __LIMB_BITS);
		}
		mx = p0 * tmp[0];
		limbx = hi + (__llimb_t) carry;
		tmp[i] = (__limb_t) limbx;
		carry = (__limb_t) (limbx >> __LIMB_BITS);
	}

	for (size_t i = 0; i < __UINT_LIMBS; i++) {
		ret->arr[i] = tmp[i];
	}

	borrow = 0;
	for (i = 0; i < __UINT_LIMBS; i++) {
		limbx = tmp[i] - (p->arr[i] + (__llimb_t) borrow);
		ret->arr[i] = (__limb_t) limbx;
		borrow = (__limb_t) (limbx >> __LIMB_BITS) & 1;
	}

	mask = carry - borrow;

	for(i = 0; i < __UINT_LIMBS; i++)
		ret->arr[i] = (ret->arr[i] & ~mask) | (tmp[i] & mask);
	
}

__global__ void montmul_supra_kernel(__uint_t *results, const __uint_t *points, uint32_t num_points) {

    const __uint_t p = MODULUS_UINT;
    uint32_t p0 = INV_UINT;

    uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t globalStride = blockDim.x * gridDim.x;

    for (uint32_t j = 2 * globalThreadId; j < num_points; j += 2 * globalStride) {

        __uint_t r_in;

        montmul(&r_in, &points[j], &points[j + 1], &p, p0);
        results[j / 2] = r_in;
    }
}

void montmul_supra(__uint_t *ret, const __uint_t *points, uint32_t num_points) {

    bool timing = false;

    // init memory
    __uint_t *pointsPtrGPU;
    __uint_t *retPtrGPU;

    cudaMalloc(&pointsPtrGPU, sizeof(__uint_t) * num_points);
    cudaMalloc(&retPtrGPU, sizeof(__uint_t) * num_points / 2);

    cudaMemcpy(pointsPtrGPU, points, sizeof(__uint_t) * num_points, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    montmul_supra_kernel<<<40, 1024>>>(retPtrGPU, pointsPtrGPU, num_points);

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
    cudaMemcpy(ret, retPtrGPU, sizeof(__uint_t) * num_points / 2, cudaMemcpyDeviceToHost);
}

int main() {
	// Sample random points
	int num_points = 20000000;
	__uint_t* points = (__uint_t*) malloc(num_points * sizeof(__uint_t));

	points[0] = {2434398911, 1341486800, 2149629466, 760351369, 865586749, 302494279, 3012983145, 950309675, 3687163001, 311611070, 4166041132, 3633413113};
	points[1] = {1366576235, 909555713, 1431863607, 3937335020, 3339380049, 2503284124, 1569754050, 610316959, 2201712813, 2217731649, 322256437, 2053650267};
	
	__uint_t* result = (__uint_t*) malloc(num_points / 2 * sizeof(__uint_t));

	for (int i = 2; i < num_points; i+=2) {
		points[i] = points[0];
		points[i + 1] = points[1];
  	}

	montmul_supra(result, points, num_points);

	for(int i = 0; i < 12; i++) {
		printf("%u\n", result[0].arr[i]);
	}


	return 0;

}
