#include "float.h"

/*
        Adaptation of: https://ieeexplore.ieee.org/abstract/document/9139772
*/


typedef union {
    double value;
    struct {
        unsigned long mantissa : 52;
        unsigned int exponent : 11;
        unsigned int sign : 1;
    } parts;
} double_breakdown;


__device__ void printBinary(unsigned long n, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        printf("%lu", (n >> i) & 1);
    }
    printf("\n");
}


void uint_storage_to_dpf_storage(dpf_storage *d, storage *s) {

    int index = 0;
    int bits_consumed = 0;
    for(int i = 0; i < N; i++) {
        int bits_left = 52;
        unsigned long acc = 0;

        while(true) {
            uint64_t temp = s->limbs[index];
            if(i == N - 1) {
                uint64_t mask = (((uint64_t) 1 << 20) - 1) << bits_consumed;
                acc = (temp & mask) >> bits_consumed;
                d->limbs[i] = (double) acc;
                break;
            }
            else if(bits_left >= (32 - bits_consumed)) {
                uint64_t mask = (((uint64_t) 1 << (32 - bits_consumed)) - 1) << bits_consumed;
                acc += ((temp  & mask) >> bits_consumed) << (52 - bits_left);
                bits_left -= (32 - bits_consumed);
                bits_consumed = 0;
                index++;
            } 
            else {
                uint64_t mask = ((uint64_t) 1 << bits_left) - 1;
                acc += (temp & mask) << (52 - bits_left);
                bits_consumed += bits_left;
                break;
            }
        }
        d->limbs[i] = (double) acc;

    }
    return;
}

__device__ __forceinline__ uint64_t to_uint64(double d) {
    uint64_t result;
    uint64_t mask = 0xfffffffffffffull;

    asm("mov.b64 %0, %1;" : "=l"(result) : "d"(d));
    asm("and.b64 %0, %0, %1;" : "+l"(result) : "l"(mask));
    
    return result;
}


__global__ void int_full_product_kernel(unsigned long *c_hi, unsigned long *c_lo, double a, double b) {

    double c1 = pow(2, 100);
    double c2 = pow(2, 100) + pow(2, 52);

    double p_hi = __fma_rz(a, b, c1);
    
    double sub = c2 - p_hi;
    
    double p_lo = __fma_rz(a, b, sub);

    *c_hi = to_uint64(p_hi);
    *c_lo = to_uint64(p_lo);

    return;
}

__global__ void multi_precision_multiplication(unsigned long *c, double *a, double *b) {

    double c1 = pow(2, 104);
    double c2 = pow(2, 104) + pow(2, 52);

    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double p_hi = __fma_rz(a[j], b[i], c1);
            double sub = c2 - p_hi;
            double p_lo = __fma_rz(a[j], b[i], sub);

            c[i + j + 1] += to_uint64(p_hi);
            c[i + j] += to_uint64(p_lo);
        }
    }

    
    for(int i = 0; i < 2 * N - 1; i++) {
        c[i + 1] += c[i] >> 52;
        c[i] = c[i] & 0xfffffffffffffull;
    }
    
    return;
}

__global__ void montmul_float_kernel(unsigned long *c, double *a, double *b) {

    double c1 = pow(2, 104);
    double c2 = pow(2, 104) + pow(2, 52);

    const double *modulus = MODU.limbs;
    const uint64_t inv = INV_D;

    double bi = b[0];
    for(int j = 0; j < N; j++) {
        // S <- S + A * bi
        double p_hi = __fma_rz(a[j], bi, c1);
        double sub = c2 - p_hi;
        double p_lo = __fma_rz(a[j], bi, sub);

        c[j] += to_uint64(p_lo);
        c[j + 1] += to_uint64(p_hi);

    }

    unsigned long qi = (c[0] * inv) & 0xfffffffffffffull;

    for(int i = 0;;) {

        for (int j = 0; j < N; j++) {
            // S <- S + P * qi
            double p_hi = __fma_rz(modulus[j], qi, c1);
            double sub = c2 - p_hi;
            double p_lo = __fma_rz(modulus[j], qi, sub);

            c[j] += to_uint64(p_lo);
            c[j + 1] += to_uint64(p_hi);
        }

        for(int j = 0; j < N; j++) {
            // S <- S / R
            c[j] = c[j] >> 52;
            c[j] += c[j + 1] & 0xfffffffffffffull;
        }
        c[N] = c[N] >> 52;

        for(int k = 0; k < N + 1; k++) {
			printf("%u, ", c[k]);
		}
		printf("\n");

        if(++i == N ){
            break;
        }

        bi = b[i];
        for(int j = 0; j < N; j++) {
            // S <- S + A * bi
            double p_hi = __fma_rz(a[j], bi, c1);
            double sub = c2 - p_hi;
            double p_lo = __fma_rz(a[j], bi, sub);

            c[j] += to_uint64(p_lo);
            c[j + 1] += to_uint64(p_hi);
        }

        qi = (c[0] * inv) & 0xfffffffffffffull;

    }


}


void montmul_float(storage *ret, const dpf_storage *points, uint32_t num_points) {

    // init memory
    unsigned long *c_d;

    unsigned long c_h[N + 1];

    double *a_d;
    double *b_d;

    const double *a_h = points[0].limbs;
    const double *b_h = points[1].limbs;

    cudaMalloc(&c_d, sizeof(unsigned long) * (N + 1));

    cudaMalloc(&a_d, sizeof(double) * N);
    cudaMalloc(&b_d, sizeof(double) * N);

    cudaMemcpy(a_d, a_h, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(double) * N, cudaMemcpyHostToDevice);


    // Launch the kernel
    montmul_float_kernel<<<1, 1>>>(c_d, a_d, b_d);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    printf("Kernel finished\n");

    // Copy result back to host
    cudaMemcpy(c_h, c_d, sizeof(unsigned long) * (N + 1), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        printf("%lu\n", c_h[i]);
    }

}



int main() {

    // Sample random points
	storage points[2];
	points[0] = {2434398911, 1341486800, 2149629466, 760351369, 865586749, 302494279, 3012983145, 950309675, 3687163001, 311611070, 4166041132, 3633413113};
	points[1] = {1366576235, 909555713, 1431863607, 3937335020, 3339380049, 2503284124, 1569754050, 610316959, 2201712813, 2217731649, 322256437, 2053650267};

    dpf_storage points_dpf[2];

    uint_storage_to_dpf_storage(&points_dpf[0], &points[0]);
    uint_storage_to_dpf_storage(&points_dpf[1], &points[1]);



    storage result[1];

    montmul_float(result, points_dpf, 2);
	

	return 0;

}
