#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 2

typedef union {
    double value;
    struct {
        unsigned long mantissa : 52;
        unsigned int exponent : 11;
        unsigned int sign : 1;
    } parts;
} double_breakdown;


void printBinary(unsigned long n, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        printf("%lu", (n >> i) & 1);
    }
    printf("\n");
}

void printDouble(double d) {
    double_breakdown db = { .value = d };
    printf("Value: %f\n", db.value);
    printf("Mantissa: ");
    printBinary(db.parts.mantissa, 52);
    printf("Exponent: ");
    printBinary(db.parts.exponent, 11);
    printf("Sign: ");
    printBinary(db.parts.sign, 1);
    printf("\n");
}

__global__ void fmaRzKernel(double a, double b, double c, double *result) {
    *result = __fma_rz(a, b, c);
}


void int_full_product(unsigned long *c_hi, unsigned long *c_lo, double a, double b) {

    /*
        Implementation of DPF Arithmetic: https://ieeexplore.ieee.org/document/8464792
    */

    double c1 = pow(2, 104);
    double c2 = pow(2, 104) + pow(2, 52);

    // printf("C1\n");
    // printDouble(c1);

    // printf("C2\n");
    // printDouble(c2);

    double p_hi;
    double *d_p_hi;
    cudaMalloc(&d_p_hi, sizeof(double));
    fmaRzKernel<<<1, 1>>>(a, b, c1, d_p_hi);
    cudaMemcpy(&p_hi, d_p_hi, sizeof(double), cudaMemcpyDeviceToHost);
    
    // printf("p_hi\n");
    // printDouble(p_hi);

    double sub = c2 - p_hi;
    // printf("Sub\n");
    // printDouble(sub);
    
    double p_lo;
    double *d_p_lo;
    cudaMalloc(&d_p_lo, sizeof(double));
    fmaRzKernel<<<1, 1>>>(a, b, sub, d_p_lo);
    cudaMemcpy(&p_lo, d_p_lo, sizeof(double), cudaMemcpyDeviceToHost);

    // printf("p_lo\n");
    // printDouble(p_lo);

    double_breakdown c_hi_db = { .value = p_hi };
    double_breakdown c_lo_db = { .value = p_lo };

    *c_hi = c_hi_db.parts.mantissa;
    *c_lo = c_lo_db.parts.mantissa;

    return;
}

void sampled_product(unsigned long *column_sums, double *a, double *b) {

    unsigned long c_hi, c_lo;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int_full_product(&c_hi, &c_lo, a[i], b[j]);
            column_sums[i + j] += c_lo;
            column_sums[i + j + 1] += c_hi;
        }
    }

    return;
}

int main() {

    // multiplicand 1
    double a[N] = {35184372088837, 1161084346041349};

    // multiplicand 2
    double b[N] = {549755813894, 549755944966};

    unsigned long column_sums[2 * N];
    sampled_product(column_sums, a, b);

    for(int i = 0; i < 2 * N; i++) {
        printf("Column sum %d: %lu\n", i, column_sums[i]);
    }
    
    return 0;
}
