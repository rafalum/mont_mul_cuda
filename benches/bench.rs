use criterion::{criterion_group, criterion_main, Criterion};

use mont_mul_cuda::*;
use mont_mul_cuda::utils::*;


fn criterion_benchmark(c: &mut Criterion) {

    // Sample two random G1 points in projective form and convert to affine
    let num_points = 20000000;
    let points = generate_points(num_points);

    let mut group = c.benchmark_group("CUDA");
    group.sample_size(10);


    // Ensure batches is defined, assuming it is for demonstration purposes
    let name = format!("montmul_raw_{}", num_points);
    group.bench_function(&name, |b| {
        b.iter(|| {
            // Call the CUDA function with the prepared Storage inputs
            let _result = montmul_era_wrapper(&points.as_slice(), num_points);
        })
    });

    group.finish();
}

// Define criterion group and main function
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
